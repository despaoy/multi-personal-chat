"""MultiPersonal Chat System - 后端主应用

整合所有模块，创建 FastAPI 应用实例。
路由按领域拆分到 api/ 下各模块，通过 APIRouter 挂载。
"""

import inspect
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path

# 确保 backend 根目录在 Python 路径中，支持跨包导入
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 统一加载 backend/.env；显式注入的容器/进程变量优先。
from app.env import load_backend_env

load_backend_env()

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import import_module
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Read lifespan-owned resources through the module to observe current instances.
from app import config as _cfg

# Freeze after configured-key loading/migration publishes the effective key.
_STARTUP_ENV = dict(os.environ)

from app.readiness import ReadinessProbe, ReadinessProbeTimeout
from app.runtime import RuntimeContainer, get_runtime_container
from inference.context_budget import InputBudgetError
from inference.openai_protocol import ProviderResponseError
from services.model_usage import UsageRecordingError  # noqa: E402
from infra.deployment import validate_or_raise_for_startup
from infra.access_control import AccessControlManager
from infra.backup_manager import BackupManager
from infra.resource_pool import ConnectionPool, HttpClientPool
from infra.bounded_executor import BoundedThreadExecutor
from infra.environment import read_bool

logger = logging.getLogger("main")


def _initialize_database(database) -> None:
    """Initialize and probe either database adapter through its public contract."""
    if hasattr(database, "init"):
        database.init()
    database.execute_sql("SELECT 1")


async def _close_resource(label: str, resource, method_name: str) -> None:
    """Close an owned resource or lazy module callback; report failures to the exit stack."""
    try:
        if isinstance(resource, str):
            resource = import_module(resource)
        for attribute in method_name.split("."):
            if resource is None:
                return
            resource = getattr(resource, attribute)
        method = resource
        if inspect.iscoroutinefunction(method):
            result = await method()
        else:
            result = await asyncio.to_thread(method)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        logger.error("关闭%s失败: %s", label, type(exc).__name__)
        raise


@asynccontextmanager
async def _closing_resource(label: str, resource, method_name: str):
    """Use exit contexts so successive close failures retain earlier exceptions."""
    try:
        yield
    finally:
        await _close_resource(label, resource, method_name)


# ── 导入所有 API 路由 ──
from api.ask import router as ask_router
from api.auth import router as auth_router
from api.characters import router as characters_router
from api.claw import router as claw_router
from api.config import router as config_router
from api.enhanced import router as enhanced_router
from api.evaluation import router as evaluation_router
from api.experiments import router as experiments_router
from api.generate import router as generate_router
from api.integrations import router as integrations_router
from api.knowledge import router as knowledge_router
from api.loras import router as loras_router
from api.messages import router as messages_router
from api.models import router as models_router
from api.narrative import router as narrative_router
from api.preferences import router as preferences_router
from api.retrieval_eval import router as retrieval_eval_router
from api.router import router as lora_router_router
from api.stats import router as stats_router
from api.training import router as training_router
from api.user_data import router as user_data_router

# ═══════════════════════════════════════════
# 应用生命周期
# ═══════════════════════════════════════════
_LIFESPAN_RESOURCE_NAMES = (
    "connection_pool",
    "http_client_pool",
    "backup_mgr",
    "access_control_mgr",
)


def _clear_lifespan_resource_references(config_module) -> None:
    for name in _LIFESPAN_RESOURCE_NAMES:
        setattr(config_module, name, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _clear_lifespan_resource_references(_cfg)
    container = get_runtime_container(app)
    database = container.db
    postgres_mode = bool(container.is_pg_mode())

    validate_or_raise_for_startup(container.startup_env)
    from character.memory_llm import MemoryLlmConfig

    MemoryLlmConfig.from_env(container.startup_env)
    try:
        logger.info("🚀 MultiPersonal Chat System后端服务启动中（增强版）...")

        container.knowledge_scan_executor = BoundedThreadExecutor(
            name="knowledge-scan", max_workers=2, max_pending=4, default_timeout=30.0
        )

        # 初始化数据库
        try:
            _initialize_database(database)
            logger.info("✅ 数据库初始化完成 (%s)", "PostgreSQL" if postgres_mode else "SQLite")
        except Exception as e:
            logger.critical(f"❌ 数据库初始化失败，服务无法启动: {e}", exc_info=True)
            raise RuntimeError(f"数据库初始化失败: {e}") from e

        # Both SQLite resources must use the database selected by the runtime container.
        db_file = Path(database.db_path).expanduser().resolve() if not postgres_mode else None
        _cfg.http_client_pool = HttpClientPool(max_connections=100, request_timeout=120.0)
        if not postgres_mode:
            _cfg.connection_pool = ConnectionPool(str(db_file), max_size=20)
            configured_backup_dir = container.startup_env.get("BACKUP_DIR", "").strip()
            backup_dir = Path(configured_backup_dir).expanduser() if configured_backup_dir else db_file.parent / "backups"
            if not backup_dir.is_absolute():
                backup_dir = db_file.parent / backup_dir
            _cfg.backup_mgr = BackupManager(str(db_file), backup_dir=str(backup_dir))
            await _cfg.backup_mgr.start_scheduled_backup()
        _cfg.access_control_mgr = AccessControlManager(database)
        logger.info("✅ 资源池、备份调度与访问控制初始化完成")

        # 延迟重建向量索引（首次搜索时自动触发，避免启动时阻塞）
        # 见 api/knowledge.py search_knowledge 中的 _ensure_vector_index()

        from services.turn_completion import start_turn_completions

        start_turn_completions()
        from services.delivery_memory import ensure_delivery_memory_worker

        ensure_delivery_memory_worker(database)
        logger.info("✅ 增强版服务启动完成！")
        yield
    finally:
        logger.info("👋 服务关闭中，清理资源...")
        # Drain accepted work before closing its scheduler and database dependencies.
        resources = (
            ("评估任务", "evaluation.runtime_runner", "shutdown_generation_evaluations"),
            ("LoRA 训练任务", "training.task_manager", "shutdown_simple_lora_trainer"),
            ("RAG 意图任务", "api.knowledge", "shutdown_intent_tasks"),
            ("推理调度器", container.inference_runtime, "shutdown"),
            ("交付记忆任务", "services.delivery_memory", "shutdown_delivery_memory"),
            ("角色回写任务", "services.turn_completion", "shutdown_turn_completions"),
            ("后台长期记忆任务", "character.memory_llm", "shutdown_memory_enrichment"),
            ("知识库扫描", container.knowledge_scan_executor, "shutdown"),
            ("readiness probe", getattr(app.state, "readiness_probe", None), "shutdown"),
            ("SQLite 连接池", _cfg.connection_pool, "close"),
            ("HTTP 客户端池", _cfg.http_client_pool, "close"),
            ("备份调度器", _cfg.backup_mgr, "stop_scheduled_backup"),
            ("PostgreSQL 适配器", database if postgres_mode else None, "close"),
            ("模型管理器", "inference.model_manager", "_model_manager.shutdown"),
            ("vLLM 客户端", "inference.vllm_client", "close_shared_vllm_client"),
            ("熔断器注册表", _cfg.circuit_breaker_registry, "clear"),
            ("async Redis 客户端", "cache.redis_client", "close_async_redis"),
        )
        cleanup = AsyncExitStack()
        cleanup.callback(_clear_lifespan_resource_references, _cfg)
        cleanup.callback(setattr, container, "knowledge_scan_executor", None)
        for resource in reversed(resources):
            await cleanup.enter_async_context(_closing_resource(*resource))
        # Forward the active failure so cleanup errors retain its exception chain.
        await cleanup.__aexit__(*sys.exc_info())
        logger.info("✅ 资源清理完成")


# ═══════════════════════════════════════════
# 创建应用
# ═══════════════════════════════════════════
def _allowed_origins(env: Mapping[str, str] | None = None) -> list[str]:
    """解析 CORS 允许源，优先 ALLOWED_ORIGINS，旧键 CORS_ORIGINS 发 deprecation warning。"""
    source = os.environ if env is None else env
    configured = source.get("ALLOWED_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    legacy = source.get("CORS_ORIGINS")
    if legacy:
        logger.warning("CORS_ORIGINS 已弃用，请改用 ALLOWED_ORIGINS。旧键将在下个大版本移除。")
        return [origin.strip() for origin in legacy.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
    ]


# ── 安全中间件按当前应用的启动配置装配 ──
def _install_middleware(application: FastAPI, env: Mapping[str, str]) -> None:
    if read_bool(env, "SECURITY_MIDDLEWARE_ENABLED", True):
        from middleware.security import (
            AuditLogMiddleware,
            InputValidationMiddleware,
            RateLimitMiddleware,
            SecurityHeadersMiddleware,
            SecurityMiddleware,
        )

        # Starlette's last added middleware is outermost. Request order:
        # CORS -> audit -> security headers -> auth -> rate limit -> validation.
        application.add_middleware(InputValidationMiddleware)
        application.add_middleware(RateLimitMiddleware)
        application.add_middleware(SecurityMiddleware)
        application.add_middleware(SecurityHeadersMiddleware)
        application.add_middleware(AuditLogMiddleware)
        logger.info("✅ 安全中间件已启用（认证+限流+输入验证+审计+安全头）")
    # CORS is outermost so even authentication and rate-limit errors include CORS headers.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(env),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ── 挂载所有路由 ──
_ROUTERS = (
    stats_router,
    messages_router,
    narrative_router,
    generate_router,
    loras_router,
    training_router,
    knowledge_router,
    models_router,
    config_router,
    auth_router,
    user_data_router,
    enhanced_router,
    claw_router,
    integrations_router,
    evaluation_router,
    experiments_router,
    retrieval_eval_router,
    preferences_router,
    lora_router_router,
    characters_router,
    ask_router,
)


# ═══════════════════════════════════════════
# 根路由 & 健康检查
# ═══════════════════════════════════════════
async def root():
    return {
        "name": "MultiPersonal Chat System API (增强版)",
        "version": "2.0.0",
        "status": "running",
    }


async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


async def readiness_check(request: Request):
    """Return one cached dependency snapshot without initializing optional RAG."""
    probe = getattr(request.app.state, "readiness_probe", None)
    if not isinstance(probe, ReadinessProbe):
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "deps": {"database": False, "model": False},
                "details": {"probe": "not_configured"},
            },
        )
    try:
        snapshot = await probe.get()
    except ReadinessProbeTimeout as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "deps": {"database": False, "model": False},
                "details": {"probe": type(exc).__name__},
            },
        ) from exc
    if not snapshot["ready"]:
        raise HTTPException(
            status_code=503,
            detail=snapshot,
        )
    return {
        "status": "ready",
        "deps": snapshot["deps"],
        "details": snapshot["details"],
    }


# ═══════════════════════════════════════════
# 全局异常处理
# ═══════════════════════════════════════════
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, (ProviderResponseError, InputBudgetError, UsageRecordingError)):
        detail = exc.public_detail()
        logger.warning("Request failed: %s upstream_status=%s", detail["kind"], detail.get("upstreamStatus"))
        return JSONResponse(status_code=exc.http_status, content={
            "success": False, "error": detail["kind"], **detail,
        })
    logger.error("请求异常: %s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": "服务器内部错误",
        },
    )


def create_app(container: RuntimeContainer | None = None) -> FastAPI:
    """Assemble one FastAPI application around an explicit runtime container."""
    from inference.model_readiness import check_model_readiness

    runtime_container = container if container is not None else RuntimeContainer.default(startup_env=_STARTUP_ENV)
    application = FastAPI(
        title="MultiPersonal Chat System API (增强版)",
        description="MultiPersonal Chat System后端服务API - 高并发/高可靠/高安全增强版",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.runtime_container = runtime_container
    application.state.readiness_probe = ReadinessProbe(
        database_check=lambda: get_runtime_container(application).db.execute_sql("SELECT 1"),
        model_check=runtime_container.model_check or check_model_readiness,
        model_required=True,
    )
    _install_middleware(application, runtime_container.startup_env)
    for router in _ROUTERS:
        application.include_router(router)

    application.add_api_route("/", root, methods=["GET"])
    application.add_api_route("/health", health_check, methods=["GET"])
    application.add_api_route("/ready", readiness_check, methods=["GET"])
    for error_type in (ProviderResponseError, InputBudgetError, UsageRecordingError, Exception):
        application.add_exception_handler(error_type, global_exception_handler)
    return application


app = create_app()
