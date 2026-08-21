"""MultiPersonal Chat System - 后端主应用

整合所有模块，创建 FastAPI 应用实例。
路由按领域拆分到 api/ 下各模块，通过 APIRouter 挂载。
"""
import sys
from pathlib import Path
import os
import logging
import inspect
from collections.abc import Mapping

# 确保 backend 根目录在 Python 路径中，支持跨包导入
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 统一加载 backend/.env；显式注入的容器/进程变量优先。
from app.env import load_backend_env

load_backend_env()
_STARTUP_ENV = dict(os.environ)

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 应用配置 + 增强模块全局实例
# C-F1 fix: lifespan 管理的单例 (failover_mgr/backup_mgr/connection_pool/
# http_client_pool/access_control_mgr) 在 lifespan 中通过 `_cfg.xxx = ...`
# 重新赋值。这里只导入常量与导入时初始化的单例，不导入会被 lifespan 重赋值的单例，
# 避免读者误以为本模块中的本地名会跟随 lifespan 更新（实际它们仍指向 None）。
from app.config import (
    RESOURCE_POOL_AVAILABLE, BACKUP_MANAGER_AVAILABLE,
    FAILOVER_AVAILABLE, ACCESS_CONTROL_AVAILABLE,
)
from app.readiness import ReadinessProbe, ReadinessProbeTimeout
from app.runtime import RuntimeContainer, get_runtime_container
from infra.deployment import validate_or_raise_for_startup

logger = logging.getLogger("main")


def _initialize_database(database) -> None:
    """Initialize and probe either database adapter through its public contract."""
    if hasattr(database, "init"):
        database.init()
    database.execute_sql("SELECT 1")


async def _close_resource(label: str, resource, method_name: str) -> None:
    """Close one resource without blocking the event loop or later cleanup."""
    if resource is None:
        return
    try:
        method = getattr(resource, method_name)
        if inspect.iscoroutinefunction(method):
            result = await method()
        else:
            result = await asyncio.to_thread(method)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        logger.warning("关闭%s失败: %s", label, exc)

# ── 导入所有 API 路由 ──
from api.stats import router as stats_router
from api.messages import router as messages_router
from api.generate import router as generate_router
from api.loras import router as loras_router
from api.training import router as training_router
from api.knowledge import router as knowledge_router
from api.models import router as models_router
from api.config import router as config_router
from api.auth import router as auth_router
from api.user_data import router as user_data_router
from api.enhanced import router as enhanced_router
from api.claw import router as claw_router
from api.integrations import router as integrations_router
from api.evaluation import router as evaluation_router
from api.experiments import router as experiments_router
from api.retrieval_eval import router as retrieval_eval_router
from api.preferences import router as preferences_router
from api.router import router as lora_router_router
from api.characters import router as characters_router


# ═══════════════════════════════════════════
# 应用生命周期
# ═══════════════════════════════════════════
_LIFESPAN_RESOURCE_NAMES = (
    "connection_pool",
    "http_client_pool",
    "backup_mgr",
    "failover_mgr",
    "access_control_mgr",
)


def _clear_lifespan_resource_references(config_module) -> None:
    for name in _LIFESPAN_RESOURCE_NAMES:
        setattr(config_module, name, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # C-F1 fix: 原先用 `global x` 赋值只更新 main.py 模块命名空间，
    # 不会更新 app.config 模块中的同名变量。api/generate.py 等通过
    # `from app.config import failover_mgr` 在导入时绑定到 None，永远
    # 看不到 lifespan 中创建的实例。改为通过 `app.config.xxx = ...`
    # 显式赋值到 config 模块属性，所有通过 `from app import config`
    # 或 `import app.config` 访问的模块都能看到最新实例。
    # 注意：已通过 `from app.config import failover_mgr` 导入的模块仍持有
    # 旧的 None 引用，这些模块需改为 `from app import config` 后用
    # `config.failover_mgr` 访问（本次修复同步更新调用方）。
    import app.config as _cfg

    _clear_lifespan_resource_references(_cfg)
    container = get_runtime_container(app)
    database = container.db
    postgres_mode = bool(container.is_pg_mode())

    logger.info("🚀 MultiPersonal Chat System后端服务启动中（增强版）...")

    validate_or_raise_for_startup(container.startup_env)

    # 初始化数据库
    try:
        _initialize_database(database)
        logger.info("✅ 数据库初始化完成 (%s)", "PostgreSQL" if postgres_mode else "SQLite")
    except Exception as e:
        logger.critical(f"❌ 数据库初始化失败，服务无法启动: {e}", exc_info=True)
        raise RuntimeError(f"数据库初始化失败: {e}") from e

    # 初始化 Redis 缓存（可选，失败不影响服务）
    try:
        from cache.redis_client import get_redis, health_check
        if health_check():
            logger.info("✅ Redis 缓存连接正常")
        else:
            logger.warning("⚠️ Redis 缓存连接失败，将使用数据库直连模式")
    except Exception as e:
        logger.warning(f"Redis 缓存初始化跳过: {e}")

    if RESOURCE_POOL_AVAILABLE:
        try:
            from infra.resource_pool import ConnectionPool, HttpClientPool

            _cfg.http_client_pool = HttpClientPool(
                max_connections=100,
                request_timeout=120.0,
            )
            if not postgres_mode:
                db_path = str(getattr(database, "db_path", _BACKEND_ROOT / "qq_assistant.db"))
                _cfg.connection_pool = ConnectionPool(db_path, max_size=20)
            logger.info(
                "✅ 资源池初始化完成 (%s)",
                "HTTP" if postgres_mode else "SQLite + HTTP",
            )
        except Exception as e:
            logger.warning(f"资源池初始化失败: {e}")

    if BACKUP_MANAGER_AVAILABLE and not postgres_mode:
        try:
            from infra.backup_manager import BackupManager
            db_path = str(getattr(database, "db_path", _BACKEND_ROOT / "qq_assistant.db"))
            db_file = Path(db_path).expanduser()
            if not db_file.is_absolute():
                db_file = db_file.resolve()
            db_path = str(db_file)
            configured_backup_dir = container.startup_env.get("BACKUP_DIR", "").strip()
            backup_dir = (
                Path(configured_backup_dir).expanduser()
                if configured_backup_dir
                else db_file.parent / "backups"
            )
            if not backup_dir.is_absolute():
                backup_dir = db_file.parent / backup_dir
            _cfg.backup_mgr = BackupManager(db_path, backup_dir=str(backup_dir))
            await _cfg.backup_mgr.start_scheduled_backup()
            logger.info("✅ 备份管理器初始化完成")
        except Exception as e:
            logger.warning(f"备份管理器初始化失败: {e}")

    if FAILOVER_AVAILABLE:
        try:
            from infra.failover import FailoverManager, FailoverStrategy
            _cfg.failover_mgr = FailoverManager(strategy=FailoverStrategy.AUTO_FAILOVER)

            # I-2 fix: 不再注册 vLLM provider 到 FailoverManager。
            # VLLMClient 内部已有完整的实例健康检查（try_recover + UNHEALTHY 标记）、
            # 熔断器和故障转移能力，此处冗余注册只会导致两套系统重复做 /health 探测，
            # 且只注册一个 provider 无转移目标，check_and_failover() 永远返回 None。
            # 未来如需非 vLLM fallback（如 ollama），可在此处注册。

            # 无 provider 时不启动 HealthChecker，避免空转循环（每 10s 唤醒无意义）
            if _cfg.failover_mgr._providers:
                await _cfg.failover_mgr.start()
                logger.info("✅ 故障转移管理器已启动（含 HealthChecker）")
            else:
                logger.info("✅ 故障转移管理器已初始化（无 provider，跳过 HealthChecker 启动；vLLM 由 VLLMClient 内部管理）")
        except Exception as e:
            logger.warning(f"故障转移管理器初始化失败: {e}")

    if ACCESS_CONTROL_AVAILABLE:
        try:
            from infra.access_control import AccessControlManager
            _cfg.access_control_mgr = AccessControlManager(database)
            logger.info("✅ 访问控制管理器初始化完成（统一 DB adapter）")
        except Exception as e:
            logger.warning(f"访问控制管理器初始化失败: {e}")

    # 延迟重建向量索引（首次搜索时自动触发，避免启动时阻塞）
    # 见 api/knowledge.py search_knowledge 中的 _ensure_vector_index()

    logger.info("✅ 增强版服务启动完成！")
    try:
        yield
    finally:
        logger.info("👋 服务关闭中，清理资源...")
        try:
            from evaluation.runtime_runner import shutdown_generation_evaluations

            await shutdown_generation_evaluations()
        except Exception as e:
            logger.warning("关闭评估任务失败: %s", e)
        try:
            from training.task_manager import shutdown_simple_lora_trainer

            await shutdown_simple_lora_trainer()
        except Exception as e:
            logger.warning("关闭 LoRA 训练任务失败: %s", e)
        try:
            from api.knowledge import shutdown_intent_tasks

            await shutdown_intent_tasks()
        except Exception as e:
            logger.warning("关闭 RAG 意图任务失败: %s", e)
        await _close_resource(
            "readiness probe",
            getattr(app.state, "readiness_probe", None),
            "shutdown",
        )
        await _close_resource("推理调度器", container.inference_runtime, "shutdown")
        await _close_resource("SQLite 连接池", _cfg.connection_pool, "close")
        await _close_resource("HTTP 客户端池", _cfg.http_client_pool, "close")
        await _close_resource("备份调度器", _cfg.backup_mgr, "stop_scheduled_backup")
        await _close_resource("故障转移管理器", _cfg.failover_mgr, "stop")
        # PG 模式下显式关闭引擎连接池与 SyncPgAdapter 后台事件循环。
        if postgres_mode:
            await _close_resource("PostgreSQL 适配器", database, "close")
        # C6 fix: 关闭所有模型 Provider 持有的 httpx 客户端与 GPU 句柄。
        try:
            from inference.model_manager import get_model_manager

            await _close_resource("模型管理器", get_model_manager(), "shutdown")
        except Exception as e:
            logger.warning(f"获取模型管理器失败: {e}")
        # C10 fix: 关闭 generate.py 中独立的 vLLM 客户端连接池
        try:
            from api.generate import close_vllm_client
            await close_vllm_client()
        except Exception as e:
            logger.warning(f"关闭 vLLM 客户端失败: {e}")
        # 关闭 bot 模块中共享的 HTTP 客户端（RAG 搜索 + Ollama 推理）
        try:
            from bot.bot import _close_bot_http_clients
            await _close_bot_http_clients()
        except Exception as e:
            logger.warning(f"关闭 bot HTTP 客户端失败: {e}")
        if _cfg.circuit_breaker_registry is not None:
            await _close_resource(
                "熔断器注册表",
                _cfg.circuit_breaker_registry,
                "clear",
            )
        # 关闭共享的 async Redis 客户端
        try:
            from cache.redis_client import close_async_redis
            await close_async_redis()
        except Exception as e:
            logger.warning(f"关闭 async Redis 客户端失败: {e}")
        # 关闭共享的 sync Redis 客户端与连接池
        try:
            from cache.redis_client import close_sync_redis

            await asyncio.to_thread(close_sync_redis)
        except Exception as e:
            logger.warning(f"关闭 sync Redis 客户端失败: {e}")
        _clear_lifespan_resource_references(_cfg)
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


# ── 安全中间件（可通过环境变量控制开关） ──
_SECURITY_ENABLED = os.getenv("SECURITY_MIDDLEWARE_ENABLED", "true").lower() == "true"

def _install_middleware(
    application: FastAPI,
    env: Mapping[str, str] | None = None,
) -> None:
    source = os.environ if env is None else env
    security_enabled = (
        _SECURITY_ENABLED
        if env is None
        else source.get("SECURITY_MIDDLEWARE_ENABLED", "true").lower() == "true"
    )
    if security_enabled:
        try:
            from middleware.security import (
                SecurityMiddleware,
                RateLimitMiddleware,
                InputValidationMiddleware,
                SecurityHeadersMiddleware,
                AuditLogMiddleware,
            )
            # Starlette's last added middleware is outermost. Request order:
            # CORS -> audit -> security headers -> auth -> rate limit -> validation.
            application.add_middleware(InputValidationMiddleware)
            application.add_middleware(RateLimitMiddleware)
            application.add_middleware(SecurityMiddleware)
            application.add_middleware(SecurityHeadersMiddleware)
            application.add_middleware(AuditLogMiddleware)
            logger.info("✅ 安全中间件已启用（认证+限流+输入验证+审计+安全头）")
        except ImportError as e:
            if source.get("ENVIRONMENT", "development").strip().lower() == "production":
                raise RuntimeError("Security middleware is required in production") from e
            logger.warning(f"安全中间件导入失败，跳过: {e}")

    # CORS is outermost so even authentication and rate-limit errors include CORS headers.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(source),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ── 挂载所有路由 ──
_ROUTERS = (
    stats_router,
    messages_router,
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


async def _readiness_model_check() -> bool:
    from api.generate import get_vllm_client

    client = await get_vllm_client()
    if client is None:
        return False
    health = await client.health_check()
    return health.get("summary", {}).get("healthy", 0) > 0


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
    logger.error(f"请求异常: {exc}")
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
    from app.config import is_vllm_enabled

    runtime_container = container if container is not None else RuntimeContainer.default(startup_env=_STARTUP_ENV)
    application = FastAPI(
        title="MultiPersonal Chat System API (增强版)",
        description="MultiPersonal Chat System后端服务API - 高并发/高可靠/高安全增强版",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.runtime_container = runtime_container
    runtime_env = runtime_container.startup_env
    model_required = (
        runtime_env.get("MODEL_PROVIDER", "").strip().lower() == "vllm"
        or is_vllm_enabled(runtime_env)
    )
    application.state.readiness_probe = ReadinessProbe(
        database_check=lambda: get_runtime_container(application).db.execute_sql("SELECT 1"),
        model_check=_readiness_model_check if model_required else None,
        model_required=model_required,
    )
    _install_middleware(application, runtime_container.startup_env)
    for router in _ROUTERS:
        application.include_router(router)

    application.add_api_route("/", root, methods=["GET"])
    application.add_api_route("/health", health_check, methods=["GET"])
    application.add_api_route("/ready", readiness_check, methods=["GET"])
    application.add_exception_handler(Exception, global_exception_handler)
    return application


app = create_app()
