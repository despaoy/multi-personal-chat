"""QQ智能助手 - 后端主应用

整合所有模块，创建 FastAPI 应用实例。
路由按领域拆分到 api/ 下各模块，通过 APIRouter 挂载。
"""
import sys
from pathlib import Path
import os
import logging

# 确保 backend 根目录在 Python 路径中，支持跨包导入
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 加载 .env 文件到环境变量（确保 worker 进程也能读取配置）
_env_path = _BACKEND_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _value = _line.partition("=")
                _key = _key.strip()
                _value = _value.strip()
                if _key and _key not in os.environ:
                    os.environ[_key] = _value

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
    async_task_queue, llm_optimizer,
    RESOURCE_POOL_AVAILABLE, BACKUP_MANAGER_AVAILABLE,
    FAILOVER_AVAILABLE, ACCESS_CONTROL_AVAILABLE,
    load_balancer_mgr, circuit_breaker_registry,
)
from db.adapter import db, is_pg_mode
from infra.deployment import validate_or_raise_for_startup

logger = logging.getLogger("main")


def _initialize_database(database) -> None:
    """Initialize and probe either database adapter through its public contract."""
    if hasattr(database, "init"):
        database.init()
    database.execute_sql("SELECT 1")

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


# ═══════════════════════════════════════════
# 应用生命周期
# ═══════════════════════════════════════════
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

    logger.info("🚀 QQ智能助手后端服务启动中（增强版）...")

    validate_or_raise_for_startup(_STARTUP_ENV)

    # 初始化数据库
    try:
        _initialize_database(db)
        logger.info("✅ 数据库初始化完成 (%s)", "PostgreSQL" if is_pg_mode() else "SQLite")
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

    if RESOURCE_POOL_AVAILABLE and not is_pg_mode():
        try:
            from infra.resource_pool import ConnectionPool, HttpClientPool
            db_path = str(getattr(db, "db_path", _BACKEND_ROOT / "qq_assistant.db"))
            _cfg.connection_pool = ConnectionPool(db_path, max_size=20)
            _cfg.http_client_pool = HttpClientPool(max_connections=100)
            logger.info("✅ 资源池初始化完成")
        except Exception as e:
            logger.warning(f"资源池初始化失败: {e}")

    if BACKUP_MANAGER_AVAILABLE and not is_pg_mode():
        try:
            from infra.backup_manager import BackupManager
            db_path = str(getattr(db, "db_path", _BACKEND_ROOT / "qq_assistant.db"))
            _cfg.backup_mgr = BackupManager(db_path)
            import asyncio
            asyncio.create_task(_cfg.backup_mgr.start_scheduled_backup())
            logger.info("✅ 备份管理器初始化完成")
        except Exception as e:
            logger.warning(f"备份管理器初始化失败: {e}")

    if FAILOVER_AVAILABLE:
        try:
            from infra.failover import FailoverManager, ProviderConfig, FailoverStrategy
            _cfg.failover_mgr = FailoverManager(strategy=FailoverStrategy.AUTO_FAILOVER)

            # 注册 vLLM 推理 provider
            vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001")
            _cfg.failover_mgr.add_provider(ProviderConfig(
                name="vllm_primary",
                priority=1,
                health_check_url=f"{vllm_url}/health",
            ))

            # 启动健康检查循环
            await _cfg.failover_mgr.start()
            logger.info("✅ 故障转移管理器初始化完成（vLLM provider 已注册）")
        except Exception as e:
            logger.warning(f"故障转移管理器初始化失败: {e}")

    if ACCESS_CONTROL_AVAILABLE and not is_pg_mode():
        try:
            from infra.access_control import AccessControlManager
            db_path = str(getattr(db, "db_path", _BACKEND_ROOT / "qq_assistant.db"))
            _cfg.access_control_mgr = AccessControlManager(db_path)
            logger.info("✅ 访问控制管理器初始化完成")
        except Exception as e:
            logger.warning(f"访问控制管理器初始化失败: {e}")

    # 延迟重建向量索引（首次搜索时自动触发，避免启动时阻塞）
    # 见 api/knowledge.py search_knowledge 中的 _ensure_vector_index()

    logger.info("✅ 增强版服务启动完成！")
    yield

    logger.info("👋 服务关闭中，清理资源...")
    if _cfg.connection_pool:
        await _cfg.connection_pool.close()
    if _cfg.http_client_pool:
        await _cfg.http_client_pool.close()
    if _cfg.backup_mgr:
        stop_result = _cfg.backup_mgr.stop_scheduled_backup()
        if hasattr(stop_result, "__await__"):
            await stop_result
    if _cfg.failover_mgr:
        await _cfg.failover_mgr.stop()
    if _cfg.async_task_queue:
        await _cfg.async_task_queue.shutdown()
    if _cfg.llm_optimizer:
        await _cfg.llm_optimizer.close()
    # C6 fix: 关闭所有模型 Provider 持有的 httpx 客户端与 GPU 句柄
    try:
        from inference.model_manager import get_model_manager
        get_model_manager().shutdown()
    except Exception as e:
        logger.warning(f"关闭模型管理器失败: {e}")
    # C10 fix: 关闭 generate.py 中独立的 vLLM 客户端连接池
    try:
        from api.generate import close_vllm_client
        await close_vllm_client()
    except Exception as e:
        logger.warning(f"关闭 vLLM 客户端失败: {e}")
    # C15 fix: 关闭 bot 模块中挂载在函数属性上的 VLLM 客户端
    try:
        from bot.bot import close_bot_vllm_client
        await close_bot_vllm_client()
    except Exception as e:
        logger.warning(f"关闭 bot VLLM 客户端失败: {e}")
    # 关闭 bot 模块中共享的 HTTP 客户端（RAG 搜索 + Ollama 推理）
    try:
        from bot.bot import _close_bot_http_clients
        await _close_bot_http_clients()
    except Exception as e:
        logger.warning(f"关闭 bot HTTP 客户端失败: {e}")
    # 关闭共享的 async Redis 客户端
    try:
        from cache.redis_client import close_async_redis
        await close_async_redis()
    except Exception as e:
        logger.warning(f"关闭 async Redis 客户端失败: {e}")
    logger.info("✅ 资源清理完成")


# ═══════════════════════════════════════════
# 创建应用
# ═══════════════════════════════════════════
app = FastAPI(
    title="QQ智能助手 API (增强版)",
    description="QQ智能助手后端服务API - 高并发/高可靠/高安全增强版",
    version="2.0.0",
    lifespan=lifespan,
)

def _allowed_origins() -> list[str]:
    """解析 CORS 允许源，优先 ALLOWED_ORIGINS，旧键 CORS_ORIGINS 发 deprecation warning。"""
    configured = os.getenv("ALLOWED_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    legacy = os.getenv("CORS_ORIGINS")
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

if _SECURITY_ENABLED:
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
        app.add_middleware(InputValidationMiddleware)
        app.add_middleware(RateLimitMiddleware)
        app.add_middleware(SecurityMiddleware)
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(AuditLogMiddleware)
        logger.info("✅ 安全中间件已启用（认证+限流+输入验证+审计+安全头）")
    except ImportError as e:
        if os.getenv("ENVIRONMENT", "development").strip().lower() == "production":
            raise RuntimeError("Security middleware is required in production") from e
        logger.warning(f"安全中间件导入失败，跳过: {e}")

# CORS is outermost so even authentication and rate-limit errors include CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 挂载所有路由 ──
app.include_router(stats_router)
app.include_router(messages_router)
app.include_router(generate_router)
app.include_router(loras_router)
app.include_router(training_router)
app.include_router(knowledge_router)
app.include_router(models_router)
app.include_router(config_router)
app.include_router(auth_router)
app.include_router(user_data_router)
app.include_router(enhanced_router)
app.include_router(claw_router)
app.include_router(integrations_router)
app.include_router(evaluation_router)
app.include_router(experiments_router)
app.include_router(retrieval_eval_router)
app.include_router(preferences_router)
app.include_router(lora_router_router)


# ═══════════════════════════════════════════
# 根路由 & 健康检查
# ═══════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "name": "QQ智能助手 API (增强版)",
        "version": "2.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/ready")
async def readiness_check():
    """Readiness checks required local state and the configured inference provider."""
    model_required = (
        os.getenv("MODEL_PROVIDER", "").strip().lower() == "vllm"
        or os.getenv("VLLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    )
    deps = {"database": False, "faiss": False, "model": not model_required}
    details: dict[str, str] = {}

    try:
        from db.adapter import is_pg_mode
        if is_pg_mode():
            db.get_config()
        else:
            conn = db.get_connection()
            conn.execute("SELECT 1")
        deps["database"] = True
    except Exception as exc:
        details["database"] = type(exc).__name__

    try:
        from knowledge.vector_db import get_vector_db
        deps["faiss"] = get_vector_db() is not None
        details["faiss"] = "ok" if deps["faiss"] else "not_initialized"
    except Exception as exc:
        details["faiss"] = type(exc).__name__

    if model_required:
        try:
            from api.generate import get_vllm_client

            client = await asyncio.wait_for(get_vllm_client(), timeout=3.0)
            if client is not None:
                health = await asyncio.wait_for(client.health_check(), timeout=6.0)
                deps["model"] = health.get("summary", {}).get("healthy", 0) > 0
            details["model"] = "ok" if deps["model"] else "unavailable"
        except Exception as exc:
            details["model"] = type(exc).__name__

    ready = all(deps.values())
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "deps": deps, "details": details},
        )
    return {"status": "ready", "deps": deps}


# ═══════════════════════════════════════════
# 全局异常处理
# ═══════════════════════════════════════════
@app.exception_handler(Exception)
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
