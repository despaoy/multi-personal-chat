"""
应用配置模块
从 main.py 提取：JWT、LLM 并发控制、增强模块导入、全局实例
"""

import os
import asyncio
import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from infra.deployment import validate_jwt_secret
from infra.environment import read_positive_int

from cache.response_cache import ResponseCache

# ============================================
# JWT 配置
# ============================================

import jwt

__logger = logging.getLogger(__name__)


JWT_SECRET = validate_jwt_secret(os.getenv("JWT_SECRET", ""))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24


def create_access_token(username: str, user_id: int, role: str = "user") -> str:
    """创建 JWT access token

    C-S1 fix: 将 role 写入 payload，让 get_current_user 无需 DB 查询即可返回 role。
    注意：role 变更后旧 token 仍持旧值；敏感操作应通过 get_current_admin
    做一次 DB 复核以保证准确性。
    """
    payload = {
        "sub": username,
        "user_id": user_id,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16),  # 唯一ID，支持吊销
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """验证 JWT token，返回 payload 或抛出异常"""
    try:
        payload = jwt.decode(
            token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"require": ["sub"]}
        )
        if not isinstance(payload["sub"], str) or not payload["sub"].strip():
            raise jwt.InvalidTokenError("Token subject must be non-empty text")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token 无效")


# ============================================
# LLM 并发控制器
# ============================================

def _resolve_concurrency_limit() -> int:
    """解析 LLM 并发上限，优先 LLM_MAX_CONCURRENCY，旧键 MODEL_MAX_CONCURRENCY 发 deprecation warning。"""
    primary = os.getenv("LLM_MAX_CONCURRENCY")
    if primary is not None:
        return read_positive_int(os.environ, "LLM_MAX_CONCURRENCY", 2)
    legacy = os.getenv("MODEL_MAX_CONCURRENCY")
    if legacy is not None:
        __logger.warning(
            "MODEL_MAX_CONCURRENCY 已弃用，请改用 LLM_MAX_CONCURRENCY。旧键将在下个大版本移除。"
        )
        return read_positive_int(os.environ, "MODEL_MAX_CONCURRENCY", 2)
    return 2


LLM_CONCURRENCY_LIMIT = _resolve_concurrency_limit()
# 本地模型回退与统一推理队列职责不同：前者限制 GPU provider 并发，
# 后者负责所有入口的排队、优先级和会话串行化。
_llm_semaphore: asyncio.Semaphore | None = None
_llm_semaphore_loop: asyncio.AbstractEventLoop | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore, _llm_semaphore_loop
    loop = asyncio.get_running_loop()
    if _llm_semaphore is None or _llm_semaphore_loop is not loop:
        _llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)
        _llm_semaphore_loop = loop
    return _llm_semaphore


# ============================================
# vLLM 启用判定（单一入口，消除 5 处重复逻辑）
# ============================================

def is_vllm_enabled(env=os.environ) -> bool:
    """统一判定 vLLM 是否启用。

    启用条件（任一满足即可）：
    1. VLLM_ENABLED=true/1/yes/on
    2. VLLM_BASE_URLS 非空（逗号分隔多实例）
    3. VLLM_BASE_URL 非空（单实例）

    所有调用方应使用本函数，避免散落的 os.getenv 判定导致配 A 忘配 B。
    """
    if str(env.get("VLLM_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if str(env.get("VLLM_BASE_URLS", "")).strip():
        return True
    if str(env.get("VLLM_BASE_URL", "")).strip():
        return True
    return False


def get_vllm_served_model_name(env=os.environ) -> str:
    """统一解析 vLLM 服务模型名。

    优先级：VLLM_SERVED_MODEL_NAME > VLLM_MODEL > 默认值。
    """
    return str(
        env.get("VLLM_SERVED_MODEL_NAME")
        or env.get("VLLM_MODEL")
        or "qwen3-8b-instruct-awq"
    )


# ============================================
# 增强模块导入
# ============================================


from infra.circuit_breaker import global_registry


from infra.encryption import EncryptionManager


# ============================================
# 服务启动时间
# ============================================

service_start_time = datetime.now()


# ============================================
# 增强模块全局实例
# ============================================

connection_pool = None  # 在lifespan中初始化
http_client_pool = None  # 在lifespan中初始化
circuit_breaker_registry = global_registry
backup_mgr = None  # 在lifespan中初始化
encryption_mgr = EncryptionManager()
access_control_mgr = None  # 在lifespan中初始化
response_cache = ResponseCache()


# ============================================
# 对话生成状态追踪（异步安全）
# ============================================

_generation_state_lock: asyncio.Lock | None = None
_generation_state_lock_loop: asyncio.AbstractEventLoop | None = None


def get_generation_state_lock() -> asyncio.Lock:
    global _generation_state_lock, _generation_state_lock_loop
    loop = asyncio.get_running_loop()
    if _generation_state_lock is None or _generation_state_lock_loop is not loop:
        _generation_state_lock = asyncio.Lock()
        _generation_state_lock_loop = loop
    return _generation_state_lock

generation_state = {
    "is_generating": False,
    "cancel_requested": False,
    "progress": 0,
    "total": 0,
    "batch_num": 0,
    "total_batches": 0,
    "generated_count": 0,
    "all_generated_dialogues": [],
    "started_at": None,
}
