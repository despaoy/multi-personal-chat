"""
应用配置模块
从 main.py 提取：JWT、LLM 并发控制、搜索工具、增强模块导入、全局实例
"""

import os
import asyncio
import logging
import secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

# ============================================
# JWT 配置
# ============================================

import jwt

__logger = logging.getLogger(__name__)


def _validate_jwt_secret(secret: str, environment: str) -> str:
    if secret and secret != "multipersonal-jwt-secret-change-in-production" and len(secret) >= 32:
        return secret
    if environment.strip().lower() == "production":
        raise RuntimeError("JWT_SECRET must be explicitly set to at least 32 characters in production")
    return ""


def _ensure_jwt_secret() -> str:
    """确保 JWT 密钥安全：优先从环境变量读取，否则自动生成并持久化到 .env"""
    secret = _validate_jwt_secret(
        os.getenv("JWT_SECRET", ""),
        os.getenv("ENVIRONMENT", "development"),
    )
    if secret:
        return secret
    # 自动生成安全密钥
    new_secret = secrets.token_urlsafe(48)
    env_path = Path(__file__).parent.parent / ".env"
    try:
        lines = []
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        # 更新或添加 JWT_SECRET
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("JWT_SECRET="):
                lines[i] = f"JWT_SECRET={new_secret}\n"
                found = True
                break
        if not found:
            lines.append(f"JWT_SECRET={new_secret}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.environ["JWT_SECRET"] = new_secret
        __logger.warning("⚠️ JWT_SECRET 已自动生成安全密钥并保存到 .env，请妥善保管")
    except Exception as e:
        __logger.error(f"自动生成 JWT_SECRET 失败: {e}")
        # 降级：仅在内存中使用生成的密钥（重启后失效）
        os.environ["JWT_SECRET"] = new_secret
    return new_secret


JWT_SECRET = _ensure_jwt_secret()
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
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
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
    if primary:
        return max(1, int(primary))
    legacy = os.getenv("MODEL_MAX_CONCURRENCY")
    if legacy:
        __logger.warning(
            "MODEL_MAX_CONCURRENCY 已弃用，请改用 LLM_MAX_CONCURRENCY。旧键将在下个大版本移除。"
        )
        return max(1, int(legacy))
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
# 角色信息网络搜索工具
# ============================================

# _logger 已在文件顶部定义为 __logger

# 模块级共享 httpx.Client，避免每次搜索都新建 TCP 连接+TLS 握手。
# 两个搜索源(DuckDuckGo/Wikipedia)均为短超时(5s)，可安全共享同一客户端。
import httpx as _httpx_module
_search_http_client = _httpx_module.Client(timeout=5.0, follow_redirects=True)


def _search_character_info(character_desc: str, max_results: int = 3) -> str:
    """在网络上搜索角色信息（短超时+降级，不阻塞生成）"""

    import concurrent.futures
    from urllib.parse import quote

    query = f"{character_desc} 角色 人物介绍"
    query_encoded = quote(query)

    def try_duckduckgo():
        """DuckDuckGo HTML 搜索"""
        try:
            url = f"https://html.duckduckgo.com/html/?q={query_encoded}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = _search_http_client.get(url, headers=headers)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(resp.text, "html.parser")
                items = soup.select(".result")
                lines = []
                for r in items[:max_results]:
                    t = r.select_one(".result__title")
                    s = r.select_one(".result__snippet")
                    if t and s:
                        lines.append(
                            f"- {t.get_text(strip=True)}\n  {s.get_text(strip=True)[:250]}"
                        )
                return "\n".join(lines) if lines else ""
        except Exception as e:
            # H3 fix (扩展): 此前静默吞噬，DuckDuckGo 搜索失败时无法排障
            __logger.warning("DuckDuckGo 搜索失败 (query=%s): %s", character_desc, e)
        return ""

    def try_wikipedia():
        """Wikipedia API"""
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": character_desc,
                "format": "json",
                "srlimit": 3,
            }
            resp = _search_http_client.get("https://zh.wikipedia.org/w/api.php", params=params)
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("search", [])
                import re

                return "\n".join(
                    f"- {p['title']}\n  {re.sub(r'<[^>]+>', '', p.get('snippet', ''))[:250]}"
                    for p in pages[:max_results]
                )
        except Exception as e:
            # H3 fix: 此前静默吞噬，Wikipedia 搜索失败时无法排障
            __logger.warning("Wikipedia 搜索失败 (query=%s): %s", character_desc, e)
        return ""

    # 并行搜索，总超时 6 秒
    result = ""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(try_duckduckgo): "ddg",
            executor.submit(try_wikipedia): "wiki",
        }
        for future in concurrent.futures.as_completed(futures, timeout=6):
            try:
                r = future.result(timeout=5)
                if r and len(r) > len(result):
                    result = r
            except Exception as e:
                # H3 fix (扩展): 此前静默吞噬 future 异常（含超时、子线程内未捕获异常）
                __logger.warning("搜索 future 失败 (query=%s): %s", character_desc, e)

    if result:
        __logger.info(f"角色信息搜索成功: {len(result)} 字符")
    else:
        __logger.info("角色信息搜索无结果（网络受限），将基于用户描述生成")

    return result


# ============================================
# 增强模块导入
# ============================================

LOAD_BALANCER_AVAILABLE = True

try:
    from infra.resource_pool import ConnectionPool, HttpClientPool

    RESOURCE_POOL_AVAILABLE = True
except ImportError:
    RESOURCE_POOL_AVAILABLE = False

try:
    from infra.circuit_breaker import global_registry

    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from infra.backup_manager import BackupManager

    BACKUP_MANAGER_AVAILABLE = True
except ImportError:
    BACKUP_MANAGER_AVAILABLE = False

try:
    from infra.failover import FailoverManager

    FAILOVER_AVAILABLE = True
except ImportError:
    FAILOVER_AVAILABLE = False

try:
    from infra.input_validator import (
        InputValidator,
        MESSAGE_SCHEMA,
        KNOWLEDGE_DOCUMENT_SCHEMA,
        KNOWLEDGE_SCHEMA,
        TRAINING_SCHEMA,
        CONFIG_SCHEMA,
    )

    INPUT_VALIDATOR_AVAILABLE = True
except ImportError:
    INPUT_VALIDATOR_AVAILABLE = False

try:
    from infra.encryption import EncryptionManager

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

try:
    from infra.access_control import AccessControlManager, Permission

    ACCESS_CONTROL_AVAILABLE = True
except ImportError:
    ACCESS_CONTROL_AVAILABLE = False

try:
    from cache.response_cache import ResponseCache

    LLM_OPTIMIZER_AVAILABLE = True
except ImportError:
    LLM_OPTIMIZER_AVAILABLE = False

# ============================================
# 服务启动时间
# ============================================

service_start_time = datetime.now()


# ============================================
# 增强模块全局实例
# ============================================

connection_pool = None  # 在lifespan中初始化
http_client_pool = None  # 在lifespan中初始化
circuit_breaker_registry = global_registry if CIRCUIT_BREAKER_AVAILABLE else None
backup_mgr = None  # 在lifespan中初始化
failover_mgr = None  # 在lifespan中初始化
encryption_mgr = EncryptionManager() if ENCRYPTION_AVAILABLE else None
access_control_mgr = None  # 在lifespan中初始化
response_cache = ResponseCache() if LLM_OPTIMIZER_AVAILABLE else None


# ============================================
# 向量数据库
# ============================================

try:
    from knowledge.vector_db import get_vector_db

    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False


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
