"""安全中间件模块

提供 API Key 认证、限流、输入验证、安全头、审计日志等中间件。
基于 Starlette 中间件基类实现，可直接挂载到 FastAPI 应用。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 配置常量
# ═══════════════════════════════════════════════════════════════

# API Key 认证
API_KEYS: list[str] = [
    k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()
]

# C18 fix: 服务间认证 Token - 允许内部服务（如 bot）调用后端 API
# 通过 SERVICE_TOKEN 环境变量配置，逗号分隔支持多个 token
SERVICE_TOKENS: set[str] = {
    t.strip() for t in os.getenv("SERVICE_TOKEN", "").split(",") if t.strip()
}

# 认证白名单路径（前缀匹配）
AUTH_WHITELIST: set[str] = {
    "/health",
    "/ready",
    "/docs",
    "/openapi.json",
    "/api/auth/login",
    "/api/auth/register",
    "/",
}

# 限流配置
RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "300"))
RATE_LIMIT_TPM: int = int(os.getenv("RATE_LIMIT_TPM", "500000"))
GENERATE_RPM: int = int(os.getenv("GENERATE_RPM", "60"))
GENERATE_TPM: int = int(os.getenv("GENERATE_TPM", "500000"))
AUTH_RPM: int = int(os.getenv("AUTH_RPM", "10"))
TRUST_PROXY_HEADERS: bool = os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in {"1", "true", "yes", "on"}

# 输入验证
MAX_BODY_SIZE: int = int(os.getenv("MAX_BODY_SIZE", str(1024 * 1024)))  # 默认 1MB
PROMPT_MAX_LENGTH: int = int(os.getenv("PROMPT_MAX_LENGTH", "10000"))

# 审计日志
AUDIT_LOG_DIR: str = os.getenv("AUDIT_LOG_DIR", str(Path(__file__).parent.parent / "logs"))

# Prompt 注入检测模式
PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+previous", re.IGNORECASE),
    re.compile(r"ignore\s+above", re.IGNORECASE),
    re.compile(r"ignore\s+all\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+previous", re.IGNORECASE),
    re.compile(r"forget\s+(?:your|previous|all)", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"你是一个", re.IGNORECASE),
    re.compile(r"你现在是", re.IGNORECASE),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if\s+you\s+are|a)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"sudo\s+mode", re.IGNORECASE),
    re.compile(r"override\s+(?:safety|security|restrictions)", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your|the)\s+(?:instructions|prompt|system)", re.IGNORECASE),
    re.compile(r"output\s+your\s+(?:instructions|prompt|system)", re.IGNORECASE),
]

# 敏感字段脱敏关键词
SENSITIVE_FIELD_NAMES: set[str] = {
    "password", "password_hash", "api_key", "secret", "token",
    "authorization", "cookie", "credential", "openid", "unionid",
    "phone", "mobile",
}


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _mask_sensitive(value: str, visible: int = 4) -> str:
    """脱敏敏感值，仅保留前 visible 个字符可见。"""
    if not value or len(value) <= visible:
        return "***"
    return value[:visible] + "***"


def _sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """递归脱敏字典中的敏感字段。"""
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(s in key_lower for s in SENSITIVE_FIELD_NAMES):
            sanitized[key] = _mask_sensitive(str(value)) if value else value
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def _get_client_ip(request: Request) -> str:
    """获取客户端真实 IP，支持代理头。"""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if TRUST_PROXY_HEADERS and forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()
    client = request.client
    return client.host if client else "unknown"


def _detect_prompt_injection(text: str) -> bool:
    """检测文本中是否包含 Prompt 注入模式。"""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# 滑动窗口限流器
# ═══════════════════════════════════════════════════════════════

class SlidingWindowLimiter:
    """基于内存的滑动窗口限流器。

    使用时间戳列表记录请求，按窗口大小滑动计算请求数/Token数。
    线程安全，适用于单实例部署。
    """

    def __init__(self) -> None:
        # key -> 请求时间戳列表
        self._request_windows: dict[str, deque[float]] = defaultdict(deque)
        # key -> token 使用时间戳和数量列表 [(timestamp, token_count)]
        self._token_windows: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _cleanup_window(self, timestamps: deque[float], window_sec: float) -> None:
        """清理窗口外的过期时间戳。"""
        cutoff = time.time() - window_sec
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        # Remove empty keys to prevent memory leak
        # (caller should handle key removal since it knows the key)

    def _cleanup_token_window(
        self, records: deque[tuple[float, int]], window_sec: float
    ) -> None:
        """清理 Token 窗口外的过期记录。"""
        cutoff = time.time() - window_sec
        while records and records[0][0] < cutoff:
            records.popleft()

    def check_rpm(self, key: str, limit: int, window_sec: float = 60.0) -> tuple[bool, int, float]:
        """检查 RPM 限制。

        Args:
            key: 限流键（IP + 用户组合）
            limit: 每分钟请求数限制
            window_sec: 窗口大小（秒）

        Returns:
            (是否允许, 当前窗口请求数, 重试等待秒数)
        """
        with self._lock:
            now = time.time()
            timestamps = self._request_windows[key]
            self._cleanup_window(timestamps, window_sec)
            # Remove empty keys to prevent memory leak
            if not timestamps:
                self._request_windows.pop(key, None)
                self._token_windows.pop(key, None)
            current_count = len(timestamps)

            if current_count >= limit:
                # 计算最早请求过期时间
                oldest = timestamps[0] if timestamps else now
                retry_after = oldest + window_sec - now
                return False, current_count, max(0, retry_after)

            timestamps.append(now)
            return True, current_count + 1, 0

    def check_tpm(self, key: str, token_count: int, limit: int, window_sec: float = 60.0) -> tuple[bool, int, float]:
        """检查 TPM 限制。

        Args:
            key: 限流键
            token_count: 本次请求消耗的 Token 数
            limit: 每分钟 Token 数限制
            window_sec: 窗口大小（秒）

        Returns:
            (是否允许, 当前窗口 Token 总数, 重试等待秒数)
        """
        with self._lock:
            now = time.time()
            records = self._token_windows[key]
            self._cleanup_token_window(records, window_sec)
            # Remove empty keys to prevent memory leak
            if not records:
                self._token_windows.pop(key, None)
                self._request_windows.pop(key, None)
            current_tokens = sum(count for _, count in records)

            if current_tokens + token_count > limit:
                oldest = records[0][0] if records else now
                retry_after = oldest + window_sec - now
                return False, current_tokens, max(0, retry_after)

            records.append((now, token_count))
            return True, current_tokens + token_count, 0

    def get_stats(self) -> dict[str, Any]:
        """获取当前限流统计信息。"""
        with self._lock:
            now = time.time()
            stats: dict[str, Any] = {
                "rpm_windows": {},
                "tpm_windows": {},
            }
            empty_rpm_keys = []
            for key, timestamps in self._request_windows.items():
                self._cleanup_window(timestamps, 60.0)
                if timestamps:
                    stats["rpm_windows"][key] = {
                        "count": len(timestamps),
                        "oldest": datetime.fromtimestamp(timestamps[0]).isoformat(),
                    }
                else:
                    empty_rpm_keys.append(key)
            empty_tpm_keys = []
            for key, records in self._token_windows.items():
                self._cleanup_token_window(records, 60.0)
                if records:
                    total = sum(c for _, c in records)
                    stats["tpm_windows"][key] = {
                        "total_tokens": total,
                        "record_count": len(records),
                    }
                else:
                    empty_tpm_keys.append(key)
            # Remove empty keys to prevent memory leak
            for key in empty_rpm_keys:
                self._request_windows.pop(key, None)
            for key in empty_tpm_keys:
                self._token_windows.pop(key, None)
            return stats


# 全局限流器实例
_rate_limiter = SlidingWindowLimiter()


def get_rate_limiter() -> SlidingWindowLimiter:
    """获取全局限流器实例。"""
    return _rate_limiter


# ═══════════════════════════════════════════════════════════════
# 审计日志管理器
# ═══════════════════════════════════════════════════════════════

class AuditLogger:
    """审计日志管理器，按日轮转写入日志文件。"""

    def __init__(self, log_dir: str | None = None) -> None:
        self._log_dir = Path(log_dir or AUDIT_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.INFO)
        # 避免重复添加 handler
        if not self._logger.handlers:
            handler = TimedRotatingFileHandler(
                filename=str(self._log_dir / "audit.log"),
                when="midnight",
                interval=1,
                backupCount=90,  # 保留 90 天
                encoding="utf-8",
            )
            handler.suffix = "%Y-%m-%d"
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(handler)

    def log_request(
        self,
        *,
        timestamp: str,
        client_ip: str,
        user: str,
        path: str,
        method: str,
        status_code: int,
        duration_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """记录 API 请求审计日志。"""
        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "client_ip": client_ip,
            "user": user,
            "path": path,
            "method": method,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }
        if extra:
            # 脱敏后合并
            entry["extra"] = _sanitize_dict(extra)
        self._logger.info(json.dumps(entry, ensure_ascii=False))


# 全局审计日志实例
_audit_logger = AuditLogger()


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志实例。"""
    return _audit_logger


# ═══════════════════════════════════════════════════════════════
# 1. API Key 认证中间件
# ═══════════════════════════════════════════════════════════════

class SecurityMiddleware(BaseHTTPMiddleware):
    """API Key / JWT 认证中间件。

    认证逻辑：
    1. 白名单路径直接放行
    2. 从 X-API-Key 头或 Authorization: Bearer <key> 读取凭证
    3. 先尝试 API Key 匹配，再尝试 JWT 验证
    4. 未认证请求返回 401
    """

    def __init__(self, app: Any, api_keys: list[str] | None = None) -> None:
        super().__init__(app)
        self._api_keys: set[str] = set(api_keys) if api_keys else set(API_KEYS)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 白名单路径放行（前缀匹配）
        path = request.url.path
        if path in AUTH_WHITELIST or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        # OPTIONS 请求放行（CORS 预检）
        if request.method == "OPTIONS":
            return await call_next(request)

        # 公开只读端点放行（无需认证）
        PUBLIC_GET_PATHS = {"/health", "/ready"}
        # AstrBot has independent token/signature authentication in its route.
        PUBLIC_POST_PATHS = {"/api/integrations/astrbot/messages"}
        if request.method in ("GET", "HEAD") and (path in PUBLIC_GET_PATHS or path.startswith("/docs") or path.startswith("/redoc")):
            return await call_next(request)
        if request.method == "POST" and path in PUBLIC_POST_PATHS:
            return await call_next(request)

        # C18 fix: 服务间认证 - 允许携带有效 X-Service-Token 的内部服务调用（如 bot → RAG）
        # 这避免了 bot 进程必须持有用户 JWT 的耦合，同时保证非白名单接口不会被匿名访问
        service_token = request.headers.get("X-Service-Token", "")
        if service_token and service_token in SERVICE_TOKENS:
            request.state.user = "service-bot"
            request.state.auth_type = "service_token"
            return await call_next(request)

        # 读取认证凭证
        api_key = request.headers.get("X-API-Key", "")
        auth_header = request.headers.get("Authorization", "")

        # 从 Cookie 中提取 token（httpOnly Cookie 认证）
        # 使用 Starlette 内置 API 自动处理 URL 解码
        cookie_token = request.cookies.get("access_token", "")

        bearer_token = ""
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:]
        elif cookie_token:
            # Cookie 中的 token 作为 Bearer token
            bearer_token = cookie_token

        # 优先尝试 JWT 验证（前端主要使用 JWT）
        if bearer_token:
            try:
                import jwt as pyjwt
                from app.config import JWT_SECRET, JWT_ALGORITHM

                payload = pyjwt.decode(bearer_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                # C-S2 fix: 原先中间件解码 JWT 后直接放行，不检查 Token 黑名单。
                # 导致未使用 Depends(get_current_user) 的路由（如 /api/stats、
                # /api/model/status）在 Token 注销后仍可访问 24 小时。
                # 现在在中间件层统一检查黑名单，所有路由都受益。
                from api.auth import is_token_revoked
                if is_token_revoked(payload.get("jti", "")):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Token 已注销，请重新登录"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                request.state.user = payload.get("sub", "unknown")
                request.state.auth_type = "jwt"
                request.state.jwt_payload = payload
            except ImportError:
                logger.warning("jwt 库未安装，无法验证 JWT Token")
            except Exception as e:
                logger.debug(f"JWT 验证失败: {type(e).__name__}: {e}")

            # JWT 验证成功，放行请求（call_next 的异常不应被认证逻辑捕获）
            if hasattr(request.state, "jwt_payload") and request.state.jwt_payload:
                return await call_next(request)

        # 尝试 API Key 匹配
        if api_key and self._api_keys and api_key in self._api_keys:
            request.state.user = "api_key_user"
            request.state.auth_type = "api_key"
            return await call_next(request)

        # 没有任何有效凭证
        return JSONResponse(
            status_code=401,
            content={"detail": "缺少认证凭证，请提供 X-API-Key 或 Authorization: Bearer <key>"},
            headers={"WWW-Authenticate": "Bearer"},
        )


# ═══════════════════════════════════════════════════════════════
# 2. 限流中间件
# ═══════════════════════════════════════════════════════════════

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token 级别限流中间件。

    基于 IP + 用户组合进行限流，使用内存滑动窗口算法。
    推理 API 单独限流，其他 API 统一限流。
    超限返回 429 + Retry-After 头。
    """

    def __init__(
        self,
        app: Any,
        limiter: SlidingWindowLimiter | None = None,
        default_rpm: int | None = None,
        default_tpm: int | None = None,
        generate_rpm: int | None = None,
        generate_tpm: int | None = None,
        auth_rpm: int | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter or _rate_limiter
        self._default_rpm = default_rpm or RATE_LIMIT_RPM
        self._default_tpm = default_tpm or RATE_LIMIT_TPM
        self._generate_rpm = generate_rpm or GENERATE_RPM
        self._generate_tpm = generate_tpm or GENERATE_TPM
        self._auth_rpm = auth_rpm or AUTH_RPM

    def _get_rate_limit_key(self, request: Request) -> str:
        """生成限流键：IP + 用户。"""
        client_ip = _get_client_ip(request)
        user = getattr(request.state, "user", None) or "anonymous"
        return f"{client_ip}:{user}"

    def _is_generate_endpoint(self, path: str) -> bool:
        """判断是否为推理 API 路径。"""
        return path == "/api/generate"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        key = self._get_rate_limit_key(request)

        # Only health probes bypass rate limiting. Login/register remain public
        # but have a stricter anonymous limit.
        if path in {"/health", "/ready", "/"}:
            return await call_next(request)

        # 根据路径选择限流配置
        if path in {"/api/auth/login", "/api/auth/register"}:
            rpm_limit = self._auth_rpm
            tpm_limit = self._default_tpm
        elif self._is_generate_endpoint(path):
            rpm_limit = self._generate_rpm
            tpm_limit = self._generate_tpm
        else:
            rpm_limit = self._default_rpm
            tpm_limit = self._default_tpm

        # 检查 RPM
        allowed, current_count, retry_after = self._limiter.check_rpm(key, rpm_limit)
        if not allowed:
            logger.warning(f"RPM 限流触发: key={key}, count={current_count}, limit={rpm_limit}")
            return JSONResponse(
                status_code=429,
                content={"detail": f"请求频率超限，每分钟最多 {rpm_limit} 次请求"},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        # 对于推理 API，额外检查 TPM（预估 Token 数）
        if self._is_generate_endpoint(path) and request.method == "POST":
            try:
                # 读取请求体估算 Token 数（简单估算：字符数 / 4）
                body = await request.body()
                estimated_tokens = len(body) // 4
                if estimated_tokens > 0:
                    tpm_allowed, current_tpm, tpm_retry = self._limiter.check_tpm(
                        key, estimated_tokens, tpm_limit
                    )
                    if not tpm_allowed:
                        logger.warning(
                            f"TPM 限流触发: key={key}, tokens={current_tpm}, limit={tpm_limit}"
                        )
                        return JSONResponse(
                            status_code=429,
                            content={"detail": f"Token 用量超限，每分钟最多 {tpm_limit} Tokens"},
                            headers={"Retry-After": str(int(tpm_retry) + 1)},
                        )
            except Exception:
                pass  # 读取 body 失败时不阻断请求

        return await call_next(request)


# ═══════════════════════════════════════════════════════════════
# 3. 输入验证中间件
# ═══════════════════════════════════════════════════════════════

class InputValidationMiddleware(BaseHTTPMiddleware):
    """输入验证中间件。

    检查请求体大小、Prompt 长度，并进行 Prompt 注入检测。
    检测到可疑输入时直接阻断请求，返回 422。
    """

    def __init__(
        self,
        app: Any,
        max_body_size: int | None = None,
        prompt_max_length: int | None = None,
    ) -> None:
        super().__init__(app)
        self._max_body_size = max_body_size or MAX_BODY_SIZE
        self._prompt_max_length = prompt_max_length or PROMPT_MAX_LENGTH

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 仅对有请求体的方法进行验证
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        # 安全：BaseHTTPMiddleware 消费 body 后下游端点（Pydantic）无法再读取。
        # 仅对 /api/generate 等需要 Prompt 注入检测的端点预先读取 body 并缓存回 request。
        # 其他 POST 端点（知识库/训练/LoRA等）有独立 Pydantic 校验，跳过此处 body 读取。
        _body_guard_paths = ("/api/generate",)
        if not request.url.path.startswith(_body_guard_paths):
            return await call_next(request)

        # 读取请求体
        body = await request.body()
        # 缓存 body 到 request，使下游 Request.body() 命中缓存
        request._body = body

        # 请求体大小检查
        if len(body) > self._max_body_size:
            return JSONResponse(
                status_code=413,
                content={"detail": f"请求体过大，最大允许 {self._max_body_size // (1024 * 1024)}MB"},
            )

        # 解析 JSON 并检查 Prompt 相关字段
        suspicious = False
        prompt_too_long = False
        try:
            body_json = json.loads(body)
            suspicious = self._check_prompt_fields(body_json)
            prompt_too_long = self._check_prompt_length(body_json)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # 非 JSON 请求体跳过 Prompt 检查

        # Prompt 长度超限：阻断
        if prompt_too_long:
            logger.warning(f"Prompt 长度超限: path={request.url.path}, ip={_get_client_ip(request)}")
            return JSONResponse(
                status_code=422,
                content={"detail": f"Prompt 长度超过限制（最大 {self._prompt_max_length} 字符）"},
            )

        # Prompt 注入检测：直接阻断
        if suspicious:
            logger.warning(f"检测到可疑 Prompt 注入攻击: path={request.url.path}, ip={_get_client_ip(request)}")
            return JSONResponse(
                status_code=422,
                content={"detail": "输入内容包含可疑的 Prompt 注入模式，请求已被拦截"},
            )

        return await call_next(request)

    def _check_prompt_fields(self, data: dict[str, Any] | list[Any]) -> bool:
        """递归检查数据中的 Prompt 注入模式。"""
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                if isinstance(value, str) and any(
                    kw in key_lower for kw in ("prompt", "message", "content", "query", "text")
                ):
                    # Prompt 注入检测
                    if _detect_prompt_injection(value):
                        return True
                # 递归检查嵌套结构
                if isinstance(value, (dict, list)):
                    if self._check_prompt_fields(value):
                        return True
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    if self._check_prompt_fields(item):
                        return True
        return False

    def _check_prompt_length(self, data: dict[str, Any] | list[Any]) -> bool:
        """递归检查 Prompt 字段长度是否超限。"""
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                if isinstance(value, str) and any(
                    kw in key_lower for kw in ("prompt", "message", "content", "query", "text")
                ):
                    if len(value) > self._prompt_max_length:
                        return True
                if isinstance(value, (dict, list)):
                    if self._check_prompt_length(value):
                        return True
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    if self._check_prompt_length(item):
                        return True
        return False


# ═══════════════════════════════════════════════════════════════
# 4. 安全头中间件
# ═══════════════════════════════════════════════════════════════

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全响应头中间件。

    自动添加安全相关的 HTTP 响应头：
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Content-Security-Policy: default-src 'self'
    - Strict-Transport-Security: max-age=31536000（仅 HTTPS）
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # 防止 MIME 类型嗅探
        response.headers["X-Content-Type-Options"] = "nosniff"

        # 防止点击劫持
        response.headers["X-Frame-Options"] = "DENY"

        # XSS 防护（旧浏览器兼容）
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # 内容安全策略
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # HSTS（仅 HTTPS 时添加）
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


# ═══════════════════════════════════════════════════════════════
# 5. 审计日志中间件
# ═══════════════════════════════════════════════════════════════

class AuditLogMiddleware(BaseHTTPMiddleware):
    """审计日志中间件。

    记录所有 API 请求的审计信息，包括时间、IP、用户、路径、方法、状态码、耗时。
    推理请求额外记录 prompt 长度、生成 token 数、使用的模型/LoRA。
    """

    def __init__(self, app: Any, audit_logger: AuditLogger | None = None) -> None:
        super().__init__(app)
        self._audit_logger = audit_logger or _audit_logger

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        path = request.url.path
        method = request.method

        # 执行请求
        response = await call_next(request)

        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000

        # 获取用户信息
        user = getattr(request.state, "user", None) or "anonymous"
        client_ip = _get_client_ip(request)

        # 构建额外信息
        extra: dict[str, Any] = {}

        # 推理请求额外记录
        if path == "/api/generate" and method == "POST":
            try:
                # 注意：body 可能已被之前的中间件消费，这里尝试从 state 获取
                body_size = getattr(request.state, "body_size", None)
                if body_size is not None:
                    extra["prompt_length"] = body_size

                # 从响应中提取生成信息（如果有的话）
                generation_info = getattr(request.state, "generation_info", None)
                if generation_info:
                    extra.update(generation_info)
            except Exception:
                pass

        # 可疑输入标记
        if getattr(request.state, "suspicious_input", False):
            extra["suspicious_input"] = True

        # 写入审计日志
        self._audit_logger.log_request(
            timestamp=datetime.now().isoformat(),
            client_ip=client_ip,
            user=user,
            path=path,
            method=method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            extra=extra if extra else None,
        )

        return response


# ═══════════════════════════════════════════════════════════════
# 6. 请求 ID 中间件（链路追踪）
# ═══════════════════════════════════════════════════════════════

import uuid
import contextvars

# 上下文变量：同一次请求内所有日志自动携带 request_id
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def get_request_id() -> str:
    """获取当前请求的 request_id（供日志/业务代码使用）。"""
    return _request_id_ctx.get()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """请求 ID 注入中间件。

    为每个请求生成唯一 UUID，注入到：
    - X-Request-Id 响应头（客户端可记录）
    - contextvars（日志自动携带）
    - request.state.request_id（业务代码可读取）

    优先级：客户端传入 > 服务端生成
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 优先使用客户端传入的 X-Request-Id，否则生成
        req_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        token = _request_id_ctx.set(req_id)
        request.state.request_id = req_id
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = req_id
            return response
        finally:
            _request_id_ctx.reset(token)
