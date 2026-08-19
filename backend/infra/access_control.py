"""
访问权限精细化控制模块 - 基于角色的访问控制(RBAC)

提供完整的访问控制功能，包括：
- 基于角色的访问控制（admin/operator/viewer/api_user）
- API Key认证和管理
- FastAPI依赖注入中间件
- 权限检查装饰器
- 速率限制（每API Key独立）
- 审计日志

使用方式：
    from access_control import (
        AccessControlManager, Permission, Role,
        require_permission, get_current_user
    )

    # FastAPI依赖注入
    @router.post("/config")
    async def update_config(user: dict = Depends(get_current_user)):
        ...

    # 权限检查装饰器
    @router.delete("/loras/{id}")
    @require_permission(Permission.DELETE_LORA)
    async def delete_lora(id: str):
        ...
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import logging
import os
import secrets
import time
from enum import Enum, Flag, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# API Key前缀
_API_KEY_PREFIX = "qqa_"

# API Key随机部分长度（十六进制字符数）
_API_KEY_RANDOM_LENGTH = 32

# 默认速率限制：每分钟请求数
_DEFAULT_RATE_LIMIT = 60

# 速率限制窗口（秒）
_RATE_LIMIT_WINDOW = 60


# ---------------------------------------------------------------------------
# 权限枚举
# ---------------------------------------------------------------------------

class Permission(Flag):
    """系统权限枚举，使用Flag支持权限组合。"""

    # 读取权限
    READ_STATS = auto()
    READ_MESSAGES = auto()
    READ_LORAS = auto()
    READ_KNOWLEDGE = auto()

    # 写入权限
    WRITE_CONFIG = auto()
    WRITE_KNOWLEDGE = auto()
    START_TRAINING = auto()

    # 删除权限
    DELETE_LORA = auto()
    DELETE_KNOWLEDGE = auto()
    CANCEL_TRAINING = auto()

    # API调用权限
    GENERATE_REPLY = auto()
    SEARCH_KNOWLEDGE = auto()

    # 用户管理权限
    MANAGE_USERS = auto()

    # 权限组
    READ_ALL = READ_STATS | READ_MESSAGES | READ_LORAS | READ_KNOWLEDGE
    WRITE_ALL = WRITE_CONFIG | WRITE_KNOWLEDGE | START_TRAINING
    DELETE_ALL = DELETE_LORA | DELETE_KNOWLEDGE | CANCEL_TRAINING
    API_ALL = GENERATE_REPLY | SEARCH_KNOWLEDGE
    ALL = READ_ALL | WRITE_ALL | DELETE_ALL | API_ALL | MANAGE_USERS


# ---------------------------------------------------------------------------
# 角色定义
# ---------------------------------------------------------------------------

class Role(Enum):
    """系统角色定义，每个角色关联一组权限。"""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_USER = "api_user"


# 角色-权限映射
_ROLE_PERMISSIONS: dict[Role, Permission] = {
    Role.ADMIN: Permission.ALL,
    Role.OPERATOR: (
        Permission.READ_ALL
        | Permission.WRITE_ALL
        | Permission.API_ALL
    ),
    Role.VIEWER: Permission.READ_ALL,
    Role.API_USER: (
        Permission.READ_STATS
        | Permission.GENERATE_REPLY
        | Permission.SEARCH_KNOWLEDGE
    ),
}


# ---------------------------------------------------------------------------
# 异常类
# ---------------------------------------------------------------------------

class AuthenticationError(Exception):
    """认证失败异常。"""
    pass


class AuthorizationError(Exception):
    """授权失败异常。"""
    pass


class RateLimitError(Exception):
    """速率限制超出异常。"""

    def __init__(self, retry_after: int = 60) -> None:
        self.retry_after = retry_after
        super().__init__(f"请求频率超限，请 {retry_after} 秒后重试")


class APIKeyError(Exception):
    """API Key操作异常。"""
    pass


# ---------------------------------------------------------------------------
# 速率限制器
# ---------------------------------------------------------------------------

class RateLimiter:
    """基于滑动窗口的速率限制器。

    每个API Key独立计数，使用内存存储请求时间戳。

    Attributes:
        _windows: API Key到请求时间戳列表的映射
        _limit: 窗口内允许的最大请求数
        _window_seconds: 窗口时间（秒）
    """

    def __init__(
        self,
        limit: int = _DEFAULT_RATE_LIMIT,
        window_seconds: int = _RATE_LIMIT_WINDOW,
    ) -> None:
        """初始化速率限制器。

        Args:
            limit: 窗口内允许的最大请求数
            window_seconds: 窗口时间（秒）
        """
        self._limit = limit
        self._window_seconds = window_seconds
        self._windows: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()
        logger.info("速率限制器初始化: %d次/%d秒", limit, window_seconds)

    async def check(self, api_key: str) -> bool:
        """检查API Key是否在速率限制内。

        如果超出限制，抛出RateLimitError。

        Args:
            api_key: API Key

        Returns:
            True如果请求被允许

        Raises:
            RateLimitError: 超出速率限制
        """
        async with self._lock:
            now = time.time()
            window_start = now - self._window_seconds

            # 获取或创建请求记录
            if api_key not in self._windows:
                self._windows[api_key] = []

            # 清理过期记录
            self._windows[api_key] = [
                ts for ts in self._windows[api_key] if ts > window_start
            ]

            # 检查限制
            if len(self._windows[api_key]) >= self._limit:
                oldest = self._windows[api_key][0]
                retry_after = int(oldest + self._window_seconds - now) + 1
                logger.warning("API Key %s*** 速率限制触发", api_key[:8])
                raise RateLimitError(retry_after=max(1, retry_after))

            # 记录本次请求
            self._windows[api_key].append(now)
            return True

    def cleanup(self) -> None:
        """清理所有过期的请求记录，释放内存。"""
        now = time.time()
        window_start = now - self._window_seconds
        expired_keys = []

        for key, timestamps in self._windows.items():
            self._windows[key] = [ts for ts in timestamps if ts > window_start]
            if not self._windows[key]:
                expired_keys.append(key)

        for key in expired_keys:
            del self._windows[key]

        logger.debug("速率限制器清理完成，活跃Key数: %d", len(self._windows))


# ---------------------------------------------------------------------------
# 访问控制管理器
# ---------------------------------------------------------------------------

class AccessControlManager:
    """基于角色的访问控制管理器。

    所有 API Key 和审计记录都写入统一 DB adapter（SQLite/PostgreSQL 主库），
    不再自行创建 SQLite 文件或表。
    """

    def __init__(
        self,
        database: Any | None = None,
        rate_limit: int = _DEFAULT_RATE_LIMIT,
    ) -> None:
        """初始化访问控制管理器。

        Args:
            database: 数据库 adapter。默认使用 ``db.adapter.db``。
            rate_limit: 每分钟默认请求限制。
        """
        if database is None:
            from db.adapter import db as database
        self._db = database
        self._rate_limiter = RateLimiter(limit=rate_limit)
        self._custom_rate_limiters: dict[str, RateLimiter] = {}
        logger.info("AccessControlManager 初始化完成（统一 DB adapter）")


    @staticmethod
    def _hash_api_key(api_key: str) -> str:
        """计算API Key的哈希值，使用pbkdf2_hmac加盐哈希。

        使用随机盐 + PBKDF2-HMAC-SHA256 进行安全哈希，存储哈希值而非原始Key。
        """
        salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac('sha256', api_key.encode('utf-8'), salt, 100000)
        return f"pbkdf2:{salt.hex()}:{key.hex()}"

    @staticmethod
    def _verify_api_key(api_key: str, stored_hash: str) -> bool:
        """Verify a key against current PBKDF2 and legacy SHA-256 records."""
        try:
            if stored_hash.startswith("pbkdf2:"):
                _, salt_hex, key_hex = stored_hash.split(":", 2)
                salt = bytes.fromhex(salt_hex)
                key = hashlib.pbkdf2_hmac(
                    "sha256",
                    api_key.encode("utf-8"),
                    salt,
                    100000,
                )
                return hmac.compare_digest(key.hex(), key_hex)
            legacy = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
            return hmac.compare_digest(legacy, stored_hash)
        except (AttributeError, TypeError, ValueError):
            logger.warning("忽略格式损坏的 API Key 哈希记录")
            return False

    # -------------------------------------------------------------------
    # API Key管理
    # -------------------------------------------------------------------

    def _find_matching_key_hash(self, api_key: str) -> str:
        """Resolve the stored hash for a raw API key using the DB adapter."""
        role_name = next((
            role.value
            for role in Role
            if api_key.startswith(f"{_API_KEY_PREFIX}{role.value}_")
        ), "")
        if role_name not in {role.value for role in Role}:
            raise AuthenticationError("无效的API Key")

        role_prefix = f"{_API_KEY_PREFIX}{role_name}_"
        suffix = api_key[len(role_prefix):]
        public_id, separator, _secret = suffix.partition("_")
        candidate_prefix = (
            f"{role_prefix}{public_id}_"
            if separator and len(public_id) == 12 and all(ch in "0123456789abcdef" for ch in public_id)
            else ""
        )

        rows: list[dict[str, Any]] = []
        if candidate_prefix:
            rows.extend(self._db.get_api_key_rows_by_prefix(candidate_prefix))
        # Legacy keys used only the role prefix. Keep verification support.
        rows.extend(self._db.get_api_key_rows_by_prefix(role_prefix))
        matched = next(
            (row for row in rows if self._verify_api_key(api_key, row["key_hash"])),
            None,
        )
        if matched is None:
            raise AuthenticationError("无效的API Key")
        if not matched["is_active"]:
            raise AuthenticationError("API Key已被吊销")
        return matched["key_hash"]

    def create_api_key(
        self,
        role: Role,
        description: Optional[str] = None,
        rate_limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """创建新的API Key并写入主数据库。"""
        public_id = secrets.token_hex(6)
        random_hex = secrets.token_hex(_API_KEY_RANDOM_LENGTH // 2)
        key_prefix = f"{_API_KEY_PREFIX}{role.value}_{public_id}_"
        api_key = f"{key_prefix}{random_hex}"
        key_hash = self._hash_api_key(api_key)

        try:
            record = self._db.create_api_key_record(
                key_hash=key_hash,
                key_prefix=key_prefix,
                role=role.value,
                description=description,
                rate_limit=rate_limit,
            )
            logger.info("API Key创建成功: %s*** (角色: %s)", key_prefix, role.value)
            self._db.add_audit_log(
                api_key_hash=key_hash,
                role=role.value,
                action="create_api_key",
                detail=f"创建了角色为 {role.value} 的API Key",
            )
            return {
                "api_key": api_key,
                "key_prefix": key_prefix,
                "role": role.value,
                "description": description,
                "created_at": record["created_at"],
                "rate_limit": rate_limit,
            }
        except Exception as exc:
            logger.error("创建API Key失败: %s", exc)
            raise APIKeyError("创建 API Key 失败") from exc

    def revoke_api_key(self, api_key: str) -> bool:
        """吊销API Key（通过原始 Key 查找哈希）。"""
        try:
            matched_hash = self._find_matching_key_hash(api_key)
            revoked = self._db.revoke_api_key_by_hash(matched_hash)
            if revoked:
                logger.info("API Key已吊销")
                self._db.add_audit_log(
                    api_key_hash=matched_hash,
                    role="system",
                    action="revoke_api_key",
                    detail="API Key已被吊销",
                )
            return revoked
        except AuthenticationError:
            return False
        except Exception as exc:
            logger.error("吊销API Key失败: %s", exc)
            raise APIKeyError("吊销 API Key 失败") from exc

    def revoke_api_key_by_id(self, key_id: int) -> bool:
        """Revoke a managed key by database id without exposing the secret in a URL."""
        try:
            row = self._db.get_api_key_by_id(key_id)
            if row is None or not row.get("is_active"):
                return False
            revoked = self._db.revoke_api_key_by_id(key_id)
            if revoked:
                self._db.add_audit_log(
                    api_key_hash=row["key_hash"],
                    role=row["role"],
                    action="revoke_api_key",
                    detail=f"吊销 API Key id={key_id}",
                )
            return revoked
        except Exception as exc:
            logger.error("按 ID 吊销 API Key 失败: %s", exc)
            raise APIKeyError("吊销 API Key 失败") from exc

    def list_api_keys(self, include_revoked: bool = False) -> list[dict[str, Any]]:
        """列出所有API Key元数据。"""
        try:
            return self._db.list_api_keys(include_revoked=include_revoked)
        except Exception as exc:
            logger.error("列出API Key失败: %s", exc)
            raise APIKeyError("列出 API Key 失败") from exc

    # -------------------------------------------------------------------
    # 认证与授权
    # -------------------------------------------------------------------

    def _authenticate_record(self, api_key: str) -> dict[str, Any]:
        """Verify and touch one managed API key in a worker thread."""
        key_hash = self._find_matching_key_hash(api_key)
        matched_row = self._db.get_api_key_by_hash(key_hash)
        if matched_row is None:
            raise AuthenticationError("无效的API Key")
        if not matched_row["is_active"]:
            raise AuthenticationError("API Key已被吊销")

        self._db.touch_api_key(key_hash)
        return dict(matched_row)

    async def authenticate(self, api_key: str) -> dict[str, Any]:
        """Authenticate a managed API key without blocking the event loop."""
        if not api_key:
            raise AuthenticationError("缺少API Key")

        try:
            matched_row = await asyncio.to_thread(self._authenticate_record, api_key)
            role = Role(matched_row["role"])
            permissions = _ROLE_PERMISSIONS.get(role, Permission(0))
            stored_hash = matched_row["key_hash"]
            custom_limit = matched_row["rate_limit"]

            if custom_limit:
                limiter = self._custom_rate_limiters.get(stored_hash)
                if limiter is None:
                    limiter = RateLimiter(limit=custom_limit)
                    self._custom_rate_limiters[stored_hash] = limiter
                await limiter.check(stored_hash)
            else:
                await self._rate_limiter.check(stored_hash)

            return {
                "role": role,
                "permissions": permissions,
                "key_hash": stored_hash,
                "rate_limit": custom_limit or _DEFAULT_RATE_LIMIT,
            }
        except (AuthenticationError, RateLimitError):
            raise
        except Exception as exc:
            logger.error("认证过程异常: %s", exc)
            raise AuthenticationError("认证失败") from exc

    def authorize(self, user_info: dict[str, Any], required_permission: Permission) -> bool:
        """检查用户是否拥有所需权限。"""
        user_permissions: Permission = user_info.get("permissions", Permission(0))

        if not (user_permissions & required_permission):
            role = user_info.get("role", Role.VIEWER)
            logger.warning(
                "授权失败: 角色 %s 缺少权限 %s",
                role.value if isinstance(role, Role) else role,
                required_permission.name,
            )
            raise AuthorizationError(
                f"权限不足：需要 {required_permission.name} 权限"
            )

        return True

    # -------------------------------------------------------------------
    # 审计日志
    # -------------------------------------------------------------------

    def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        role: Optional[str] = None,
        action: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """查询统一 DB 中的审计日志。"""
        return self._db.get_audit_logs(limit, offset, role, action)

    def log_action(
        self,
        user_info: dict[str, Any],
        action: str,
        resource: Optional[str] = None,
        detail: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """记录用户操作审计日志到统一 DB。"""
        role = user_info.get("role", Role.VIEWER)
        role_value = role.value if isinstance(role, Role) else str(role)
        key_hash = user_info.get("key_hash", "unknown")

        self._db.add_audit_log(
            api_key_hash=key_hash,
            role=role_value,
            action=action,
            resource=resource,
            detail=detail,
            ip_address=ip_address,
        )


# ---------------------------------------------------------------------------
# FastAPI集成
# ---------------------------------------------------------------------------

def get_current_user() -> Callable:
    """FastAPI依赖注入函数，从请求头获取并验证API Key。

    从X-API-Key请求头读取API Key，执行认证和速率限制。

    Returns:
        依赖注入函数

    使用示例:
        from fastapi import Depends

        @router.get("/stats")
        async def get_stats(user: dict = Depends(get_current_user())):
            ...
    """
    from fastapi import Request

    _manager: Optional[AccessControlManager] = None

    async def _get_user(request: Request) -> dict[str, Any]:
        nonlocal _manager
        if _manager is None:
            _manager = AccessControlManager()

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=401,
                detail="缺少API Key，请在请求头中提供 X-API-Key",
            )

        try:
            user_info = await _manager.authenticate(api_key)
            # 将用户信息存储到request.state供后续使用
            request.state.user = user_info
            return user_info
        except AuthenticationError as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except RateLimitError as exc:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(exc.retry_after)},
            ) from exc

    return _get_user


def require_permission(permission: Permission) -> Callable:
    """权限检查装饰器，用于FastAPI路由函数。

    需要与get_current_user()依赖注入配合使用。

    Args:
        permission: 需要的权限

    Returns:
        装饰器函数

    使用示例:
        @router.delete("/loras/{id}")
        @require_permission(Permission.DELETE_LORA)
        async def delete_lora(id: str, user: dict = Depends(get_current_user())):
            ...
    """
    _manager: Optional[AccessControlManager] = None

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal _manager
            if _manager is None:
                _manager = AccessControlManager()

            # 从kwargs中查找user参数
            user_info = kwargs.get("user")

            # 如果没有在kwargs中，尝试从args中查找
            if user_info is None:
                for arg in args:
                    if isinstance(arg, dict) and "permissions" in arg:
                        user_info = arg
                        break

            # 尝试从request.state获取
            if user_info is None:
                for arg in args:
                    if hasattr(arg, "state") and hasattr(arg.state, "user"):
                        user_info = arg.state.user
                        break

            if user_info is None:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=401,
                    detail="未认证，请先通过API Key认证",
                )

            try:
                _manager.authorize(user_info, permission)

                # 记录审计日志
                _manager.log_action(
                    user_info=user_info,
                    action=func.__name__,
                    resource=str(kwargs.get("id", "")),
                )

            except AuthorizationError as exc:
                from fastapi import HTTPException
                raise HTTPException(status_code=403, detail=str(exc)) from exc

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 全局单例（懒加载）
# ---------------------------------------------------------------------------

_access_control_manager: Optional[AccessControlManager] = None


def get_access_control_manager() -> AccessControlManager:
    """获取全局访问控制管理器单例。

    Returns:
        AccessControlManager实例
    """
    global _access_control_manager
    if _access_control_manager is None:
        _access_control_manager = AccessControlManager()
    return _access_control_manager
