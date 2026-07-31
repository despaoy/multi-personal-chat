"""用户认证API"""

import asyncio
import logging
import os
import time
from fastapi import APIRouter, HTTPException, Depends, Response, Request

from db.schemas import RegisterRequest, LoginRequest
from db.adapter import db
from db.errors import RegistrationClosedError
from app.config import create_access_token
from app.dependencies import get_current_user
from infra.auth_work import run_auth_database
from infra.bounded_executor import (
    BlockingWorkRejected,
    BlockingWorkTimeout,
    BoundedThreadExecutor,
)
from infra.observability import increment

router = APIRouter()
logger = logging.getLogger(__name__)
_registration_lock: asyncio.Lock | None = None
_registration_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_registration_lock() -> asyncio.Lock:
    """Return a lock owned by the active application event loop."""
    global _registration_lock, _registration_lock_loop
    loop = asyncio.get_running_loop()
    if _registration_lock is None or _registration_lock_loop is not loop:
        _registration_lock = asyncio.Lock()
        _registration_lock_loop = loop
    return _registration_lock


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


_password_executor = BoundedThreadExecutor(
    name="auth-password",
    max_workers=_positive_int_env("AUTH_PASSWORD_WORKERS", 2),
    max_pending=_positive_int_env("AUTH_PASSWORD_MAX_PENDING", 8),
    default_timeout=_positive_float_env("AUTH_PASSWORD_TIMEOUT_SECONDS", 10.0),
)


async def _run_password_work(func, /, *args):
    try:
        return await _password_executor.run(func, *args)
    except BlockingWorkRejected as exc:
        increment("auth_password_rejected")
        raise HTTPException(
            status_code=503,
            detail="Authentication service is busy; retry later",
            headers={"Retry-After": "1"},
        ) from exc
    except BlockingWorkTimeout as exc:
        increment("auth_password_timeout")
        raise HTTPException(
            status_code=503,
            detail="Authentication service is busy; retry later",
            headers={"Retry-After": "1"},
        ) from exc


async def _run_database_work(func, /, *args):
    try:
        return await run_auth_database(func, *args)
    except BlockingWorkRejected as exc:
        increment("auth_database_rejected")
        raise HTTPException(
            status_code=503,
            detail="Authentication service is busy; retry later",
            headers={"Retry-After": "1"},
        ) from exc
    except BlockingWorkTimeout as exc:
        increment("auth_database_timeout")
        raise HTTPException(
            status_code=503,
            detail="Authentication service is busy; retry later",
            headers={"Retry-After": "1"},
        ) from exc


# Token 黑名单（内存 TTL，服务重启清空）
# 存储已注销的 jti → 过期时间戳
_TOKEN_BLACKLIST: dict[str, float] = {}
_BLACKLIST_MAX_SIZE = 10000


def _revoke_token(token: str) -> None:
    """将 Token 加入黑名单，有效期为其剩余 JWT 寿命"""
    import jwt as pyjwt
    from app.config import JWT_SECRET, JWT_ALGORITHM
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM],
                               options={"verify_exp": False})
        jti = payload.get("jti", "")
        exp = payload.get("exp", 0)
        if jti and exp > time.time():
            _TOKEN_BLACKLIST[jti] = exp
            # 清理过期黑名单条目
            _cleanup_blacklist()
    except Exception:
        pass  # token 无效则无需加入黑名单


def _cleanup_blacklist() -> None:
    """清理已过期的黑名单条目"""
    now = time.time()
    expired = [jti for jti, exp in _TOKEN_BLACKLIST.items() if exp <= now]
    for jti in expired:
        del _TOKEN_BLACKLIST[jti]
    # 防止内存无限增长
    if len(_TOKEN_BLACKLIST) > _BLACKLIST_MAX_SIZE:
        oldest = sorted(_TOKEN_BLACKLIST.items(), key=lambda x: x[1])[:len(_TOKEN_BLACKLIST) // 2]
        for jti, _ in oldest:
            del _TOKEN_BLACKLIST[jti]


def is_token_revoked(jti: str) -> bool:
    """检查 Token 是否已被注销"""
    return jti in _TOKEN_BLACKLIST and _TOKEN_BLACKLIST[jti] > time.time()


def _bootstrap_registration_only() -> bool:
    """Whether production permits only the first, bootstrap administrator."""
    if os.getenv("ENVIRONMENT", "development").strip().lower() != "production":
        return False
    return os.getenv("ALLOW_PUBLIC_REGISTRATION", "false").strip().lower() not in {
        "1", "true", "yes", "on",
    }


def _registration_allowed() -> bool:
    if not _bootstrap_registration_only():
        return True

    rows = db.execute_sql("SELECT COUNT(*) AS count FROM users")
    user_count = int(rows[0]["count"]) if rows else 0
    return user_count == 0

def _hash_password(password: str) -> str:
    """使用 bcrypt 哈希密码（自动加盐）"""
    import bcrypt

    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify a bcrypt password outside the event-loop thread."""
    import bcrypt

    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _is_integrity_error(exc: BaseException) -> bool:
    """Recognize DB integrity errors without coupling the API to one backend."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "IntegrityError":
            return True
        current = current.__cause__ or current.__context__
    return False


def _set_auth_cookie(response: Response, token: str):
    """设置 httpOnly Cookie 存储 JWT Token"""
    is_production = os.getenv("ENVIRONMENT", "development").strip().lower() == "production"
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=is_production,  # 生产环境启用 Secure 标志（需HTTPS）
        samesite="lax",
        max_age=86400,  # 24小时，与JWT_EXPIRY_HOURS一致
        path="/",
    )


@router.post("/api/auth/register")
async def register(request: RegisterRequest, response: Response):
    """用户注册"""
    try:
        async with _get_registration_lock():
            bootstrap_only = _bootstrap_registration_only()
            if not await _run_database_work(_registration_allowed):
                raise HTTPException(status_code=403, detail="生产环境已关闭公开注册")

            existing = await _run_database_work(db.get_user_by_username, request.username)
            if existing:
                raise HTTPException(status_code=409, detail="用户名已存在")

            password_hash = await _run_password_work(_hash_password, request.password)
            user = await _run_database_work(db.add_user, request.username, password_hash, bootstrap_only)

        user_id = user["id"]
        now = user["created_at"]
        # C-S1 fix: 把 role 写入 JWT，让 get_current_user 无需 DB 查询即可返回 role
        token = create_access_token(request.username, user_id, user.get("role", "user"))
        _set_auth_cookie(response, token)
        return {
            "success": True,
            "user": {"id": user_id, "username": request.username, "created_at": now, "role": user.get("role", "user")}
        }
    except HTTPException:
        raise
    except RegistrationClosedError as exc:
        raise HTTPException(status_code=403, detail="生产环境已关闭公开注册") from exc
    except Exception as exc:
        if _is_integrity_error(exc):
            raise HTTPException(status_code=409, detail="用户名已存在") from exc
        logger.exception("User registration failed")
        raise HTTPException(status_code=500, detail="注册失败，请稍后重试") from exc


@router.post("/api/auth/login")
async def login(request: LoginRequest, response: Response):
    """用户登录"""
    try:
        row = await _run_database_work(db.get_user_by_username, request.username)
        if not row:
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        if not await _run_password_work(_verify_password, request.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        token = create_access_token(row['username'], row['id'], row.get('role', 'user'))
        _set_auth_cookie(response, token)
        return {
            "success": True,
            "user": {
                "id": row['id'],
                "username": row['username'],
                "created_at": row['created_at'],
                "role": row.get('role', 'user'),
            }
        }
    except HTTPException:
        raise


@router.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    """用户登出 - 清除 Cookie + Token 黑名单吊销"""
    # 从 Cookie 或 Authorization 头提取 token 并加入黑名单
    token = request.cookies.get("access_token", "")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if token:
        _revoke_token(token)
    response.delete_cookie(key="access_token", path="/")
    return {"success": True}


@router.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    user_row = await _run_database_work(db.get_user_by_username, current_user["username"])
    return {
        "success": True,
        "user": {
            "id": current_user["user_id"],
            "username": current_user["username"],
            "created_at": user_row["created_at"] if user_row else "",
            "role": (user_row or {}).get("role", current_user.get("role", "user")),
        }
    }
