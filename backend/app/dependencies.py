"""FastAPI 依赖注入"""

import os

from fastapi import Request, HTTPException
from infra.environment import read_bool
from app.config import verify_token
from app.runtime import get_runtime_container
from infra.auth_work import run_auth_database
from infra.bounded_executor import BlockingWorkRejected, BlockingWorkTimeout
from infra.observability import increment


def get_request_runtime(request: Request):
    return get_runtime_container(request.app)


def get_request_http_client_pool(request: Request):
    pool = get_request_runtime(request).http_client_pool
    if pool is None:
        raise HTTPException(503, "应用 HTTP 连接池未初始化，请检查服务启动状态")
    return pool


def get_request_database(request: Request):
    return get_runtime_container(request.app).db


async def get_current_user(request: Request) -> dict:
    """从请求中提取并验证当前用户（支持 Authorization 头、Cookie 和中间件预验证）

    返回 dict 包含：user_id, username, role。
    role 来自 JWT payload（旧 token 缺失时默认 "user"）；如需强一致请用 get_current_admin。
    """
    payload = getattr(request.state, "jwt_payload", None)
    if not payload:
        api_key_payload = getattr(request.state, "api_key_payload", None)
        if isinstance(api_key_payload, dict):
            return {
                "user_id": None,
                "username": getattr(request.state, "user", "api_key_user"),
                "role": api_key_payload.get("role", "api_user"),
                "permissions": api_key_payload.get("permissions"),
                "auth_type": getattr(request.state, "auth_type", "api_key"),
            }
        auth_header = request.headers.get("Authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else None
        token = token or request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="缺少认证 Token")
        payload = verify_token(token)

    # 检查 Token 是否已被注销
    from infra.token_revocation import is_token_revoked

    if is_token_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Token 已注销，请重新登录")

    return {
        "user_id": payload.get("user_id"),
        "username": payload["sub"],
        "role": payload.get("role", "user"),
    }


async def get_current_admin(request: Request) -> dict:
    """要求当前用户具有 admin 角色。

    C-S1 fix: 敏感路由（系统配置、模型管理、备份恢复、训练任务等）的访问控制依赖。
    为防止 token 中 role 与 DB 不一致（例如管理员被降级后旧 token 仍持 admin），
    这里在 get_current_user 之后做一次 DB 复核；DB 不可达时按最小权限原则拒绝。
    """
    user = await get_current_user(request)
    # Managed API keys are authenticated against their own revocation store.
    # Static environment keys intentionally remain api_user and cannot administer.
    if user.get("auth_type") in {"api_key", "managed_api_key"}:
        if user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="需要管理员权限")
        return user

    # Revalidate the current role against this application's database.
    try:
        database = get_request_database(request)

        row = await run_auth_database(
            database.get_user_by_username,
            user.get("username", ""),
        )
        current_role = (row or {}).get("role", "user")
    except (BlockingWorkRejected, BlockingWorkTimeout) as exc:
        increment("auth_database_admin_unavailable")
        raise HTTPException(
            status_code=503,
            detail="Authentication service is busy; retry later",
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="无法验证用户权限，请稍后重试") from exc

    if current_role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")

    # 用 DB 中的最新 role 覆盖 token 中的可能过期值
    user["role"] = current_role
    return user


def require_narrative_enabled():
    if not read_bool(os.environ, "NARRATIVE_BRANCHES_ENABLED"):
        raise HTTPException(404, "假想分支功能尚未启用")
