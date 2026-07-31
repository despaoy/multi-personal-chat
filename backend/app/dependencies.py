"""FastAPI 依赖注入"""
from fastapi import Request, HTTPException
from app.config import verify_token
from infra.auth_work import run_auth_database
from infra.bounded_executor import BlockingWorkRejected, BlockingWorkTimeout
from infra.observability import increment


async def get_current_user(request: Request) -> dict:
    """从请求中提取并验证当前用户（支持 Authorization 头、Cookie 和中间件预验证）

    返回 dict 包含：user_id, username, role。
    role 来自 JWT payload（旧 token 缺失时默认 "user"）；如需强一致请用 get_current_admin。
    """
    # 1. 如果安全中间件已验证，直接使用
    if hasattr(request.state, "jwt_payload") and request.state.jwt_payload:
        payload = request.state.jwt_payload
        # 检查 Token 是否已被注销
        from api.auth import is_token_revoked
        if is_token_revoked(payload.get("jti", "")):
            raise HTTPException(status_code=401, detail="Token 已注销，请重新登录")
        return {
            "user_id": payload.get("user_id"),
            "username": payload.get("sub", "unknown"),
            "role": payload.get("role", "user"),
        }

    api_key_payload = getattr(request.state, "api_key_payload", None)
    if isinstance(api_key_payload, dict):
        return {
            "user_id": None,
            "username": getattr(request.state, "user", "api_key_user"),
            "role": api_key_payload.get("role", "api_user"),
            "permissions": api_key_payload.get("permissions"),
            "auth_type": getattr(request.state, "auth_type", "api_key"),
        }
    token = None

    # 2. 从 Authorization 头获取
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

    # 3. 从 Cookie 获取
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="缺少认证 Token")

    payload = verify_token(token)

    # 检查 Token 是否已被注销
    from api.auth import is_token_revoked
    if is_token_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Token 已注销，请重新登录")

    return {
        "user_id": payload.get("user_id"),
        "username": payload.get("sub", "unknown"),
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

    # DB 复核当前用户角色。真实请求使用应用容器；最小化单元测试
    # 没有 request.app 时保留全局适配器回退。
    try:
        try:
            from app.runtime import get_runtime_container

            database = get_runtime_container(request.app).db
        except (AttributeError, RuntimeError):
            from db.adapter import db as database

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
    except Exception:
        # DB 不可达时按最小权限原则拒绝，避免误授权
        raise HTTPException(status_code=503, detail="无法验证用户权限，请稍后重试")

    if current_role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")

    # 用 DB 中的最新 role 覆盖 token 中的可能过期值
    user["role"] = current_role
    return user
