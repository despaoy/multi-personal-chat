"""Process-local token revocation shared by authentication boundaries."""

import time


# Token 黑名单（内存 TTL，服务重启清空）
# 存储已注销的 jti → 过期时间戳
_TOKEN_BLACKLIST: dict[str, float] = {}


def revoke_token(token: str) -> None:
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
    except pyjwt.InvalidTokenError:
        return  # Invalid tokens have no authenticated identity to revoke.


def _cleanup_blacklist() -> None:
    """清理已过期的黑名单条目"""
    now = time.time()
    expired = [jti for jti, exp in _TOKEN_BLACKLIST.items() if exp <= now]
    for jti in expired:
        del _TOKEN_BLACKLIST[jti]


def is_token_revoked(jti: str) -> bool:
    """检查 Token 是否已被注销"""
    return jti in _TOKEN_BLACKLIST and _TOKEN_BLACKLIST[jti] > time.time()
