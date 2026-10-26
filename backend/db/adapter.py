"""数据库适配器 - 统一访问接口

根据环境变量自动选择数据库后端：
- USE_POSTGRESQL=true，或未显式设置且 DATABASE_URL 为 PostgreSQL URL → PostgreSQL
- USE_POSTGRESQL=false → SQLite

所有 API 层应通过 `from db.adapter import db` 获取数据库实例。
"""
import logging
import os

from infra.environment import read_bool
from interfaces import DatabaseInterface

logger = logging.getLogger(__name__)

def _should_use_postgresql(env=os.environ) -> bool:
    """Resolve the database backend without silently ignoring DATABASE_URL."""
    if "USE_POSTGRESQL" in env:
        return read_bool(env, "USE_POSTGRESQL")

    database_url = str(env.get("DATABASE_URL", "")).strip().lower()
    if database_url:
        if database_url.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
            return True
        if database_url.startswith("sqlite:///"):
            return False
        raise ValueError("DATABASE_URL must use a PostgreSQL or SQLite URL")

    components = [str(env.get(key, "")).strip() for key in ("PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DATABASE")]
    if any(components) and not all(components):
        raise ValueError("PostgreSQL requires PG_HOST, PG_USER, PG_PASSWORD and PG_DATABASE together")
    return all(components)


_USE_PG = _should_use_postgresql()

if _USE_PG:
    try:
        from db.pg_database import sync_pg_db
        db = sync_pg_db
        logger.info("✅ 数据库适配器：PostgreSQL 模式（SyncPgAdapter）")
    except ImportError as exc:
        raise RuntimeError("PostgreSQL 已配置，但依赖不可用；请安装后端依赖后重启") from exc

else:
    from db.database import SQLiteDB

    db = SQLiteDB()

# 接口契约验证：确保 db 实例实现 DatabaseInterface 接口
assert isinstance(db, DatabaseInterface), f"Database adapter must implement DatabaseInterface, got {type(db)}"


def is_pg_mode() -> bool:
    """当前是否使用 PostgreSQL"""
    return _USE_PG
