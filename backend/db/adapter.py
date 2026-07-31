"""数据库适配器 - 统一访问接口

根据环境变量自动选择数据库后端：
- USE_POSTGRESQL=true，或未显式设置且 DATABASE_URL 为 PostgreSQL URL → PostgreSQL
- USE_POSTGRESQL=false → SQLite

所有 API 层应通过 `from db.adapter import db` 获取数据库实例。
"""
import os
import logging

from interfaces import DatabaseInterface

logger = logging.getLogger(__name__)

def _should_use_postgresql(env=os.environ) -> bool:
    """Resolve the database backend without silently ignoring DATABASE_URL."""
    explicit = str(env.get("USE_POSTGRESQL", "")).strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}

    database_url = str(env.get("DATABASE_URL", "")).strip().lower()
    if database_url.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
        return True
    return all(
        str(env.get(key, "")).strip()
        for key in ("PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DATABASE")
    )


_USE_PG = _should_use_postgresql()

if _USE_PG:
    try:
        from db.pg_database import sync_pg_db
        db = sync_pg_db
        logger.info("✅ 数据库适配器：PostgreSQL 模式（SyncPgAdapter）")
    except ImportError as e:
        if os.getenv("ENVIRONMENT", "development").strip().lower() == "production":
            raise RuntimeError("PostgreSQL 已配置，但 PostgreSQL 依赖不可用") from e
        logger.warning(f"PostgreSQL 模块导入失败，开发环境回退到 SQLite: {e}")
        from db.database import db
        _USE_PG = False
else:
    from db.database import db

# 接口契约验证：确保 db 实例实现 DatabaseInterface 接口
assert isinstance(db, DatabaseInterface), f"Database adapter must implement DatabaseInterface, got {type(db)}"


def get_db():
    """获取当前活跃的数据库实例"""
    return db


def is_pg_mode() -> bool:
    """当前是否使用 PostgreSQL"""
    return _USE_PG
