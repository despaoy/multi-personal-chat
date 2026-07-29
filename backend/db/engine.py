"""统一数据库引擎管理

根据环境变量创建 SQLAlchemy 引擎：
- SQLite：同步引擎，适用于开发和单机部署
- PostgreSQL：异步引擎（asyncpg），适用于生产多实例部署

引擎创建后会自动调用 Base.metadata.create_all() 确保表存在。
注意：生产环境应使用 alembic 迁移管理 schema，create_all 仅用于开发初始化。
"""

import os
import logging
from typing import Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db.models import Base

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_async_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _resolve_sqlite_url() -> str:
    """构建 SQLite 连接字符串"""
    db_path = os.getenv("SQLITE_PATH", os.path.join(os.getcwd(), "data", "qqassistant.db"))
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return f"sqlite:///{db_path}"


def _resolve_pg_url() -> str:
    """构建 PostgreSQL 异步连接字符串"""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = "postgresql+asyncpg://" + db_url[len("postgres://"):]
        elif db_url.startswith("postgresql://"):
            db_url = "postgresql+asyncpg://" + db_url[len("postgresql://"):]
        return db_url
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_user = os.getenv("PG_USER", "qqassistant")
    pg_password = os.getenv("PG_PASSWORD", "")
    pg_database = os.getenv("PG_DATABASE", "qqassistant")
    return f"postgresql+asyncpg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"


def _should_use_postgresql(env=os.environ) -> bool:
    """判断是否使用 PostgreSQL"""
    explicit = str(env.get("USE_POSTGRESQL", "")).strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    database_url = str(env.get("DATABASE_URL", "")).strip().lower()
    return database_url.startswith(("postgresql://", "postgresql+asyncpg://"))


def is_pg_mode() -> bool:
    """当前是否使用 PostgreSQL"""
    return _should_use_postgresql()


def get_engine() -> Engine:
    """获取同步 SQLAlchemy 引擎（SQLite 模式）"""
    global _engine
    if _engine is not None:
        return _engine
    url = _resolve_sqlite_url()
    _engine = create_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(_engine)
    logger.info(f"SQLite 引擎已创建: {url}")
    return _engine


async def get_async_engine() -> AsyncEngine:
    """获取异步 SQLAlchemy 引擎（PostgreSQL 模式）"""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        return _async_engine
    url = _resolve_pg_url()
    _async_engine = create_async_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
    )
    _async_session_factory = async_sessionmaker(
        _async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"PostgreSQL 异步引擎已创建: {url}")
    return _async_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """获取异步会话工厂（PostgreSQL 模式）"""
    if _async_session_factory is None:
        raise RuntimeError("Async session factory not initialized. Call get_async_engine() first.")
    return _async_session_factory


async def close_engines() -> None:
    """关闭所有引擎，应在应用 shutdown 阶段调用"""
    global _engine, _async_engine, _async_session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
        logger.info("SQLite 引擎已关闭")
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
        logger.info("PostgreSQL 异步引擎已关闭")
