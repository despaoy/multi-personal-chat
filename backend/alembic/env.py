"""Alembic 环境配置 - 支持 PostgreSQL 异步迁移

从 db/models.py 读取 target_metadata，确保迁移与 ORM 模型定义一致。
"""

import os
import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# 确保 backend/ 在 sys.path 中，以便导入 db.models
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 从环境变量覆盖数据库URL
db_url = os.getenv("ALEMBIC_DATABASE_URL")
if not db_url:
    # 兼容 Dockerfile 中的 PG_* 环境变量，自动构建连接串
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_user = os.getenv("PG_USER", "qqassistant")
    pg_password = os.getenv("PG_PASSWORD", "")
    pg_database = os.getenv("PG_DATABASE", "qqassistant")
    if pg_password:
        db_url = f"postgresql+asyncpg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# 单一真相源：db/models.py 的 metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式 - 生成SQL脚本而不连接数据库"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步模式 - 使用 asyncpg 连接 PostgreSQL"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式 - 连接数据库执行迁移"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
