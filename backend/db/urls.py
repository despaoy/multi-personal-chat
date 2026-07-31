"""PostgreSQL URL normalization shared by runtime and migrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy.engine import URL


def normalize_async_database_url(database_url: str) -> str:
    """Normalize common PostgreSQL schemes to SQLAlchemy's asyncpg dialect."""
    value = str(database_url or "").strip()
    if value.startswith("postgres://"):
        return "postgresql+asyncpg://" + value[len("postgres://"):]
    if value.startswith("postgresql://"):
        return "postgresql+asyncpg://" + value[len("postgresql://"):]
    return value


def _component_database_url(env: Mapping[str, str]) -> str:
    password = env.get("PG_PASSWORD", "")
    if not password:
        return ""

    url = URL.create(
        drivername="postgresql+asyncpg",
        username=env.get("PG_USER", "qqassistant"),
        password=password,
        host=env.get("PG_HOST", "localhost"),
        port=int(env.get("PG_PORT", "5432")),
        database=env.get("PG_DATABASE", "qqassistant"),
    )
    return url.render_as_string(hide_password=False)


def resolve_runtime_database_url(
    database_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve an explicit URL first, then safely build one from PG_* values."""
    values = os.environ if env is None else env
    explicit = database_url or values.get("DATABASE_URL")
    if explicit:
        return normalize_async_database_url(explicit)
    return _component_database_url(values)


def resolve_alembic_database_url(env: Mapping[str, str] | None = None) -> str:
    """Resolve migrations like runtime, with ALEMBIC_DATABASE_URL as override."""
    values = os.environ if env is None else env
    override = values.get("ALEMBIC_DATABASE_URL")
    if override:
        return normalize_async_database_url(override)
    return resolve_runtime_database_url(env=values)