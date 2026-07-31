"""Database URL resolution contracts shared by runtime and Alembic."""

from db.urls import (
    normalize_async_database_url,
    resolve_alembic_database_url,
    resolve_runtime_database_url,
)


def test_normalize_common_postgresql_schemes():
    assert normalize_async_database_url("postgres://u:p@db/app") == (
        "postgresql+asyncpg://u:p@db/app"
    )
    assert normalize_async_database_url("postgresql://u:p@db/app") == (
        "postgresql+asyncpg://u:p@db/app"
    )
    async_url = "postgresql+asyncpg://u:p@db/app"
    assert normalize_async_database_url(async_url) == async_url


def test_alembic_url_precedence_matches_documented_contract():
    env = {
        "ALEMBIC_DATABASE_URL": "postgres://migration:m@migration/db",
        "DATABASE_URL": "postgres://runtime:r@runtime/db",
        "PG_PASSWORD": "component-password",
    }
    assert resolve_alembic_database_url(env) == (
        "postgresql+asyncpg://migration:m@migration/db"
    )

    env.pop("ALEMBIC_DATABASE_URL")
    assert resolve_alembic_database_url(env) == (
        "postgresql+asyncpg://runtime:r@runtime/db"
    )


def test_runtime_url_prefers_explicit_value_over_components():
    env = {
        "DATABASE_URL": "postgres://runtime:password@runtime/db",
        "PG_PASSWORD": "component-password",
    }
    assert resolve_runtime_database_url(env=env) == (
        "postgresql+asyncpg://runtime:password@runtime/db"
    )
    assert resolve_runtime_database_url(
        "postgresql://argument:password@argument/db", env=env
    ) == "postgresql+asyncpg://argument:password@argument/db"


def test_component_database_url_escapes_credentials():
    url = resolve_alembic_database_url(
        {
            "PG_USER": "app-user",
            "PG_PASSWORD": "p@:%/word",
            "PG_HOST": "postgres",
            "PG_PORT": "5433",
            "PG_DATABASE": "qqchat",
        }
    )

    assert url == (
        "postgresql+asyncpg://app-user:p%40%3A%25%2Fword@postgres:5433/qqchat"
    )


def test_component_database_url_requires_password():
    assert resolve_alembic_database_url({"PG_HOST": "postgres"}) == ""