"""Authentication reads and writes the database bound to its application."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.runtime import RuntimeContainer


def _app(database):
    from api.auth import router

    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, startup_env={})
    app.include_router(router)
    return app


@pytest.fixture
def databases(tmp_path, monkeypatch):
    from db.adapter import db as global_db
    from db.database import SQLiteDB

    databases = [SQLiteDB(tmp_path / f"app-{index}.sqlite") for index in range(2)]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Authentication must not access the global database")

    for method in ["get_user_by_username", "add_user", "execute_sql"]:
        monkeypatch.setattr(global_db, method, forbidden)
    monkeypatch.setenv("ENVIRONMENT", "test")
    yield databases
    for database in databases:
        database.close_connection()


async def test_two_applications_keep_registration_login_and_profile_in_their_own_database(databases):
    from api.auth import _verify_password

    first, second = databases
    for index, database in enumerate(databases):
        password = f"synthetic-password-{index}"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(database)), base_url="https://isolated"
        ) as client:
            response = await client.post("/api/auth/register", json={"username": "same-name", "password": password})
            assert response.status_code == 200
            assert response.json()["user"]["username"] == "same-name"
            assert client.cookies.get("access_token")
            client.cookies.clear()
            wrong = await client.post("/api/auth/login", json={"username": "same-name", "password": "wrong-password"})
            assert wrong.status_code == 401
            assert not client.cookies.get("access_token")
            response = await client.post("/api/auth/login", json={"username": "same-name", "password": password})
            assert response.status_code == 200
            profile = await client.get("/api/auth/me")
            assert profile.status_code == 200
            assert profile.json()["user"] == response.json()["user"]
            assert (await client.post("/api/auth/logout")).status_code == 200
            assert (await client.get("/api/auth/me")).status_code == 401
    assert _verify_password("synthetic-password-0", first.get_user_by_username("same-name")["password_hash"])
    assert _verify_password("synthetic-password-1", second.get_user_by_username("same-name")["password_hash"])
    assert not _verify_password("synthetic-password-0", second.get_user_by_username("same-name")["password_hash"])


async def test_production_bootstrap_counts_only_the_bound_database(databases, monkeypatch):
    first, second = databases
    first.add_user("other-app-admin", "unused-hash")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOW_PUBLIC_REGISTRATION", "false")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(second)), base_url="https://isolated"
    ) as client:
        response = await client.post(
            "/api/auth/register", json={"username": "bound-admin", "password": "synthetic-password"}
        )
        assert response.status_code == 200
        assert response.json()["user"]["role"] == "admin"
        response = await client.post(
            "/api/auth/register", json={"username": "another-user", "password": "synthetic-password"}
        )
        assert response.status_code == 403
    assert [row["username"] for row in second.execute_sql("SELECT username FROM users")] == ["bound-admin"]


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("POST", "/api/auth/login", {"username": "scoped", "password": "synthetic-password"}),
        ("POST", "/api/auth/register", {"username": "scoped", "password": "synthetic-password"}),
        ("GET", "/api/auth/me", None),
    ],
)
async def test_missing_application_container_fails_without_global_database(method, path, payload, databases):
    from api.auth import router
    from app.config import create_access_token

    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="https://isolated"
    ) as client:
        response = await client.request(
            method, path, json=payload, headers={"Authorization": "Bearer " + create_access_token("scoped", 1, "user")}
        )
    assert response.status_code == 500


async def test_login_database_failure_is_not_an_authentication_result(databases):
    def fail(_username):
        raise RuntimeError("synthetic database unavailable")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(SimpleNamespace(get_user_by_username=fail)), raise_app_exceptions=False),
        base_url="https://isolated",
    ) as client:
        response = await client.post("/api/auth/login", json={"username": "scoped", "password": "synthetic-password"})
    assert response.status_code == 500
    assert not client.cookies.get("access_token")


def test_missing_registration_count_is_not_an_empty_database(monkeypatch):
    from api.auth import _registration_allowed

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOW_PUBLIC_REGISTRATION", "false")
    with pytest.raises(IndexError):
        _registration_allowed(SimpleNamespace(execute_sql=lambda _query: []))
