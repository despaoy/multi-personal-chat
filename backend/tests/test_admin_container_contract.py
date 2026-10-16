"""Administrator authorization must use the actual application's database."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends, FastAPI

from app.config import create_access_token
from app.dependencies import get_current_admin
from app.runtime import RuntimeContainer


def _app():
    app = FastAPI()

    @app.get("/admin-probe")
    async def protected(user=Depends(get_current_admin)):
        return {"role": user["role"]}

    return app


@pytest.fixture
def global_database_forbidden(monkeypatch):
    from db.adapter import db

    calls = []

    def forbidden(_username):
        calls.append(_username)
        raise AssertionError("Administrator authorization must not use the global database")

    monkeypatch.setattr(db, "get_user_by_username", forbidden)
    return calls


@pytest.mark.parametrize("container", [None, object()], ids=["missing", "invalid"])
async def test_real_request_without_valid_container_fails_without_fallback(container, global_database_forbidden):
    app = _app()
    if container is not None:
        app.state.runtime_container = container
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://isolated") as client:
        response = await client.get(
            "/admin-probe", headers={"Authorization": "Bearer " + create_access_token("synthetic-admin", 1, "admin")}
        )
    assert response.status_code == 503
    assert global_database_forbidden == []


async def test_current_container_user_can_be_missing_without_becoming_a_database_error(global_database_forbidden):
    app = _app()
    app.state.runtime_container = RuntimeContainer(
        db=SimpleNamespace(get_user_by_username=lambda _username: None), is_pg_mode=lambda: False
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://isolated") as client:
        response = await client.get(
            "/admin-probe", headers={"Authorization": "Bearer " + create_access_token("removed-admin", 1, "admin")}
        )
    assert response.status_code == 403
    assert global_database_forbidden == []


async def test_database_failure_preserves_its_cause_and_never_uses_another_database(global_database_forbidden):
    cause = RuntimeError("synthetic database unavailable")

    def fail(_username):
        raise cause

    container = RuntimeContainer(db=SimpleNamespace(get_user_by_username=fail), is_pg_mode=lambda: False)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(runtime_container=container)),
        state=SimpleNamespace(
            jwt_payload={"sub": "synthetic-admin", "user_id": 1, "role": "admin", "jti": "synthetic-admin"}
        ),
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        await get_current_admin(request)
    assert raised.value.status_code == 503
    assert raised.value.__cause__ is cause
    assert global_database_forbidden == []


@pytest.mark.parametrize(
    "auth_type,role,status",
    [("managed_api_key", "admin", 200), ("managed_api_key", "user", 403), ("api_key", "api_user", 403)],
)
async def test_already_verified_api_key_authority_remains_independent(
    auth_type, role, status, global_database_forbidden
):
    app = _app()

    @app.middleware("http")
    async def verified_key(request, call_next):
        request.state.api_key_payload = {"role": role}
        request.state.auth_type = auth_type
        request.state.user = "synthetic-key-owner"
        return await call_next(request)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://isolated") as client:
        response = await client.get("/admin-probe")
    assert response.status_code == status
    assert global_database_forbidden == []
