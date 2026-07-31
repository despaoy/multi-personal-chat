from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from infra.bounded_executor import BlockingWorkRejected, BlockingWorkTimeout
from infra.auth_work import run_auth_database


@pytest.mark.asyncio
async def test_auth_database_work_uses_dedicated_executor():
    thread_name = await run_auth_database(lambda: threading.current_thread().name)

    assert thread_name.startswith("auth-database")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        BlockingWorkRejected("full"),
        BlockingWorkTimeout("slow"),
    ],
)
async def test_auth_database_backpressure_returns_retryable_503(
    monkeypatch,
    error,
):
    from api import auth

    async def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(auth, "run_auth_database", fail)

    with pytest.raises(HTTPException) as raised:
        await auth._run_database_work(lambda: None)

    assert raised.value.status_code == 503
    assert raised.value.headers == {"Retry-After": "1"}


@pytest.mark.asyncio
async def test_admin_database_check_uses_application_runtime_container(monkeypatch):
    from api import auth
    from app import dependencies
    from app.runtime import RuntimeContainer
    from db import adapter

    calls: list[str] = []

    class Database:
        def get_user_by_username(self, username):
            calls.append(username)
            return {"id": 1, "username": username, "role": "admin"}

    container = RuntimeContainer(
        db=Database(),
        is_pg_mode=lambda: False,
        startup_env={},
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(runtime_container=container)),
        state=SimpleNamespace(
            jwt_payload={
                "sub": "container-admin",
                "user_id": 1,
                "role": "admin",
                "jti": "container-admin-jti",
            }
        ),
        headers={},
        cookies={},
    )
    monkeypatch.setattr(auth, "is_token_revoked", lambda _jti: False)

    monkeypatch.setattr(
        adapter.db,
        "get_user_by_username",
        lambda _username: (_ for _ in ()).throw(
            AssertionError("global database adapter must not be used")
        ),
    )

    result = await dependencies.get_current_admin(request)

    assert result["role"] == "admin"
    assert calls == ["container-admin"]

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        BlockingWorkRejected("full"),
        BlockingWorkTimeout("slow"),
    ],
)
async def test_admin_database_backpressure_returns_retryable_503(
    monkeypatch,
    error,
):
    from api import auth
    from app import dependencies

    async def fail(*_args, **_kwargs):
        raise error

    request = SimpleNamespace(
        state=SimpleNamespace(
            jwt_payload={
                "sub": "admin",
                "user_id": 1,
                "role": "admin",
                "jti": "admin-jti",
            }
        ),
        headers={},
        cookies={},
    )
    monkeypatch.setattr(auth, "is_token_revoked", lambda _jti: False)
    monkeypatch.setattr(dependencies, "run_auth_database", fail)

    with pytest.raises(HTTPException) as raised:
        await dependencies.get_current_admin(request)

    assert raised.value.status_code == 503
    assert raised.value.headers == {"Retry-After": "1"}


@pytest.mark.asyncio
async def test_admin_database_check_does_not_run_on_event_loop(monkeypatch):
    from api import auth
    from app import dependencies
    from db import adapter

    event_loop_thread = threading.get_ident()
    database_threads: list[int] = []

    def get_user(username):
        database_threads.append(threading.get_ident())
        return {"id": 1, "username": username, "role": "admin"}

    request = SimpleNamespace(
        state=SimpleNamespace(
            jwt_payload={
                "sub": "admin",
                "user_id": 1,
                "role": "admin",
                "jti": "admin-jti",
            }
        ),
        headers={},
        cookies={},
    )
    monkeypatch.setattr(auth, "is_token_revoked", lambda _jti: False)
    monkeypatch.setattr(adapter.db, "get_user_by_username", get_user)

    result = await dependencies.get_current_admin(request)

    assert result["role"] == "admin"
    assert database_threads
    assert all(thread_id != event_loop_thread for thread_id in database_threads)
