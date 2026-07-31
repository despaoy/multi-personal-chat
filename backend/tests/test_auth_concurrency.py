from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import Response

from db.schemas import LoginRequest, RegisterRequest


@pytest.mark.asyncio
async def test_login_offloads_database_and_bcrypt_work(monkeypatch):
    from api import auth

    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def get_user(username: str):
        worker_threads.append(threading.get_ident())
        return {
            "id": 1,
            "username": username,
            "password_hash": "stored-hash",
            "created_at": "2026-01-01T00:00:00",
            "role": "user",
        }

    def verify_password(password: str, password_hash: str) -> bool:
        worker_threads.append(threading.get_ident())
        return password == "secret" and password_hash == "stored-hash"

    monkeypatch.setattr(auth, "db", SimpleNamespace(get_user_by_username=get_user))
    monkeypatch.setattr(auth, "_verify_password", verify_password)
    monkeypatch.setattr(auth, "create_access_token", lambda *args: "token")

    result = await auth.login(LoginRequest(username="alice", password="secret"), Response())

    assert result["success"] is True
    assert len(worker_threads) == 2
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)


@pytest.mark.asyncio
async def test_register_offloads_all_blocking_steps(monkeypatch):
    from api import auth

    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def record(value):
        worker_threads.append(threading.get_ident())
        return value

    fake_db = SimpleNamespace(
        get_user_by_username=lambda username: record(None),
        add_user=lambda username, password_hash, bootstrap_only=False: record({
            "id": 2,
            "username": username,
            "created_at": "2026-01-01T00:00:00",
            "role": "user",
        }),
    )

    monkeypatch.setattr(auth, "db", fake_db)
    monkeypatch.setattr(auth, "_registration_allowed", lambda: record(True))
    monkeypatch.setattr(auth, "_hash_password", lambda password: record("hash"))
    monkeypatch.setattr(auth, "create_access_token", lambda *args: "token")

    result = await auth.register(
        RegisterRequest(username="alice", password="long-secret"), Response()
    )

    assert result["success"] is True
    assert len(worker_threads) == 4
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)
@pytest.mark.asyncio
async def test_register_maps_database_uniqueness_race_to_conflict(monkeypatch):
    from api import auth
    from fastapi import HTTPException

    class IntegrityError(Exception):
        pass

    fake_db = SimpleNamespace(
        get_user_by_username=lambda username: None,
        add_user=lambda username, password_hash, bootstrap_only=False: (_ for _ in ()).throw(
            IntegrityError("duplicate username")
        ),
    )
    monkeypatch.setattr(auth, "db", fake_db)
    monkeypatch.setattr(auth, "_registration_allowed", lambda: True)
    monkeypatch.setattr(auth, "_hash_password", lambda password: "hash")

    with pytest.raises(HTTPException) as exc_info:
        await auth.register(
            RegisterRequest(username="alice", password="long-secret"), Response()
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "用户名已存在"

def test_bootstrap_only_registration_is_enforced_inside_database_transaction(tmp_path):
    from db.database import SQLiteDB
    from db.errors import RegistrationClosedError

    database = SQLiteDB(tmp_path / "bootstrap.db")
    try:
        first = database.add_user("admin", "hash", bootstrap_only=True)
        assert first["role"] == "admin"

        with pytest.raises(RegistrationClosedError):
            database.add_user("racing-user", "hash", bootstrap_only=True)

        users = database.execute_sql("SELECT username, role FROM users ORDER BY id")
        assert users == [{"username": "admin", "role": "admin"}]
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_register_maps_bootstrap_race_to_forbidden(monkeypatch):
    from api import auth
    from db.errors import RegistrationClosedError
    from fastapi import HTTPException

    fake_db = SimpleNamespace(
        get_user_by_username=lambda username: None,
        add_user=lambda username, password_hash, bootstrap_only=False: (
            _ for _ in ()
        ).throw(RegistrationClosedError("already bootstrapped")),
    )
    monkeypatch.setattr(auth, "db", fake_db)
    monkeypatch.setattr(auth, "_bootstrap_registration_only", lambda: True)
    monkeypatch.setattr(auth, "_registration_allowed", lambda: True)
    monkeypatch.setattr(auth, "_hash_password", lambda password: "hash")

    with pytest.raises(HTTPException) as exc_info:
        await auth.register(
            RegisterRequest(username="alice", password="long-secret"), Response()
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "生产环境已关闭公开注册"
