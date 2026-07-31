from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.user_data import router
from app.dependencies import get_current_user
from app.runtime import RuntimeContainer
from repositories.user_data import DatabaseUserDataRepository, UserDataUserNotFoundError


class FakeUserDataDatabase:
    def __init__(self) -> None:
        self.user: dict[str, object] | None = {"id": 42, "username": "alice"}
        self.lookup_error: Exception | None = None
        self.calls: list[tuple[str, object]] = []
        self.page_data = {
            "page_key": "profile",
            "data_json": '{"nickname":"Alice"}',
            "updated_at": "2026-07-31T10:00:00",
        }

    def get_user_by_username(self, username: str):
        self.calls.append(("user", username))
        if self.lookup_error is not None:
            raise self.lookup_error
        return self.user

    def get_user_data(self, user_id: int, page_key: str | None = None):
        self.calls.append(("load", (user_id, page_key)))
        if page_key is not None:
            return self.page_data if page_key == "profile" else None
        return {"profile": {"data_json": self.page_data["data_json"], "updated_at": self.page_data["updated_at"]}}

    def save_user_data(self, user_id: int, page_key: str, data_json: str) -> bool:
        self.calls.append(("save", (user_id, page_key, data_json)))
        return True


async def test_database_user_data_repository_maps_load_and_save() -> None:
    database = FakeUserDataDatabase()
    repository = DatabaseUserDataRepository(database)

    page = await repository.load("alice", "profile")
    all_pages = await repository.load("alice")
    await repository.save("alice", "draft", '{"value":2}')

    assert page == database.page_data
    assert all_pages == {
        "profile": {
            "data_json": '{"nickname":"Alice"}',
            "updated_at": "2026-07-31T10:00:00",
        }
    }
    assert database.calls == [
        ("user", "alice"),
        ("load", (42, "profile")),
        ("user", "alice"),
        ("load", (42, None)),
        ("user", "alice"),
        ("save", (42, "draft", '{"value":2}')),
    ]


async def test_database_user_data_repository_stops_when_user_is_missing() -> None:
    database = FakeUserDataDatabase()
    database.user = None
    repository = DatabaseUserDataRepository(database)

    with pytest.raises(UserDataUserNotFoundError):
        await repository.load("missing", "profile")
    with pytest.raises(UserDataUserNotFoundError):
        await repository.save("missing", "profile", "{}")

    assert database.calls == [("user", "missing"), ("user", "missing")]


def _build_test_client(database: FakeUserDataDatabase) -> TestClient:
    application = FastAPI()
    application.state.runtime_container = RuntimeContainer(
        db=database,
        is_pg_mode=lambda: False,
        startup_env={},
    )
    application.include_router(router)
    application.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "username": "alice"}
    return TestClient(application)


def test_user_data_routes_preserve_success_and_not_found_responses() -> None:
    database = FakeUserDataDatabase()

    with _build_test_client(database) as client:
        keyed = client.get("/api/user/data", params={"page_key": "profile"})
        missing_page = client.get("/api/user/data", params={"page_key": "missing"})
        all_pages = client.get("/api/user/data")
        saved = client.put(
            "/api/user/data",
            json={"page_key": "draft", "data_json": '{"value":2}'},
        )

        database.user = None
        missing_user_get = client.get("/api/user/data", params={"page_key": "profile"})
        missing_user_save = client.put(
            "/api/user/data",
            json={"page_key": "draft", "data_json": "{}"},
        )

    assert keyed.status_code == 200
    assert keyed.json() == {"success": True, "data": database.page_data}
    assert missing_page.json() == {"success": True, "data": None}
    assert all_pages.json() == {
        "success": True,
        "data": {
            "profile": {
                "data_json": '{"nickname":"Alice"}',
                "updated_at": "2026-07-31T10:00:00",
            }
        },
    }
    assert saved.json() == {"success": True, "message": "数据保存成功"}
    assert missing_user_get.status_code == 404
    assert missing_user_get.json() == {"detail": "用户不存在"}
    assert missing_user_save.status_code == 404
    assert missing_user_save.json() == {"detail": "用户不存在"}


def test_user_data_routes_do_not_expose_internal_errors() -> None:
    database = FakeUserDataDatabase()
    database.lookup_error = RuntimeError("postgres://private-host/secret-database")

    with _build_test_client(database) as client:
        load_response = client.get("/api/user/data", params={"page_key": "profile"})
        save_response = client.put(
            "/api/user/data",
            json={"page_key": "draft", "data_json": "{}"},
        )

    assert load_response.status_code == 500
    assert load_response.json() == {"detail": "获取数据失败"}
    assert save_response.status_code == 500
    assert save_response.json() == {"detail": "保存失败"}
    assert "private-host" not in load_response.text
    assert "private-host" not in save_response.text
