"""Saved-dialogue APIs use one request contract and their application's store."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.config import create_access_token
from app.main import create_app, global_exception_handler
from app.runtime import RuntimeContainer
from db.database import SQLiteDB


@pytest.fixture
def stores(tmp_path, monkeypatch):
    from db.adapter import db as global_db
    from training import preprocessor

    databases = [SQLiteDB(tmp_path / f"saved-{index}.sqlite") for index in range(2)]
    for database in databases:
        database.add_user("saved-owner", "unused-signed-token-fixture")
    forbidden = Mock(side_effect=AssertionError("synthetic-private-global-store-must-not-be-used"))
    monkeypatch.setattr(global_db, "execute_sql", forbidden)
    monkeypatch.setattr(global_db, "execute_sql_insert", forbidden)
    processor = preprocessor.DatasetPreprocessor(base_dir=tmp_path)
    monkeypatch.setattr(preprocessor, "get_dataset_preprocessor", lambda: processor)
    yield SimpleNamespace(databases=databases, forbidden=forbidden, processor=processor)
    for database in databases:
        database.close_connection()


def client(database):
    app = create_app(
        RuntimeContainer(
            db=database,
            is_pg_mode=lambda: False,
            startup_env={"ENVIRONMENT": "test", "SECURITY_MIDDLEWARE_ENABLED": "false"},
        )
    )
    errors = []

    async def observe(request, error):
        errors.append(error)
        return await global_exception_handler(request, error)

    app.add_exception_handler(Exception, observe)
    token = create_access_token("saved-owner", 1, "admin")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="https://isolated",
        headers={"Authorization": "Bearer " + token},
    ), errors


def payload(name="合成保存记录"):
    return {
        "name": name,
        "character_desc": "周三下班后饮茶，工作日早晨不饮；出差只喝白水。",
        "style": "日常",
        "turn_stats": {"2": 2},
        "scene_stats": {"work": 1, "travel": 1},
        "dialogues": [
            {
                "scene": "work",
                "system": "合成角色",
                "conversations": [
                    {"from": "human", "value": "周三什么时候饮茶？"},
                    {"from": "gpt", "value": "下班后，工作日早晨不饮茶。"},
                ],
            },
            {
                "scene": "travel",
                "system": "合成角色",
                "conversations": [
                    {"from": "human", "value": "出差时也喝茶吗？"},
                    {"from": "gpt", "value": "出差时只喝白水。"},
                ],
            },
        ],
    }


def seed(database):
    value = payload()
    return database.execute_sql_insert(
        "INSERT INTO saved_dialogues (name,character_desc,dialogue_count,dialogues_json,created_at,updated_at) "
        "VALUES (:name,:description,2,:dialogues,'2026-10-26','2026-10-26')",
        {"name": value["name"], "description": value["character_desc"], "dialogues": json.dumps(value["dialogues"])},
    )["lastrowid"]


async def test_two_applications_save_read_export_and_delete_only_their_own_dialogues(stores):
    for index, database in enumerate(stores.databases):
        api, errors = client(database)
        value = payload(f"隔离记录-{index}")
        async with api:
            saved = await api.post("/api/training/saved-dialogues", json=value)
            assert saved.status_code == 200 and saved.json()["success"] is True
            item = f"/api/training/saved-dialogues/{saved.json()['id']}"
            listed = await api.get("/api/training/saved-dialogues")
            assert listed.status_code == 200
            assert [row["name"] for row in listed.json()["items"]] == [value["name"]]
            read = await api.get(item)
            assert read.status_code == 200
            assert all(
                read.json()[key] == value[key]
                for key in ["name", "character_desc", "style", "dialogues", "turn_stats", "scene_stats"]
            )
            exported = await api.post(item + "/create-dataset", params={"dataset_name": f"saved-{index}"})
            assert exported.status_code == 200 and exported.json()["dataset"]["count"] == 2
            assert Path(exported.json()["dataset"]["path"]).is_relative_to(stores.processor.data_dir)
            assert Path(exported.json()["dataset"]["path"]).is_dir()
            removed = await api.delete(item + "/dialogues/0")
            assert removed.status_code == 200 and removed.json()["remaining_count"] == 1
            remaining = (await api.get(item)).json()
            assert remaining["dialogue_count"] == 1 and remaining["dialogues"] == value["dialogues"][1:]
            assert remaining["turn_stats"] == {"2": 1}
            assert remaining["scene_stats"] == {"travel": 1}
            last = await api.delete(item + "/dialogues/0")
            assert last.status_code == 200 and last.json()["remaining_count"] == 0
            empty = (await api.get(item)).json()
            assert empty["dialogue_count"] == 0 and empty["dialogues"] == []
            assert empty["turn_stats"] == empty["scene_stats"] == {}
            assert (await api.post(item + "/create-dataset")).status_code == 400
            assert (await api.delete(item)).status_code == 200
            assert (await api.get(item)).status_code == 404
        assert errors == []
        assert database.execute_sql("SELECT count(*) AS n FROM saved_dialogues")[0]["n"] == 0
    stores.forbidden.assert_not_called()


@pytest.mark.parametrize("operation", ["list", "save", "get", "delete", "remove", "dataset"])
async def test_selected_store_failure_reaches_shared_handler_without_success(stores, monkeypatch, operation):
    database = stores.databases[0]
    item = f"/api/training/saved-dialogues/{seed(database)}"
    failure = OSError("synthetic-private-selected-store-failure")
    operation_mock = Mock(side_effect=failure)
    monkeypatch.setattr(database, "execute_sql_insert" if operation == "save" else "execute_sql", operation_mock)
    api, errors = client(database)
    requests = {
        "list": ("GET", "/api/training/saved-dialogues"),
        "save": ("POST", "/api/training/saved-dialogues"),
        "get": ("GET", item),
        "delete": ("DELETE", item),
        "remove": ("DELETE", item + "/dialogues/0"),
        "dataset": ("POST", item + "/create-dataset"),
    }
    async with api:
        method, path = requests[operation]
        response = await api.request(method, path, **({"json": payload()} if operation == "save" else {}))
    assert response.status_code == 500 and response.json()["success"] is False
    assert response.json()["error"] == "Internal Server Error" and "synthetic-private" not in response.text
    assert errors == [failure]
    operation_mock.assert_called_once()
    assert database.get_connection().execute("SELECT count(*) FROM saved_dialogues").fetchone()[0] == 1
    stores.forbidden.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("name", "", id="empty-name"),
        pytest.param("name", "n" * 201, id="long-name"),
        pytest.param("character_desc", "", id="empty-description"),
        pytest.param("turn_stats", {"2": "not-a-count"}, id="invalid-count"),
    ],
)
async def test_invalid_save_input_is_rejected_before_storage(stores, field, value):
    database = stores.databases[0]
    request = payload()
    request[field] = value
    api, errors = client(database)
    async with api:
        response = await api.post("/api/training/saved-dialogues", json=request)
    assert response.status_code == 422 and errors == []
    assert database.execute_sql("SELECT count(*) AS n FROM saved_dialogues")[0]["n"] == 0
    stores.forbidden.assert_not_called()


async def test_all_saved_dialogue_operations_require_current_admin_role(stores):
    database = stores.databases[0]
    item = f"/api/training/saved-dialogues/{seed(database)}"
    api, errors = client(database)
    calls = [
        ("GET", "/api/training/saved-dialogues"),
        ("POST", "/api/training/saved-dialogues"),
        ("GET", item),
        ("DELETE", item),
        ("DELETE", item + "/dialogues/0"),
        ("POST", item + "/create-dataset"),
    ]
    async with api:
        authorization = api.headers.pop("authorization")
        for expected in [401, 403]:
            if expected == 403:
                api.headers["authorization"] = authorization
                database.execute_sql("UPDATE users SET role='user' WHERE username='saved-owner'")
            for method, path in calls:
                response = await api.request(
                    method,
                    path,
                    **({"json": payload()} if path == "/api/training/saved-dialogues" and method == "POST" else {}),
                )
                assert response.status_code == expected
    assert errors == []
    assert database.execute_sql("SELECT count(*) AS n FROM saved_dialogues")[0]["n"] == 1
    stores.forbidden.assert_not_called()


async def test_normal_missing_records_and_invalid_indexes_preserve_business_errors(stores):
    database = stores.databases[0]
    item = f"/api/training/saved-dialogues/{seed(database)}"
    api, errors = client(database)
    async with api:
        for method, suffix in [("GET", ""), ("DELETE", ""), ("DELETE", "/dialogues/0"), ("POST", "/create-dataset")]:
            assert (await api.request(method, "/api/training/saved-dialogues/999999" + suffix)).status_code == 404
        assert (await api.delete(item + "/dialogues/99")).status_code == 400
    assert errors == []
    stores.forbidden.assert_not_called()


async def test_corrupt_stored_json_is_an_error_instead_of_empty_success(stores):
    database = stores.databases[0]
    identifier = seed(database)
    database.execute_sql("UPDATE saved_dialogues SET dialogues_json='{' WHERE id=:id", {"id": identifier})
    api, errors = client(database)
    async with api:
        response = await api.get(f"/api/training/saved-dialogues/{identifier}")
    assert response.status_code == 500 and response.json()["success"] is False
    assert len(errors) == 1 and isinstance(errors[0], json.JSONDecodeError)
    assert database.execute_sql("SELECT dialogues_json FROM saved_dialogues")[0]["dialogues_json"] == "{"
    stores.forbidden.assert_not_called()
