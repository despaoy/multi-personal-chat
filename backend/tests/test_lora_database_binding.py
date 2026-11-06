"""LoRA HTTP operations use their application's store and explicit status."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import loras
from app.config import create_access_token
from app.runtime import RuntimeContainer
from db import database as database_module
from db.database import SQLiteDB


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    root = tmp_path / "adapters"
    root.mkdir()
    monkeypatch.setattr(loras, "LORA_ROOT", root)
    monkeypatch.setattr(database_module, "LORA_ROOT", root)
    monkeypatch.setenv("BASE_MODEL_PATH", "synthetic-base")
    monkeypatch.setenv("VLLM_LORA_ROOT", str(root))
    monkeypatch.setenv("LORA_ALLOWED_REAL_ROOTS", "")
    stores = [SQLiteDB(tmp_path / f"app-{index}.sqlite") for index in range(2)]
    for index, store in enumerate(stores):
        store.add_user("synthetic-admin", "unused-hash")
        store.execute_sql("UPDATE users SET role='admin' WHERE username='synthetic-admin'")
        for identifier, name, status in [("1", "old-adapter", "active"), ("2", "candidate-adapter", "inactive")]:
            store.add_lora({"id": identifier, "name": name, "status": status, "description": f"owner-{index}"})
    for name in ["old-adapter", "candidate-adapter", "new-adapter"]:
        folder = root / name
        folder.mkdir()
        (folder / "adapter_config.json").write_text(
            json.dumps(
                {"base_model_name_or_path": "synthetic-base", "target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}
            )
        )
        (folder / "adapter_model.safetensors").write_bytes(b"explicit synthetic transport fixture; not model weights")
    forbidden = Mock(side_effect=AssertionError("LoRA route accessed a process-global database"))
    monkeypatch.setattr(
        loras,
        "db",
        SimpleNamespace(
            get_loras=forbidden,
            add_lora=forbidden,
            execute_sql=forbidden,
            update_lora_status=forbidden,
            delete_lora=forbidden,
        ),
        raising=False,
    )
    state = SimpleNamespace(stores=stores, root=root, events=[], forbidden=forbidden, fail_load=False)

    async def load(name, path):
        state.events.append(("load", name))
        if state.fail_load:
            raise OSError("synthetic-private-runtime-failure")

    async def unload(name):
        state.events.append(("unload", name))

    getter = AsyncMock(return_value=SimpleNamespace(load_lora_adapter=load, unload_lora_adapter=unload))
    monkeypatch.setattr(loras, "get_vllm_client", getter, raising=False)
    clients = []
    for store in stores:
        app = FastAPI()
        app.state.runtime_container = RuntimeContainer(db=store, is_pg_mode=lambda: False, startup_env={})
        app.include_router(loras.router)
        clients.append(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="https://isolated-lora.test",
                headers={"Authorization": "Bearer " + create_access_token("synthetic-admin", 1, "admin")},
            )
        )
    state.clients = clients
    try:
        yield state
    finally:
        for client in clients:
            await client.aclose()
        for store in stores:
            store.close_connection()
        database_module.refresh_lora_dir_map(database_module.LORA_ROOT)


async def test_list_scan_activate_deactivate_delete_remain_bound_to_one_application(rig):
    first, second = rig.clients
    unchanged = rig.stores[1].get_loras()
    listed = await first.get("/api/loras")
    other = await second.get("/api/loras")
    assert listed.status_code == other.status_code == 200
    assert {row["description"] for row in listed.json()["loras"]} == {"owner-0"}
    assert {row["description"] for row in other.json()["loras"]} == {"owner-1"}
    scanned = await first.post("/api/loras/scan")
    assert scanned.status_code == 200 and scanned.json()["success"] is True and scanned.json()["new_count"] == 1
    assert rig.stores[1].get_loras() == unchanged
    assert (await first.put("/api/loras/2/status", json={"status": "active"})).status_code == 200
    assert rig.stores[0].get_loras("active")[0]["id"] == "2"
    assert rig.stores[1].get_loras() == unchanged
    assert (await first.put("/api/loras/2/status", json={"status": "inactive"})).status_code == 200
    deleted = await first.delete("/api/loras/2")
    assert deleted.status_code == 200 and deleted.json()["success"] is True
    assert rig.events == [("unload", "old-adapter"), ("load", "candidate-adapter"), ("unload", "candidate-adapter")]
    assert not any(row["id"] == "2" for row in rig.stores[0].get_loras())
    assert rig.stores[1].get_loras() == unchanged
    assert (rig.root / "candidate-adapter/adapter_model.safetensors").is_file()
    rig.forbidden.assert_not_called()


async def test_missing_or_invalid_status_never_becomes_deactivation(rig):
    before = rig.stores[0].get_loras()
    for body in [{}, {"status": "unknown"}]:
        response = await rig.clients[0].put("/api/loras/1/status", json=body)
        assert response.status_code == 422
    assert not rig.events and rig.stores[0].get_loras() == before
    rig.forbidden.assert_not_called()


@pytest.mark.parametrize("operation", ["list", "scan", "status", "delete"])
async def test_store_read_failure_cannot_use_another_database(rig, monkeypatch, operation):
    monkeypatch.setattr(rig.stores[0], "get_loras", Mock(side_effect=OSError("synthetic-local-store-down")))
    api = rig.clients[0]
    if operation == "list":
        response = await api.get("/api/loras")
    elif operation == "scan":
        response = await api.post("/api/loras/scan")
    elif operation == "status":
        response = await api.put("/api/loras/2/status", json={"status": "active"})
    else:
        response = await api.delete("/api/loras/2")
    assert response.status_code == 500 and not rig.events
    rig.forbidden.assert_not_called()


async def test_lora_management_still_requires_current_admin(rig):
    api = rig.clients[0]
    token = api.headers.pop("Authorization")
    assert (await api.get("/api/loras")).status_code == 401
    api.headers["Authorization"] = token
    rig.stores[0].execute_sql("UPDATE users SET role='user' WHERE username='synthetic-admin'")
    assert (await api.post("/api/loras/scan")).status_code == 403
    assert (await api.put("/api/loras/2/status", json={"status": "active"})).status_code == 403
    assert (await api.delete("/api/loras/2")).status_code == 403
    assert not rig.events
    rig.forbidden.assert_not_called()


async def test_both_compensation_errors_are_reported_without_private_content(caplog):
    first, second = OSError("synthetic-private-unload"), RuntimeError("synthetic-private-load")
    client = SimpleNamespace(
        unload_lora_adapter=AsyncMock(side_effect=first), load_lora_adapter=AsyncMock(side_effect=second)
    )
    with pytest.raises(HTTPException) as caught:
        await loras._rollback_runtime_lora_state(
            client, loaded_adapter=("new", "/synthetic/new"), unloaded_adapter=("old", "/synthetic/old")
        )
    assert caught.value.status_code == 502 and caught.value.detail["kind"] == "lora_recovery_failed"
    assert caught.value.detail["runtimeState"] == "unknown"
    assert caught.value.__cause__.exceptions == (first, second)
    client.unload_lora_adapter.assert_awaited_once_with("new")
    client.load_lora_adapter.assert_awaited_once_with("old", "/synthetic/old")
    assert "synthetic-private" not in caplog.text and "synthetic-private" not in json.dumps(caught.value.detail)


async def test_failed_status_write_restores_runtime_without_changing_other_store(rig, monkeypatch):
    before = [store.get_loras() for store in rig.stores]
    monkeypatch.setattr(rig.stores[0], "update_lora_status", Mock(side_effect=OSError("synthetic-write-failure")))
    response = await rig.clients[0].put("/api/loras/2/status", json={"status": "active"})
    assert response.status_code == 500
    assert rig.events == [
        ("unload", "old-adapter"),
        ("load", "candidate-adapter"),
        ("unload", "candidate-adapter"),
        ("load", "old-adapter"),
    ]
    assert [store.get_loras() for store in rig.stores] == before
    rig.forbidden.assert_not_called()


async def test_failed_delete_and_recovery_reports_unknown_runtime_state(rig, monkeypatch):
    before = rig.stores[0].get_loras()
    monkeypatch.setattr(rig.stores[0], "delete_lora", Mock(side_effect=OSError("synthetic-delete-failure")))
    rig.fail_load = True
    response = await rig.clients[0].delete("/api/loras/1")
    assert response.status_code == 502 and response.json()["detail"]["kind"] == "lora_recovery_failed"
    assert rig.events == [("unload", "old-adapter"), ("load", "old-adapter")]
    assert rig.stores[0].get_loras() == before
    assert (rig.root / "old-adapter/adapter_model.safetensors").is_file()
    rig.forbidden.assert_not_called()


async def test_switch_unload_fault_restores_previous_adapter_before_reporting_failure(rig, monkeypatch):
    before = rig.stores[0].get_loras()

    async def unload(name):
        rig.events.append(("unload", name))
        raise OSError("synthetic-switch-unload-failure")

    async def load(name, path):
        rig.events.append(("load", name))

    monkeypatch.setattr(
        loras,
        "get_vllm_client",
        AsyncMock(return_value=SimpleNamespace(unload_lora_adapter=unload, load_lora_adapter=load)),
    )
    update = Mock(side_effect=AssertionError("failed unload must not commit a new active adapter"))
    monkeypatch.setattr(rig.stores[0], "update_lora_status", update)
    response = await rig.clients[0].put("/api/loras/2/status", json={"status": "active"})
    assert response.status_code == 502
    assert rig.events == [("unload", "old-adapter"), ("load", "old-adapter")]
    assert rig.stores[0].get_loras() == before
    update.assert_not_called()
    rig.forbidden.assert_not_called()


@pytest.mark.parametrize("operation", ["activate", "deactivate", "delete"])
async def test_complete_client_failure_keeps_registration_and_restores_runtime(rig, monkeypatch, operation):
    before = rig.stores[0].get_loras()
    failure = RuntimeError("synthetic-primary-lora-fault")

    async def load(name, path):
        rig.events.append(("load", name))
        if operation == "activate" and name == "candidate-adapter":
            raise failure

    async def unload(name):
        rig.events.append(("unload", name))
        if operation != "activate":
            raise failure

    monkeypatch.setattr(
        loras,
        "get_vllm_client",
        AsyncMock(return_value=SimpleNamespace(load_lora_adapter=load, unload_lora_adapter=unload)),
    )
    update = Mock(wraps=rig.stores[0].update_lora_status)
    delete = Mock(wraps=rig.stores[0].delete_lora)
    monkeypatch.setattr(rig.stores[0], "update_lora_status", update)
    monkeypatch.setattr(rig.stores[0], "delete_lora", delete)
    if operation == "delete":
        response = await rig.clients[0].delete("/api/loras/1")
    else:
        identifier, status = ("2", "active") if operation == "activate" else ("1", "inactive")
        response = await rig.clients[0].put(f"/api/loras/{identifier}/status", json={"status": status})
    assert response.status_code == 502
    assert isinstance(response.json()["detail"], str), "complete compensation must not become an unknown-runtime error"
    assert "synthetic-primary-lora-fault" not in response.text
    assert rig.stores[0].get_loras() == before
    update.assert_not_called()
    delete.assert_not_called()
    expected = (
        [
            ("unload", "old-adapter"),
            ("load", "candidate-adapter"),
            ("unload", "candidate-adapter"),
            ("load", "old-adapter"),
        ]
        if operation == "activate"
        else [("unload", "old-adapter"), ("load", "old-adapter")]
    )
    assert rig.events == expected
    assert (rig.root / "old-adapter/adapter_model.safetensors").is_file()
    rig.forbidden.assert_not_called()
