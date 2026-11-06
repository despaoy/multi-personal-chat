"""Tool CRUD and native HTTP errors use the real application boundary."""

import asyncio
import io
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import sqlalchemy
import uvicorn

from api import claw
from app.config import create_access_token
from app.main import create_app
from app.runtime import RuntimeContainer
from db.database import SQLiteDB

CODES = [
    '\ndef main(args):\n    return "下雨取消，尚未付款。"\n  ',
    'def main(args):\n    return "周五确认，预约不能代替原件。"\n',
]


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    state = SimpleNamespace(stores=[], apps=[], clients=[], root=tmp_path)
    state.forbidden = Mock(side_effect=AssertionError("tool CRUD used a global database"))
    monkeypatch.setattr(
        claw,
        "db",
        SimpleNamespace(
            get_claw_tools=state.forbidden, save_claw_tool=state.forbidden, delete_claw_tool=state.forbidden
        ),
        raising=False,
    )
    state.execute = Mock(side_effect=AssertionError("tool CRUD must not execute code"))
    monkeypatch.setattr(claw, "run_tool_code", state.execute)
    for index in range(2):
        database = SQLiteDB(tmp_path / f"tool-owner-{index}.sqlite")
        database.add_user("synthetic-admin", "unused-signed-token-hash")
        database.execute_sql("UPDATE users SET role='admin' WHERE username='synthetic-admin'")
        database.save_claw_tool("shared-name", f"owner-{index}", CODES[index], bool(index))
        app = create_app(
            RuntimeContainer(
                db=database,
                is_pg_mode=lambda: False,
                startup_env={"ENVIRONMENT": "test"},
                model_check=lambda: {"loaded": True},
            )
        )
        state.stores.append(database)
        state.apps.append(app)
        state.clients.append(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://127.0.0.1",
                headers={"Authorization": "Bearer " + create_access_token("synthetic-admin", 1, "admin")},
            )
        )
    try:
        yield state
    finally:
        for client in state.clients:
            await client.aclose()
        for database in state.stores:
            database.close_connection()


async def test_tool_crud_preserves_code_and_uses_only_its_application_database(rig):
    first, second = rig.clients
    untouched = rig.stores[1].get_claw_tools()
    first_list, second_list = await first.get("/api/claw/tools"), await second.get("/api/claw/tools")
    assert first_list.status_code == second_list.status_code == 200
    for index, response in enumerate([first_list, second_list]):
        custom = [item for item in response.json()["tools"] if not item["builtin"]]
        assert len(custom) == 1 and custom[0]["code"] == CODES[index]
        assert custom[0]["enabled"] is bool(index)
        assert response.json()["total"] == len(claw.BUILTIN_TOOLS) + 1
    saved = await first.post(
        "/api/claw/tools", json={"name": "  extra  ", "description": " 完整说明 ", "code": CODES[1], "enabled": False}
    )
    assert saved.status_code == 200 and saved.json()["success"] is True
    row = rig.stores[0].get_claw_tool_by_name("extra")
    assert row["code"] == CODES[1] and row["description"] == "完整说明" and not row["enabled"]
    updated = await first.post(
        "/api/claw/tools", json={"name": "shared-name", "description": "updated", "code": CODES[1], "enabled": True}
    )
    assert updated.status_code == 200 and rig.stores[0].get_claw_tool_by_name("shared-name")["code"] == CODES[1]
    deleted = await first.delete("/api/claw/tools/shared-name")
    assert deleted.status_code == 200 and deleted.json()["success"] is True
    assert (await first.delete("/api/claw/tools/shared-name")).status_code == 404
    assert rig.stores[1].get_claw_tools() == untouched
    rig.forbidden.assert_not_called()
    rig.execute.assert_not_called()


async def test_invalid_names_and_builtin_conflicts_stop_before_storage(rig):
    api = rig.clients[0]
    for body, status in [({}, 422), ({"name": ""}, 422), ({"name": "   "}, 400), ({"name": " status "}, 400)]:
        assert (await api.post("/api/claw/tools", json=body)).status_code == status
    assert (await api.delete("/api/claw/tools/status")).status_code == 400
    rig.forbidden.assert_not_called()
    rig.execute.assert_not_called()


async def test_tool_management_and_execution_require_current_admin(rig):
    api = rig.clients[0]
    token = api.headers.pop("Authorization")
    assert (await api.get("/api/claw/tools")).status_code == 401
    api.headers["Authorization"] = token
    rig.stores[0].execute_sql("UPDATE users SET role='user' WHERE username='synthetic-admin'")
    assert (await api.get("/api/claw/tools")).status_code == 403
    assert (await api.post("/api/claw/tools", json={"name": "new"})).status_code == 403
    assert (await api.delete("/api/claw/tools/shared-name")).status_code == 403
    assert (await api.post("/api/claw/tools/execute", json={"code": "print(4)"})).status_code == 403
    rig.forbidden.assert_not_called()
    rig.execute.assert_not_called()


@pytest.mark.parametrize("operation", ["read", "write", "delete"])
async def test_storage_failure_is_explicit_and_repaired_store_succeeds(rig, monkeypatch, caplog, operation):
    database, api = rig.stores[0], rig.clients[0]
    monkeypatch.setattr(claw, "db", database, raising=False)
    method = {"read": "get_claw_tools", "write": "save_claw_tool", "delete": "delete_claw_tool"}[operation]
    original = getattr(database, method)
    monkeypatch.setattr(database, method, Mock(side_effect=OSError("synthetic-private-storage-failure")))

    async def send():
        if operation == "read":
            return await api.get("/api/claw/tools")
        if operation == "write":
            return await api.post("/api/claw/tools", json={"name": "shared-name", "code": CODES[1], "enabled": True})
        return await api.delete("/api/claw/tools/shared-name")

    response = await send()
    assert response.status_code == 500 and response.json()["success"] is False
    assert "synthetic-private" not in response.text + caplog.text
    monkeypatch.setattr(database, method, original)
    healthy = await send()
    assert healthy.status_code == 200 and healthy.json()["success"] is True
    assert rig.stores[1].get_claw_tool_by_name("shared-name")["code"] == CODES[1]


async def test_direct_tool_save_retains_the_original_storage_exception(rig, monkeypatch):
    failure = OSError("synthetic-direct-save-failure")
    monkeypatch.setattr(rig.stores[0], "save_claw_tool", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        await claw.save_tool(
            claw.ToolSaveRequest(name="complete-name", code=CODES[0]),
            current_user={"role": "admin"},
            database=rig.stores[0],
        )
    assert caught.value is failure


@pytest.mark.parametrize("driver", ["sqlite", "sqlalchemy", "os"])
async def test_native_driver_fault_is_private_and_recovers_without_degraded_result(rig, monkeypatch, driver, caplog):
    database, app = rig.stores[0], rig.apps[0]
    monkeypatch.setattr(claw, "db", database, raising=False)
    original = database.get_claw_tools
    marker = "synthetic_private_storage_column_8457"

    def fault():
        if driver == "sqlite":
            return database.execute_sql("SELECT " + marker)
        if driver == "sqlalchemy":
            engine = sqlalchemy.create_engine("sqlite://")
            try:
                with engine.connect() as connection:
                    connection.exec_driver_sql("SELECT " + marker)
            finally:
                engine.dispose()
        return (rig.root / marker).read_text()

    captured = io.StringIO()
    log = logging.getLogger("uvicorn.error")
    previous = (log.handlers[:], log.propagate, log.level)
    log.handlers, log.propagate = [logging.StreamHandler(captured)], False
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=0, access_log=False, log_config=None, log_level="warning", lifespan="off"
        )
    )
    serving = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if serving.done():
                    await serving
                await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=4, headers=rig.clients[0].headers
        ) as api:
            assert (await api.get("/api/claw/tools")).status_code == 200
            monkeypatch.setattr(database, "get_claw_tools", fault)
            response = await api.get("/api/claw/tools")
            await asyncio.sleep(0)
            assert response.status_code == 500 and response.json()["success"] is False
            assert marker not in response.text and marker not in captured.getvalue()
            assert marker not in caplog.text
            assert response.json()["error"] == ("database_error" if driver != "os" else "Internal Server Error")
            monkeypatch.setattr(database, "get_claw_tools", original)
            healthy = await api.get("/api/claw/tools")
            assert healthy.status_code == 200 and healthy.json()["success"] is True
            assert any(item["code"] == CODES[0] for item in healthy.json()["tools"] if not item["builtin"])
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 5)
        log.handlers, log.propagate, log.level = previous
