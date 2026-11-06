"""Character management keeps source data and delegates known failures to the app."""

import asyncio
import io
import logging
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from repositories.character_memory import DatabaseCharacterMemoryRepository

from api import characters
from app.config import create_access_token
from app.main import create_app, global_exception_handler
from app.runtime import RuntimeContainer
from character.models import MemoryItem, RelationshipState, build_user_scope
from character.profile_registry import CharacterProfileRegistry
from db.database import SQLiteDB

BODY = "我可能喜欢蓝莓茶，还没有确定；只有夜间休息时才考虑，出差时不适用。"


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path / "character-errors.sqlite")
    database.add_user("synthetic-admin", "unused-signed-token-hash")
    database.execute_sql("UPDATE users SET role='admin' WHERE username='synthetic-admin'")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = build_user_scope("web", "isolated-character-api", "synthetic-owner", "synthetic-room", "private")
    await repository.capture_source(
        "role",
        scope,
        source_message_id="complete-source",
        body=BODY,
        observed_at=datetime(2026, 11, 5, 1, tzinfo=timezone.utc),
    )
    saved = await repository.append_claim(
        "role",
        scope,
        MemoryItem("", "user_fact", BODY, 0.7),
        memory_key="tea_preference",
        confidence=0.67,
        evidence=(BODY,),
        source_message_id="complete-source",
        source_message_ids=("complete-source",),
        metadata={"qualifiers": {"condition": "只有夜间休息时才考虑", "exception": "出差时不适用"}},
    )
    await repository.upsert_relationship("role", scope, RelationshipState(stage="familiar", summary="合成关系记录"))
    app = create_app(
        RuntimeContainer(
            db=database,
            is_pg_mode=lambda: False,
            startup_env={"ENVIRONMENT": "test"},
            model_check=lambda: {"loaded": True},
        )
    )
    state = SimpleNamespace(
        database=database,
        repo=repository,
        saved=saved,
        app=app,
        root=tmp_path,
        handled=[],
        faults=[],
        params={
            name: getattr(scope, name)
            for name in ("platform", "adapter", "sender_id", "conversation_type", "conversation_id")
        },
    )

    async def observe(request, exc):
        state.handled.append(exc)
        return await global_exception_handler(request, exc)

    app.add_exception_handler(sqlite3.Error, observe)
    app.add_exception_handler(OSError, observe)
    registry = CharacterProfileRegistry()
    registry.load_profiles()
    state.registry = registry
    monkeypatch.setattr(characters, "get_default_profile_registry", lambda: registry)
    log = logging.getLogger("uvicorn.error")
    previous = (log.handlers[:], log.propagate, log.level)
    state.server_logs = io.StringIO()
    log.handlers, log.propagate = [logging.StreamHandler(state.server_logs)], False
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
            base_url=f"http://127.0.0.1:{port}",
            trust_env=False,
            timeout=4,
            headers={"Authorization": "Bearer " + create_access_token("synthetic-admin", 1, "admin")},
        ) as api:
            state.api = api
            yield state
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 5)
        log.handlers, log.propagate, log.level = previous
        database.close_connection()


@pytest.mark.parametrize(
    "operation",
    ["profiles", "relationship-read", "relationship-write", "memory-list", "memory-write", "erase", "clear"],
)
async def test_known_fault_uses_shared_protocol_retains_cause_and_recovers(rig, monkeypatch, caplog, operation):
    saved = rig.saved
    memory_url = f"/api/characters/role/memories/{saved['id']}"
    operations = {
        "profiles": (rig.registry, "list_profiles", "GET", "/api/characters", None),
        "relationship-read": (
            rig.database,
            "get_character_relationship",
            "GET",
            "/api/characters/role/relationship",
            None,
        ),
        "relationship-write": (
            rig.database,
            "upsert_character_relationship",
            "PUT",
            "/api/characters/role/relationship",
            {"stage": "close", "summary": "修正后的合成关系"},
        ),
        "memory-list": (rig.database, "list_character_memory_claims", "GET", "/api/characters/role/memories", None),
        "memory-write": (
            rig.database,
            "append_character_memory_claim",
            "PUT",
            memory_url,
            {"content": "用户报告：" + BODY, "importance": 0.7},
        ),
        "erase": (rig.database, "erase_character_memories", "DELETE", memory_url, None),
        "clear": (rig.database, "clear_character_memories", "DELETE", "/api/characters/role/memories", None),
    }
    target, method, verb, url, body = operations[operation]
    marker = "synthetic_private_character_store_fault"

    def snapshot():
        return {
            table: rig.database.execute_sql(f"SELECT * FROM {table}")
            for table in ("character_memories", "character_relationships")
        }

    def fail(*args, **kwargs):
        try:
            if operation == "profiles":
                return (rig.root / marker).read_text()
            return rig.database.execute_sql("SELECT " + marker)
        except (OSError, sqlite3.Error) as exc:
            rig.faults.append(exc)
            raise

    healthy = await rig.api.get("/api/characters/role/memories", params=rig.params)
    assert healthy.status_code == 200 and healthy.json()["memories"][0]["evidence"] == [BODY]
    before = snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(target, method, fail)
        failed = await rig.api.request(verb, url, params=rig.params, json=body)
        assert failed.status_code == 500 and failed.json().get("success") is False
        assert failed.json().get("error") == ("Internal Server Error" if operation == "profiles" else "database_error")
        assert rig.handled[-1] is rig.faults[-1]
        assert marker not in failed.text and marker not in caplog.text and marker not in rig.server_logs.getvalue()
        assert snapshot() == before
    repaired = await rig.api.request(verb, url, params=rig.params, json=body)
    assert repaired.status_code == 200 and repaired.json()["success"] is True
    result = repaired.json()
    if operation == "memory-write":
        updated = result["memory"]
        assert updated["content"] == "用户报告：" + BODY and updated["relation_type"] == "SUPERSEDE"
        assert updated["supersedes_memory_id"] == saved["id"] and updated["confidence"] == 0.67
        assert updated["evidence"] == [BODY] and updated["source_message_ids"] == ["complete-source"]
        assert updated["metadata"]["qualifiers"] == saved["metadata"]["qualifiers"]
    elif operation in {"erase", "clear"}:
        assert result["deleted"] == 1
        assert (await rig.api.get("/api/characters/role/memories", params=rig.params)).json()["memories"] == []
    elif operation.startswith("relationship"):
        assert result["relationship"]["relationship_stage"] == ("close" if operation.endswith("write") else "familiar")
    elif operation == "profiles":
        assert result["characters"]
    else:
        assert result["memories"][0]["content"] == BODY


async def test_normal_absence_business_errors_and_admin_boundary_remain_distinct(rig):
    url = f"/api/characters/role/memories/{rig.saved['id']}"
    wrong_owner = {**rig.params, "sender_id": "another-user"}
    assert (await rig.api.get("/api/characters/role/memories", params=wrong_owner)).json()["memories"] == []
    assert (await rig.api.get("/api/characters/role/relationship", params=wrong_owner)).json()["relationship"] is None
    assert (await rig.api.put(url, params=wrong_owner, json={"content": BODY})).status_code == 404
    assert (await rig.api.delete(url, params=wrong_owner)).status_code == 404
    assert (await rig.api.put(url, params=rig.params, json={"content": "   "})).status_code == 400
    assert (await rig.api.put(url, params=rig.params, json={"content": BODY, "resolved": True})).status_code == 400
    token = rig.api.headers.pop("Authorization")
    assert (await rig.api.get("/api/characters")).status_code == 401
    rig.api.headers["Authorization"] = token
    rig.database.execute_sql("UPDATE users SET role='user' WHERE username='synthetic-admin'")
    assert (await rig.api.get("/api/characters")).status_code == 403
    assert (await rig.api.delete(url, params=rig.params)).status_code == 403
    stored = await rig.repo.get_memory_record(
        rig.saved["id"],
        "role",
        build_user_scope("web", "isolated-character-api", "synthetic-owner", "synthetic-room", "private"),
    )
    assert stored["content"] == BODY and stored["evidence"] == [BODY]
