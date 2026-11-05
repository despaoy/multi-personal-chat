"""Manual corrections preserve certainty; invalid speaker metadata never becomes user."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.characters import router
from app.dependencies import get_current_admin
from app.providers import get_character_memory_repository
from character.memory_read_authority import MemoryRecordError
from character.models import MemoryItem, build_user_scope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository, _decode_memory_record

BODY = "我可能喜欢蓝莓茶，还没有确定；只有夜间休息时才考虑，出差时不适用。"
QUALIFIERS = {"condition": "只有夜间休息时才考虑", "exception": "出差时不适用"}


@pytest.fixture
def rig(tmp_path):
    database = SQLiteDB(tmp_path / "manual-values.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = build_user_scope("web", "manual-values", "owner", "isolated-room", "private")
    app = FastAPI()
    app.include_router(router)
    # Existing authorization behavior has its own tests; this fixture exercises
    # the actual route, repository, version transaction and source links.
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    app.dependency_overrides[get_character_memory_repository] = lambda: repository
    params = {
        name: getattr(scope, name)
        for name in ("platform", "adapter", "sender_id", "conversation_id", "conversation_type")
    }
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(database=database, repo=repository, scope=scope, client=client, params=params)
    finally:
        database.close_connection()


async def save_original(rig, confidence=0.67):
    assert (
        await rig.repo.capture_source(
            "role",
            rig.scope,
            source_message_id="complete-source",
            body=BODY,
            observed_at=datetime(2026, 11, 5, 1, tzinfo=timezone.utc),
        )
        == "recorded"
    )
    return await rig.repo.append_claim(
        "role",
        rig.scope,
        MemoryItem("", "user_fact", BODY, 0.7),
        memory_key="preference_blueberry_tea",
        confidence=confidence,
        source_message_id="complete-source",
        source_message_ids=("complete-source",),
        evidence=(BODY,),
        metadata={"qualifiers": QUALIFIERS},
    )


@pytest.mark.parametrize("confidence", [0.0, 0.67], ids=["zero", "ordinary"])
async def test_manual_edit_preserves_confidence_scope_complete_evidence_and_source(rig, confidence):
    original = await save_original(rig, confidence)
    url = f"/api/characters/role/memories/{original['id']}"
    assert rig.client.put(url, params={**rig.params, "sender_id": "other"}, json={"content": BODY}).status_code == 404
    response = rig.client.put(url, params=rig.params, json={"content": "用户报告：" + BODY, "importance": 0.7})
    assert response.status_code == 200
    updated = response.json()["memory"]
    assert updated["confidence"] == original["confidence"] == confidence
    assert updated["relation_type"] == "SUPERSEDE" and updated["supersedes_memory_id"] == original["id"]
    assert updated["memory_type"] == original["memory_type"] and updated["memory_key"] == original["memory_key"]
    assert updated["scope_level"] == original["scope_level"] and updated["attributed_to"] == "user"
    assert updated["evidence"] == original["evidence"] == [BODY]
    assert updated["source_message_ids"] == original["source_message_ids"] == ["complete-source"]
    assert updated["metadata"]["qualifiers"] == QUALIFIERS
    assert updated["metadata"]["manual_correction"] and updated["metadata"]["corrected_memory_id"] == original["id"]
    stored = await rig.repo.get_memory_record(updated["id"], "role", rig.scope)
    assert stored["confidence"] == confidence and stored["content"] == "用户报告：" + BODY
    previous = await rig.repo.get_memory_record(original["id"], "role", rig.scope)
    assert previous["status"] == "superseded" and previous["confidence"] == confidence
    sources = await rig.repo.linked_source_receipts(
        "role", rig.scope, claim_sources=((updated["id"], "complete-source"),)
    )
    assert len(sources) == 1 and sources[0]["body"] == BODY
    # The historical absent-field contract remains distinct from an explicitly
    # invalid stored value. New writes use the method's declared user default.
    legacy = _decode_memory_record(
        dict(memory_type="user_fact", status="active", relation_type="ADD", metadata_json="{}")
    )
    assert legacy["attributed_to"] == "user"


@pytest.mark.parametrize("path", ["write", "read"])
@pytest.mark.parametrize("attribution", [None, "assistant"], ids=["null", "unsupported-speaker"])
async def test_invalid_attribution_fails_without_reclassification_or_successor(rig, path, attribution):
    original = await save_original(rig)
    if path == "write":
        with pytest.raises(MemoryRecordError, match="attributed_to") as error:
            await rig.repo.append_claim(
                "role",
                rig.scope,
                MemoryItem("", "user_fact", BODY, 0.7),
                memory_key="invalid_speaker",
                evidence=(BODY,),
                confidence=0.67,
                attributed_to=attribution,
                source_message_id="complete-source",
                source_message_ids=("complete-source",),
            )
        assert error.value.field == "attributed_to"
    else:
        metadata = {**original["metadata"], "attributed_to": attribution}
        connection = rig.database._get_connection()
        connection.execute(
            "UPDATE character_memories SET metadata_json=? WHERE id=?", (json.dumps(metadata), original["id"])
        )
        connection.commit()
        with pytest.raises(MemoryRecordError, match="attributed_to") as error:
            await rig.repo.get_memory_record(original["id"], "role", rig.scope)
        assert error.value.field == "attributed_to"
        with pytest.raises(MemoryRecordError, match="attributed_to"):
            await rig.repo.list_memory_records("role", rig.scope)
        failed = rig.client.put(
            f"/api/characters/role/memories/{original['id']}", params=rig.params, json={"content": BODY}
        )
        assert failed.status_code == 500 and "memory" not in failed.json()
    assert rig.database._get_connection().execute("SELECT COUNT(*) FROM character_memories").fetchone()[0] == 1

    if path == "read":
        connection.execute(
            "UPDATE character_memories SET metadata_json=? WHERE id=?",
            (json.dumps(original["metadata"]), original["id"]),
        )
        connection.commit()
        recovered = rig.client.put(
            f"/api/characters/role/memories/{original['id']}", params=rig.params, json={"content": BODY}
        )
        assert recovered.status_code == 200
        repaired = recovered.json()["memory"]
    else:
        repaired = await rig.repo.append_claim(
            "role",
            rig.scope,
            MemoryItem("", "user_fact", BODY, 0.7),
            memory_key="invalid_speaker",
            evidence=(BODY,),
            confidence=0.67,
            attributed_to="user",
            source_message_id="complete-source",
            source_message_ids=("complete-source",),
        )
    assert repaired["attributed_to"] == "user" and repaired["confidence"] == 0.67
    assert repaired["evidence"] == [BODY] and repaired["source_message_ids"] == ["complete-source"]
    assert rig.database._get_connection().execute("SELECT COUNT(*) FROM character_memories").fetchone()[0] == 2
