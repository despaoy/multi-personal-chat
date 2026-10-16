"""Complete stored claims must have explicit, valid lifecycle values."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from api.generate import _prepare_character_turn
from character.memory_service import CharacterMemoryService
from character.models import MemoryItem
from db.schemas import MessageRequest
from inference.generation_request import generate_character_response
from repositories.character_memory import MemoryRecordError, _decode_memory_record
from tests.test_structured_memory_authority import structured_request


@pytest.mark.parametrize("field", ["status", "relation_type"])
@pytest.mark.parametrize("value", ["", "synthetic-invalid-lifecycle"])
async def test_corrupt_lifecycle_fails_reads_and_stops_generation(tmp_path, field, value):
    request, _prepared, repo, db, scope, target, _public = await structured_request(tmp_path)
    connection = db._get_connection()
    model = AsyncMock(return_value="不应生成")
    try:
        connection.execute(f"UPDATE character_memories SET {field}=? WHERE id=?", (value, int(target.memory_id)))
        connection.commit()
        for read in (
            lambda: repo.get_memory_record(int(target.memory_id), "fiction-role", scope),
            lambda: repo.list_memory_records("fiction-role", scope, include_inactive=True),
            lambda: CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
                "fiction-role", scope, request.message, include_historical=True
            ),
            lambda: generate_character_response(request, model),
        ):
            with pytest.raises(ValueError, match=field) as caught:
                await read()
            assert caught.value.field == field
            assert "synthetic-invalid-lifecycle" not in str(caught.value)
        model.assert_not_awaited()
        assert connection.execute(
            f"SELECT {field} FROM character_memories WHERE id=?", (int(target.memory_id),)
        ).fetchone()[0] == value
    finally:
        db.close_connection()


@pytest.mark.parametrize("field,value", [
    ("status", "synthetic-private-invalid-status"),
    ("relation_type", ""),
    ("evidence_json", '[{"summary":"synthetic-private-body"}]'),
    ("source_message_ids_json", '[17]'),
    ("metadata_json", '{synthetic-private-invalid-json'),
])
async def test_api_reports_safe_memory_field_failure(field, value):
    record = dict(memory_type="user_fact", status="active", relation_type="ADD")
    record[field] = value
    with pytest.raises(MemoryRecordError) as caught:
        _decode_memory_record(record)
    service = SimpleNamespace(prepare_turn=AsyncMock(side_effect=caught.value))
    request = MessageRequest(message="核对已保存的完整合成记忆。", platform="web", adapter="lifecycle-control",
                             senderId="fiction-owner", conversationId="fiction-owner", conversationType="private")
    with pytest.raises(HTTPException) as reported:
        await _prepare_character_turn(request, "fiction-role", character_service=service)
    assert reported.value.status_code == 503
    assert reported.value.detail["stage"] == "memory_read"
    assert reported.value.detail["reason"] == "invalid_memory_record"
    assert reported.value.detail["field"] == field
    assert "synthetic-private" not in str(reported.value.detail)
    assert reported.value.__cause__ is caught.value
    service.prepare_turn.assert_awaited_once()


@pytest.mark.parametrize("relation", ["", None])
async def test_empty_relation_is_rejected_before_write(tmp_path, relation):
    _request, _prepared, repo, db, scope, _target, _public = await structured_request(tmp_path)
    try:
        before = await repo.list_memory_records("fiction-role", scope)
        with pytest.raises(MemoryRecordError, match="relation_type"):
            await repo.append_claim("fiction-role", scope, MemoryItem("", "user_fact", "合成不应写入的事实"),
                                    memory_key="rejected_relation", relation_type=relation)
        assert await repo.list_memory_records("fiction-role", scope) == before
    finally:
        db.close_connection()


async def test_pending_relation_keeps_explicit_operation_default(tmp_path):
    _request, _prepared, repo, db, scope, _target, _public = await structured_request(tmp_path)
    try:
        saved = await repo.append_claim("fiction-role", scope, MemoryItem("", "user_fact", "合成待确认事实"),
                                        memory_key="pending_control", relation_type="PENDING")
        assert saved["relation_type"] == "PENDING" and saved["status"] == "pending"
        assert saved["id"] not in {row["id"] for row in await repo.list_memory_records("fiction-role", scope)}
        assert saved["persisted"] is True
        assert await repo.get_memory_record(saved["id"], "fiction-role", scope) == {
            key: value for key, value in saved.items() if key != "persisted"
        }
    finally:
        db.close_connection()
