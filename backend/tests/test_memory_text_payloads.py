"""Malformed list elements must not become proof or source identities.

All storage cases begin with complete, scoped SQLite source/claim bindings.
No live model or vector quality evaluation is performed.
"""

import json
from datetime import datetime, timezone

import pytest

from character.memory_service import CharacterMemoryService
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from tests.test_structured_memory_authority import structured_request


@pytest.mark.parametrize("field", ["evidence_json", "source_message_ids_json"])
@pytest.mark.parametrize("element", [False, 17, {"summary": "合成伪证据，原话中不存在。"}, ["嵌套文本"], "  "])
async def test_malformed_text_elements_fail_scoped_reads_and_recall(tmp_path, field, element):
    request, _prepared, repo, db, scope, target, _public = await structured_request(tmp_path)
    encoded = json.dumps([element], ensure_ascii=False)
    connection = db._get_connection()
    try:
        connection.execute(f"UPDATE character_memories SET {field}=? WHERE id=?", (encoded, int(target.memory_id)))
        connection.commit()
        for read in (
            lambda: repo.get_memory_record(int(target.memory_id), "fiction-role", scope),
            lambda: repo.list_memory_records("fiction-role", scope),
            lambda: CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
                "fiction-role", scope, request.message
            ),
        ):
            with pytest.raises(ValueError, match=field) as caught:
                await read()
            assert "合成伪证据" not in str(caught.value)
            assert "嵌套文本" not in str(caught.value)
        assert connection.execute(
            f"SELECT {field} FROM character_memories WHERE id=?", (int(target.memory_id),)
        ).fetchone()[0] == encoded
    finally:
        db.close_connection()


@pytest.mark.parametrize("field,kwargs", [
    ("evidence_json", {"evidence": ({"summary": "合成伪证据"},)}),
    ("source_message_ids_json", {"source_message_ids": (17,)}),
])
async def test_invalid_write_payload_is_rejected_before_persistence(tmp_path, field, kwargs):
    _request, _prepared, repo, db, scope, _target, _public = await structured_request(tmp_path)
    try:
        before = await repo.list_memory_records("fiction-role", scope)
        with pytest.raises(ValueError, match=field):
            await repo.append_claim("fiction-role", scope, MemoryItem("", "user_fact", "合成写入内容"),
                                    memory_key="rejected_payload", **kwargs)
        assert await repo.list_memory_records("fiction-role", scope) == before
    finally:
        db.close_connection()


async def test_attributed_quote_survives_write_read_and_scoped_recall(tmp_path):
    db = SQLiteDB(tmp_path / "attributed-quote.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "quote-contract", "fiction-owner", "fiction-owner", "private")
    source_id = "fiction-quote-0"
    body = "我喜欢乌龙茶；仅在周五喝，出差时例外。"
    quote = dict(message_id=source_id, quote=body)
    try:
        assert await repo.capture_source("fiction-role", scope, source_message_id=source_id, body=body,
                                         observed_at=datetime.now(timezone.utc)) == "recorded"
        saved = await repo.append_claim(
            "fiction-role", scope, MemoryItem("", "user_fact", "用户说喜欢乌龙茶", .9),
            memory_key="preference_乌龙茶", evidence=(quote,), source_message_id=source_id,
            source_message_ids=(source_id,),
        )
        assert saved["evidence"] == [quote]
        assert json.loads(saved["evidence_json"]) == [quote]
        assert saved["source_message_ids"] == [source_id]
        memories, count = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
            "fiction-role", scope, "我保存的乌龙茶偏好是什么？"
        )
        assert count == 1 and len(memories) == 1
        assert memories[0].evidence == (body,)
        assert memories[0].source_message_ids == (source_id,)
    finally:
        db.close_connection()
