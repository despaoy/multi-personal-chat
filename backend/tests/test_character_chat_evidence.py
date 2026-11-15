"""Evidence snapshots survive retries and forks but not selected physical deletion."""

import asyncio
from types import SimpleNamespace

import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from repositories.character_chat import CharacterChatRepository
from services.character_chat import CharacterChatService
from services.character_chat_evidence import CharacterChatCompletion, evidence_snapshot


def result(citations):
    retrieval = SimpleNamespace(
        documents=(
            {"id": "s1", "title": "第一章", "content": "小满送到了灯塔。", "private_path": "/private"},
            {"id": "s2", "title": "未引用资料", "content": "不应展示"},
        ),
        status="ok",
        reason="",
    )
    return SimpleNamespace(
        plan=SimpleNamespace(retrieval=retrieval), response_citations=citations, response_mode="generated"
    )


def test_only_answer_used_sources_are_captured_without_private_metadata():
    snapshot = evidence_snapshot(result([{"source_id": "s1", "line_start": 3, "line_end": 3}]))
    assert snapshot["sources"] == [
        {
            "id": "s1",
            "title": "第一章",
            "excerpt": "小满送到了灯塔。",
            "truncated": False,
            "line_start": 3,
            "line_end": 3,
        }
    ]
    with pytest.raises(ValueError, match="snapshot"):
        evidence_snapshot(result([{"source_id": "invented", "source_title": "模型编造"}]))


def test_snapshot_marks_truncated_text():
    value = result([{"id": "s1"}])
    value.plan.retrieval.documents = ({"id": "s1", "content": "长" * 12001},)
    source = evidence_snapshot(value)["sources"][0]
    assert len(source["excerpt"]) == 12000 and source["truncated"]


def test_source_title_is_from_document_and_duplicate_identity_is_rejected():
    value = result([{'source_id': 's1', 'source_title': '编造的标题'}])
    assert evidence_snapshot(value)['sources'][0]['title'] == '第一章'
    value.plan.retrieval.documents += ({'id': 's1', 'title': '其他原文', 'content': '冲突'},)
    with pytest.raises(ValueError, match='duplicate'):
        evidence_snapshot(value)


def test_generation_replay_branch_and_delete_preserve_correct_evidence(tmp_path):
    db = SQLiteDB(tmp_path / "evidence.db")
    try:
        repo = CharacterChatRepository(db)
        character = repo.characters.create("1", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
        session = repo.create("1", character["id"], "原作")
        evidence = evidence_snapshot(result([{"id": "s1"}]))
        calls = []

        async def complete(*args):
            calls.append(True)
            return CharacterChatCompletion("小满。", "test", evidence)

        chat = CharacterChatService(db, complete)
        kwargs = dict(revision=1, request_id="one", message="谁送到了？")
        first = asyncio.run(chat.generate("1", session["id"], **kwargs))
        assert first["evidence"] == evidence
        assert asyncio.run(chat.generate("1", session["id"], **kwargs)) == first
        assert len(calls) == 1
        assert repo.turns("1", session["id"])[0]["evidence"] == evidence
        branch = repo.create(
            "1",
            character["id"],
            "分支",
            source_session_id=session["id"],
            source_revision=2,
            source_turn_id=first["id"],
            hypothesis="假设没送到",
        )
        assert branch["baseline"]["history"][0]["evidence"] == evidence
        saved = repo.retract("1", session["id"], first["id"], 2)
        assert repo.turns("1", session["id"]) == []
        repo.delete_sessions("1", [{"id": session["id"], "revision": saved["revision"]}])
        assert db.execute_sql("SELECT * FROM character_chat_evidence") == []
        assert repo.get("1", branch["id"])["baseline"]["history"][0]["evidence"] == evidence
        repo.delete_sessions("1", [{"id": branch["id"], "revision": branch["revision"]}])
        assert repo.list("1") == []
    finally:
        db.close_connection()
