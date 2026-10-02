import asyncio
import json
from datetime import datetime, timezone

import pytest

from character.models import CompiledCharacterContext, UserScope
from character.source_memory import SourceMemoryService, attach_sources, select_sources
from db.database import SQLiteDB
from db.memory_source import source_scope
from db.memory_source_search import search_plan
from repositories.character_memory import DatabaseCharacterMemoryRepository

FIELDS = dict(
    character_id="role",
    platform="web",
    adapter="contextual-source",
    sender_id="owner",
    conversation_id="room",
    conversation_type="private",
)
SCOPE = UserScope("web", "contextual-source", "owner", "room", "private")


def put(db, sid, body, **extra):
    assert (
        db.capture_memory_source(
            **(FIELDS | extra), source_message_id=sid, body=body, observed_at=datetime.now(timezone.utc)
        )
        == "recorded"
    )


def test_complete_source_only_topic_reaches_context_without_becoming_fact(tmp_path):
    db = SQLiteDB(tmp_path / "source.sqlite")
    body = "青岚项目只有离线RTX3060和12GB，禁止训练。"
    put(db, "target", body)
    repo = DatabaseCharacterMemoryRepository(db)
    service = SourceMemoryService(repo)
    old = asyncio.run(service.recall("role", SCOPE, "按刚才那个继续。"))
    result = asyncio.run(service.recall("role", SCOPE, "按刚才那个继续。", retrieval_context="现在只谈青岚项目。"))
    assert old.diagnostics["status"] == "no_match"
    assert json.loads(result.context)["records"][0]["text"] == body
    assert result.diagnostics["contextual_search_enabled"] and result.diagnostics["contextual_read_count"] == 1
    compiled = attach_sources(
        CompiledCharacterContext("profile", "", "", memory_field_presence=(("device", False),)), result
    )
    assert compiled.memory_packets == () and compiled.memory_field_presence == (("device", None),)
    assert asyncio.run(repo.list_memory_records("role", SCOPE, limit=None)) == []


def test_three_lanes_keep_current_query_linked_and_contextual_leaders():
    def row(sid):
        return dict(source_message_id=sid, observed_at="2026-10-01T00:00:00+00:00", body=sid)

    selected, count = select_sources([row("linked")], [row("current"), row("extra")], contextual=[row("history")])
    assert [r["source_message_id"] for r in selected][:3] == ["linked", "current", "history"] and count == 4


class RecordingRepository:
    def __init__(self, rows=(), *, omit_fresh=False, fail_history=False):
        self.rows = list(rows)
        self.queries = []
        self.omit_fresh = omit_fresh
        self.fail_history = fail_history

    async def linked_sources(self, *a, **kw):
        return []

    async def search_sources(self, *a, query, limit):
        self.queries.append((query, limit))
        if query == "history" and self.fail_history:
            raise RuntimeError("isolated simulated read failure")
        return list(self.rows) if query == "history" else []

    async def list_sources(self, *a, source_message_ids):
        return [] if self.omit_fresh else [r for r in self.rows if r["source_message_id"] in source_message_ids]


def test_empty_context_has_no_additional_search():
    repo = RecordingRepository()
    result = asyncio.run(SourceMemoryService(repo).recall("role", SCOPE, "current", retrieval_context="  "))
    assert repo.queries == [("current", 32)] and not result.diagnostics["contextual_search_enabled"]


def test_complete_long_context_keeps_both_ends_with_constant_sql_bindings(tmp_path):
    db = SQLiteDB(tmp_path / "long.sqlite")
    put(db, "head", "headtarget")
    put(db, "tail", "tailtarget")
    context = "headtarget " + ("irrelevantfiller " * 650) + " tailtarget"
    assert len(context) > 8000
    result = asyncio.run(
        SourceMemoryService(DatabaseCharacterMemoryRepository(db)).recall(
            "role", SCOPE, "continue", retrieval_context=context
        )
    )
    assert set(result.diagnostics["selected_ids"]) == {"head", "tail"}
    sql, params = next(search_plan(source_scope(**FIELDS), context))
    assert (
        "json_each" in sql
        and len(params) < 20
        and all(t in params["query_terms"] for t in ["headtarget", "tailtarget"])
    )


def test_context_lane_keeps_owner_role_visibility_and_erasure(tmp_path):
    db = SQLiteDB(tmp_path / "grants.sqlite")
    put(db, "owned", "青岚约束")
    put(db, "foreign", "青岚约束", sender_id="another")
    put(db, "other-role", "青岚约束", character_id="other")
    put(db, "public", "青岚约束", conversation_type="group", conversation_id="public-room")
    repo = DatabaseCharacterMemoryRepository(db)
    service = SourceMemoryService(repo)
    result = asyncio.run(service.recall("role", SCOPE, "继续", retrieval_context="青岚"))
    assert result.diagnostics["selected_ids"] == ["owned"]
    claim = db.append_character_memory_claim(
        **FIELDS, memory_type="user_fact", memory_key="erase", content="青岚约束", source_message_id="owned"
    )
    db.erase_character_memories(**FIELDS, memory_id=claim["id"])
    after = asyncio.run(service.recall("role", SCOPE, "继续", retrieval_context="青岚"))
    assert after.diagnostics["status"] == "no_match"


def test_fresh_grant_recheck_drops_context_candidate():
    repo = RecordingRepository(
        [dict(source_message_id="gone", body="complete", observed_at="2026-10-01T00:00:00+00:00")], omit_fresh=True
    )
    result = asyncio.run(SourceMemoryService(repo).recall("role", SCOPE, "current", retrieval_context="history"))
    assert (
        result.context == ""
        and result.diagnostics["selected_ids"] == []
        and result.diagnostics["anchor_ids"] == ["gone"]
    )


def test_context_read_error_is_not_claimed_absence():
    repo = RecordingRepository(fail_history=True)
    result = asyncio.run(SourceMemoryService(repo).recall("role", SCOPE, "current", retrieval_context="history"))
    assert result.diagnostics["status"] == "retrieval_error" and result.context == ""


def test_context_match_cannot_sneak_partial_source_through_budget():
    repo = RecordingRepository(
        [dict(source_message_id="long", body="complete " * 400, observed_at="2026-10-01T00:00:00+00:00")]
    )
    result = asyncio.run(SourceMemoryService(repo).recall("role", SCOPE, "current", retrieval_context="history"))
    assert (
        result.context == ""
        and result.diagnostics["status"] == "budget_omitted"
        and result.diagnostics["omitted"][0]["reason"] == "whole_source_budget"
    )


def test_context_search_rejects_nontext_query():
    with pytest.raises(ValueError):
        next(search_plan(source_scope(**FIELDS), {"topic": "青岚"}))
