"""Local expansion keeps omitted vocabulary/identity separate from semantics."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from character.models import MemoryItem, UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from db.memory_source import source_scope
from db.memory_source_window import window_plan
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("test", "windows", "u", "u", "private")
FIELDS = dict(character_id="role", platform="test", adapter="windows", sender_id="u",
              conversation_type="private", conversation_id="u")
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def put(db, key, body, offset, **scope):
    return db.capture_memory_source(**(FIELDS | scope), source_message_id=key, body=body,
                                    observed_at=START + timedelta(seconds=offset))


def test_windows_are_anchored_not_latest_history_and_cannot_cross_scope(tmp_path):
    db = SQLiteDB(tmp_path / "windows.sqlite")
    put(db, "before", "第一个背景。", 0)
    put(db, "anchor", "周末去爬山，之后去游泳。", 1)
    put(db, "other", "另一角色的内容。", 1.5, character_id="other")
    put(db, "update", "第二件取消，第一件继续。", 2)
    for i in range(260):
        put(db, f"later-{i}", "后来无关的闲聊。", i + 3)
    window, = db.memory_source_windows(**FIELDS, source_message_ids=("anchor",))
    assert window["anchor_id"] == "anchor"
    assert {row["source_message_id"] for row in window["rows"]} == {"before", "anchor", "update"}
    assert db.memory_source_windows(**(FIELDS | {"character_id": "other"}), source_message_ids=("anchor",)) == []


def test_same_timestamp_has_deterministic_neighbors_and_index_seek(tmp_path):
    db = SQLiteDB(tmp_path / "tie.sqlite")
    for i in range(100):
        put(db, f"s{i:03}", "相同时刻的独立原话。", 0)
    window, = db.memory_source_windows(**FIELDS, source_message_ids=("s050",))
    assert {row["source_message_id"] for row in window["rows"]} == {"s049", "s050", "s051"}
    plan = window_plan(source_scope(**FIELDS), ("s050",))
    rows = None
    details = []
    while True:
        try:
            sql, params = plan.send(rows)
        except StopIteration:
            break
        details.extend(row["detail"] for row in db._get_connection().execute("EXPLAIN QUERY PLAN " + sql, params))
        rows = [dict(row) for row in db._get_connection().execute(sql, params)]
    assert any("idx_memory_sources_scope" in detail and "observed_at=?" in detail for detail in details)
    assert not any("SCAN memory_sources" in detail for detail in details)


@pytest.mark.asyncio
async def test_source_only_revision_is_recalled_without_query_overlap(tmp_path):
    db = SQLiteDB(tmp_path / "revision.sqlite")
    put(db, "plan", "周末去爬山，之后去游泳。", 0)
    put(db, "revision", "第二件取消，第一件继续。", 1)
    repo = DatabaseCharacterMemoryRepository(db)
    old = await SourceMemoryService(repo).recall("role", SCOPE, "爬山游泳")
    new = await SourceMemoryService(repo, window_radius=1).recall("role", SCOPE, "爬山游泳")
    assert old.diagnostics["selected_ids"] == ["plan"]
    assert new.diagnostics["selected_ids"] == ["plan", "revision"]
    assert json.loads(new.context)["current_validity"] == "not_resolved"
    assert new.diagnostics["window_semantic_relation"] == "not_inferred"


@pytest.mark.asyncio
async def test_complete_fact_dedup_does_not_hide_neighbor_revision(tmp_path):
    db = SQLiteDB(tmp_path / "dedup.sqlite")
    put(db, "fact", "我喜欢围棋。", 0)
    put(db, "revision", "之前说的那个兴趣已经放弃了。", 1)
    item = MemoryItem("1", "user_fact", "用户喜欢围棋", evidence=("我喜欢围棋。",),
                      observed_at=START.isoformat())
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), window_radius=1).recall(
        "role", SCOPE, "围棋", memories=(item,))
    assert result.diagnostics["selected_ids"] == ["fact", "revision"]


@pytest.mark.asyncio
async def test_window_budget_cannot_leave_old_anchor_without_its_long_correction(tmp_path):
    db = SQLiteDB(tmp_path / "budget.sqlite")
    put(db, "plan", "周末去爬山。", 0)
    put(db, "update", "补充细节。" * 300 + "刚才的安排取消。", 1)
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), window_radius=1, max_chars=500).recall(
        "role", SCOPE, "爬山")
    assert result.context == "" and result.diagnostics["status"] == "budget_omitted"


@pytest.mark.asyncio
async def test_erased_sources_cannot_return_via_window_or_post_ranking_cache(tmp_path):
    db = SQLiteDB(tmp_path / "erased.sqlite")
    put(db, "plan", "周末去爬山。", 0)
    put(db, "update", "那件事取消了。", 1)

    class ErasingRepo(DatabaseCharacterMemoryRepository):
        async def source_windows(self, *args, **kwargs):
            windows = await super().source_windows(*args, **kwargs)
            db.clear_character_memories(**FIELDS)
            return windows

    result = await SourceMemoryService(ErasingRepo(db), window_radius=1).recall("role", SCOPE, "爬山")
    assert result.context == ""
    assert db.memory_source_windows(**FIELDS, source_message_ids=("plan",)) == []


@pytest.mark.parametrize("ids,radius", [("x", 1), (("x",) * 5, 3), (("a", "b", "c", "d", "e"), 1), (("x",), True)])
def test_window_api_bounds(tmp_path, ids, radius):
    db = SQLiteDB(tmp_path / "bounds.sqlite")
    with pytest.raises(ValueError):
        db.memory_source_windows(**FIELDS, source_message_ids=ids, radius=radius)
