"""Complete ranked searches have no arbitrary row cap in serving-budget mode."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from db.memory_source import source_scope
from db.memory_source_search import search_plan
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("web", "search-capacity", "owner", "owner", "private")
FIELDS = dict(
    character_id="role",
    platform="web",
    adapter="search-capacity",
    sender_id="owner",
    conversation_type="private",
    conversation_id="owner",
)


def populate(db, count=40, **extra):
    stamp = datetime.now(timezone.utc) - timedelta(minutes=10)
    for i in range(count):
        assert (
            db.capture_memory_source(
                **(FIELDS | extra),
                source_message_id=f"source-{i:03d}",
                body=f"XR493-{i:03d}：数值{i + 20}，限制：未复核禁用。",
                observed_at=stamp + timedelta(seconds=i),
            )
            == "recorded"
        )


@pytest.mark.parametrize("count", [40, 130])
async def test_complete_search_and_serving_recall_preserve_every_stored_match(tmp_path, count):
    db = SQLiteDB(tmp_path / "sources.sqlite")
    populate(db, count)
    repo = DatabaseCharacterMemoryRepository(db)
    complete = await repo.search_sources("role", SCOPE, query="XR493", limit=None)
    assert len(complete) == count
    bounded = await repo.search_sources("role", SCOPE, query="XR493")
    assert len(bounded) == 32 and bounded == complete[:32]
    result = await SourceMemoryService(repo, max_chars=50000, defer_budget=True).recall("role", SCOPE, "XR493 全部原话")
    assert len(json.loads(result.context)["records"]) == count
    assert result.diagnostics["source_search_limit"] is None and not result.diagnostics["candidate_limit_reached"]
    assert result.diagnostics["selection_omitted"] == result.diagnostics["fresh_recheck_omitted"] == 0
    legacy = await SourceMemoryService(repo, max_chars=50000).recall("role", SCOPE, "XR493 全部原话")
    assert legacy.diagnostics["indexed_read_count"] == 32 and legacy.diagnostics["candidate_limit_reached"]
    assert len(json.loads(legacy.context)["records"]) == 4


async def test_complete_history_lane_preserves_matches_after_thirty_two(tmp_path):
    db = SQLiteDB(tmp_path / "history.sqlite")
    populate(db)
    repo = DatabaseCharacterMemoryRepository(db)
    result = await SourceMemoryService(repo, max_chars=32768, defer_budget=True).recall(
        "role", SCOPE, "继续", retrieval_context="当前讨论 XR493 项目"
    )
    assert result.diagnostics["indexed_read_count"] == 0 and result.diagnostics["contextual_read_count"] == 40
    assert len(json.loads(result.context)["records"]) == 40


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_complete_plan_preserves_rank_terms_scope_and_erasure_checks(dialect):
    scope = source_scope(**FIELDS)
    sql, params = next(search_plan(scope, "XR493 尾部限定", limit=None, dialect=dialect))
    assert "LIMIT" not in sql and "ORDER BY r.score DESC, s.observed_at DESC, s.source_key DESC" in sql
    assert "memory_source_fences" in sql and sql.count("state = 'recorded'") == 2
    assert "t.scope_key = :scope_key" in sql and "s.scope_key = :scope_key" in sql
    assert params["scope_key"] == scope["scope_key"] and "xr493" in json.loads(params["query_terms"])
    limited_sql, limited_params = next(search_plan(scope, "XR493 尾部限定", limit=32, dialect=dialect))
    assert limited_sql == sql + " LIMIT :limit" and limited_params["limit"] == 32
    assert len(params) == len(limited_params)


@pytest.mark.parametrize("invalid", [True, 0, 101, 1.5, "all"])
def test_explicit_invalid_row_limits_still_rejected(invalid):
    with pytest.raises(ValueError):
        next(search_plan(source_scope(**FIELDS), "XR493", limit=invalid))


async def test_complete_search_still_excludes_foreign_owner_role_and_group(tmp_path):
    db = SQLiteDB(tmp_path / "scope.sqlite")
    populate(db, 40)
    populate(db, 3, sender_id="foreign", conversation_id="foreign")
    populate(db, 3, character_id="other")
    populate(db, 3, conversation_type="group", conversation_id="public")
    repo = DatabaseCharacterMemoryRepository(db)
    rows = await repo.search_sources("role", SCOPE, query="XR493", limit=None)
    assert len(rows) == 40
    db.clear_character_memories(**FIELDS)
    assert await repo.search_sources("role", SCOPE, query="XR493", limit=None) == []


async def test_complete_candidate_list_still_passes_fresh_grant_after_erasure(tmp_path):
    db = SQLiteDB(tmp_path / "erase.sqlite")
    populate(db)

    class ErasingRepo(DatabaseCharacterMemoryRepository):
        async def search_sources(self, *a, **kw):
            rows = await super().search_sources(*a, **kw)
            db.clear_character_memories(**FIELDS)
            return rows

    result = await SourceMemoryService(ErasingRepo(db), max_chars=32768, defer_budget=True).recall(
        "role", SCOPE, "XR493"
    )
    assert result.context == result.candidate_context == ""
    assert result.diagnostics["fresh_recheck_omitted"] == 40
