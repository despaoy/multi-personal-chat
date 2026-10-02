import json
from datetime import datetime, timedelta, timezone

import pytest

from character.models import UserScope
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from db.memory_source import source_scope
from db.memory_source_search import literal_project_source_terms, search_plan
from evaluation.complete_source_role_probe import required_fixture_sources
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("web", "project-source", "owner", "owner", "private")
FIELDS = dict(
    character_id="role",
    platform="web",
    adapter="project-source",
    sender_id="owner",
    conversation_type="private",
    conversation_id="owner",
)
QUERY = "请逐字列出之前的 QS729 项目全部原始交接记录，按QS729-01至QS729-04排列，保留末尾限制。"


@pytest.mark.parametrize(
    "query,tags",
    [
        (QUERY, ("qs729",)),
        ("查看qs729项目原话", ("qs729",)),
        ("列出QS729项目和AB831项目原始记录", ("ab831", "qs729")),
        ("查看项目QS729原话", ()),
        ("翻译QS729项目原话", ()),
        ("QS729项目进展怎么样", ()),
        ("查看QS729项目原话，以及我的名字", ()),
        ("查看QS729项目和AB831的原始记录", ()),
        ("查看QS729项目原话，排除AB831项目", ()),
        ("查看AB-QS729项目原话", ()),
        ("查看青岚项目原话", ()),
    ],
)
def test_only_complete_literal_project_read_scope_is_constrained(query, tags):
    assert literal_project_source_terms(query) == tags


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_project_scope_is_bound_inside_the_authorized_index_query(dialect):
    sql, params = next(search_plan(source_scope(**FIELDS), QUERY, limit=None, dialect=dialect))
    assert "SELECT DISTINCT anchor.source_key" in sql and "anchor.scope_key = :scope_key" in sql
    assert "project_scope.source_key = t.source_key" in sql and "qs729" not in sql
    assert json.loads(params["project_terms"]) == ["qs729"]
    assert "LIMIT" not in sql and sql.count("state = 'recorded'") == 2
    assert "memory_source_fences" in sql and "ORDER BY r.score DESC" in sql


def put(db, sid, body, i=0, **extra):
    stamp = datetime.now(timezone.utc) - timedelta(minutes=10) + timedelta(seconds=i)
    assert (
        db.capture_memory_source(**(FIELDS | extra), source_message_id=sid, body=body, observed_at=stamp) == "recorded"
    )


@pytest.mark.parametrize("target_count,noise_count", [(4, 128), (55, 6)])
async def test_all_named_project_sources_survive_large_other_project_matches(tmp_path, target_count, noise_count):
    db = SQLiteDB(tmp_path / "sources.sqlite")
    expected = {}
    for i in range(target_count):
        expected[f"target-{i}"] = f"QS729-{i:02d}：数值{i + 80}，限制：未复核禁用。"
        put(db, f"target-{i}", expected[f"target-{i}"], i)
    for i in range(noise_count):
        put(
            db,
            f"noise-{i}",
            f"NX{i + 400} 项目原始交接记录。" + ("数值需要按编号排列，保留末尾限制。" * 180),
            i + target_count,
        )
    repo = DatabaseCharacterMemoryRepository(db)
    rows = await repo.search_sources("role", SCOPE, query=QUERY, limit=None)
    assert {r["source_message_id"]: r["body"] for r in rows} == expected
    # A formerly loaded other topic cannot override the newly explicit task.
    result = await SourceMemoryService(repo, max_chars=16384, defer_budget=True).recall(
        "role", SCOPE, QUERY, retrieval_context="此前讨论NX400项目原始交接记录，数值和末尾限制。"
    )
    assert {r["source_id"]: r["text"] for r in json.loads(result.context)["records"]} == expected
    assert result.diagnostics["literal_project_terms"] == ["qs729"]
    assert result.diagnostics["contextual_scope_deferred_to_explicit_task"]
    assert result.diagnostics["selection_omitted"] == result.diagnostics["fresh_recheck_omitted"] == 0


async def test_project_tag_does_not_cross_owner_role_or_reactivate_erased_sources(tmp_path):
    db = SQLiteDB(tmp_path / "scope.sqlite")
    put(db, "valid", "QS729：原始记录，限制：禁用。")
    put(db, "foreign", "QS729：原始记录，限制：外国用户。", sender_id="other", conversation_id="other")
    put(db, "role", "QS729：原始记录，限制：其他角色。", character_id="other")
    repo = DatabaseCharacterMemoryRepository(db)
    assert [r["source_message_id"] for r in await repo.search_sources("role", SCOPE, query=QUERY, limit=None)] == [
        "valid"
    ]
    assert await repo.search_sources("role", SCOPE, query="查看ZZ551项目原始记录", limit=None) == []
    db.clear_character_memories(**FIELDS)
    assert await repo.search_sources("role", SCOPE, query=QUERY, limit=None) == []


async def test_explicit_project_read_keeps_both_named_projects_without_generic_noise(tmp_path):
    db = SQLiteDB(tmp_path / "both.sqlite")
    for sid, body in [("a", "QS729原始记录。"), ("b", "AB831原始记录。"), ("c", "NX400原始记录。")]:
        put(db, sid, body)
    rows = await DatabaseCharacterMemoryRepository(db).search_sources(
        "role", SCOPE, query="列出QS729项目和AB831项目原始记录", limit=None
    )
    assert {r["source_message_id"] for r in rows} == {"a", "b"}


async def test_unrelated_linked_observation_cannot_crowd_explicit_source_task(tmp_path):
    db = SQLiteDB(tmp_path / "linked.sqlite")
    put(db, "target", "QS729：原始记录，限制：禁用。")
    put(db, "noise", "NX400原始记录。" + "原始交接记录限制。" * 40000)

    class Repo(DatabaseCharacterMemoryRepository):
        async def linked_sources(self, *a, **kw):
            return await self.list_sources(*a, source_message_ids=("noise",))

    result = await SourceMemoryService(Repo(db), max_chars=16384, defer_budget=True).recall("role", SCOPE, QUERY)
    assert [r["source_id"] for r in json.loads(result.context)["records"]] == ["target"]
    assert result.diagnostics["linked_project_scope_omitted"] == 1
    assert result.diagnostics["linked_read_count"] == 1 and result.diagnostics["linked_count"] == 0


def test_fixture_declares_required_sources_without_reducing_full_stored_input():
    fixture = {
        "sources": [{"id": "needed", "body": "完整原话"}, {"id": "noise", "body": "完整干扰原话"}],
        "required_source_ids": ["needed"],
    }
    assert required_fixture_sources(fixture) == [fixture["sources"][0]]
    assert len(fixture["sources"]) == 2
    assert required_fixture_sources({"sources": fixture["sources"]}) == fixture["sources"]


@pytest.mark.parametrize("required", [[], ["unknown"], ["needed", "needed"], "needed"])
def test_fixture_cannot_relax_required_source_gate_with_invalid_ids(required):
    with pytest.raises(ValueError):
        required_fixture_sources({"sources": [{"id": "needed"}], "required_source_ids": required})
