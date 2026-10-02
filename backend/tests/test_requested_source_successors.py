import json
from datetime import datetime, timedelta, timezone

import pytest

from character.models import MemoryItem, UserScope
from character.source_memory import SourceMemoryService, requested_source_successors
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("web", "following-record", "owner", "owner", "private")
FIELDS = dict(
    character_id="role",
    platform="web",
    adapter="following-record",
    sender_id="owner",
    conversation_id="owner",
    conversation_type="private",
)
QUERY = "请逐字列出 VX642 项目每条原始交接记录及紧随其后的下一条仍可见用户原话记录，保留数值和限制。"
STAMP = datetime.now(timezone.utc) - timedelta(minutes=10)


def put(db, sid, body, offset=0, **extra):
    assert (
        db.capture_memory_source(
            **(FIELDS | extra), source_message_id=sid, body=body, observed_at=STAMP + timedelta(seconds=offset)
        )
        == "recorded"
    )


def populate(db):
    put(db, "before", "NT480独立原始记录。", 0)
    put(db, "anchor", "VX642：数值119，限制：复核前禁用。", 1)
    put(db, "following", "上一条的完整补充：数值37，限制：不得转交。", 2)
    put(db, "later", "NT481独立原始记录。", 3)


def ids(result):
    return [r["source_id"] for r in json.loads(result.context)["records"]]


@pytest.mark.parametrize(
    "query,expected",
    [
        (QUERY, True),
        (QUERY.replace("紧随其后的", "紧随各条记录的"), True),
        (QUERY.replace("每条", "该条"), False),
        (QUERY.replace("及紧随其后的下一条仍可见用户原话记录", ""), False),
        (QUERY.replace("及紧随", "及不要列出紧随"), False),
        (QUERY + "以及我的名字", False),
        (QUERY.replace("VX642 项目", "VX642项目和AB831项目"), False),
        (QUERY.replace("下一条", "下两条"), False),
    ],
)
def test_only_explicit_each_anchor_next_record_order_is_expanded(query, expected):
    assert requested_source_successors(query) is expected


@pytest.mark.parametrize("configured_radius", [0, 2])
async def test_complete_unlabeled_following_quote_is_kept_without_prior_or_later_noise(tmp_path, configured_radius):
    db = SQLiteDB(tmp_path / "sources.sqlite")
    populate(db)
    repo = DatabaseCharacterMemoryRepository(db)
    result = await SourceMemoryService(
        repo, max_chars=16384, defer_budget=True, window_radius=configured_radius
    ).recall("role", SCOPE, QUERY)
    assert ids(result) == ["anchor", "following"]
    assert result.diagnostics["effective_window_radius"] == 1
    assert result.diagnostics["requested_source_order"] == "next_visible_after_each_anchor"
    assert result.diagnostics["window_semantic_relation"] == "not_inferred"
    assert json.loads(result.context)["current_validity"] == "not_resolved"
    assert await repo.list_memory_records("role", SCOPE, limit=None) == []
    if configured_radius == 0:
        ordinary = await SourceMemoryService(repo, max_chars=16384, defer_budget=True).recall(
            "role", SCOPE, "列出VX642项目原始交接记录"
        )
        assert ids(ordinary) == ["anchor"]


async def test_all_requested_anchor_successors_use_existing_four_anchor_batches(tmp_path):
    db = SQLiteDB(tmp_path / "many.sqlite")
    for i in range(6):
        put(db, f"a{i}", f"VX642-{i}：原始记录，限制：禁用。", i * 2)
        put(db, f"f{i}", f"补充{i}：数值{i + 37}，限制：不得转交。", i * 2 + 1)

    class Repo(DatabaseCharacterMemoryRepository):
        batches = []

        async def source_windows(self, *a, **kw):
            self.batches.append(kw["source_message_ids"])
            assert kw["radius"] == 1
            return await super().source_windows(*a, **kw)

    repo = Repo(db)
    result = await SourceMemoryService(repo, max_chars=16384, defer_budget=True).recall("role", SCOPE, QUERY)
    assert ids(result) == [sid for i in range(6) for sid in (f"a{i}", f"f{i}")]
    assert [len(batch) for batch in repo.batches] == [4, 2]


async def test_same_time_direction_uses_sql_metadata_and_identity_serialization(tmp_path):
    db = SQLiteDB(tmp_path / "tie.sqlite")
    # In JSON identity order aZ precedes a-double-quote; raw ID order is opposite.
    put(db, "aZ", "VX642：原始记录，限制：禁用。")
    put(db, 'a"', "补充原话：数值37，限制：禁用。")
    (window,) = db.memory_source_windows(**FIELDS, source_message_ids=("aZ",), radius=1)
    assert [r["source_message_id"] for r in window["following_rows"]] == ['a"']
    assert window["preceding_rows"] == []
    result = await SourceMemoryService(
        DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True
    ).recall("role", SCOPE, QUERY)
    assert ids(result) == ["aZ", 'a"']


async def test_requested_next_record_cannot_cross_owner_or_role(tmp_path):
    db = SQLiteDB(tmp_path / "scope.sqlite")
    populate(db)
    put(db, "foreign", "外部用户数值99，限制：禁用。", 1.5, sender_id="other", conversation_id="other")
    put(db, "other-role", "其他角色数值66，限制：禁用。", 1.7, character_id="other")
    result = await SourceMemoryService(
        DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True
    ).recall("role", SCOPE, QUERY)
    assert ids(result) == ["anchor", "following"]


async def test_visible_record_order_never_restores_erased_text(tmp_path):
    db = SQLiteDB(tmp_path / "erase.sqlite")
    populate(db)
    repo = DatabaseCharacterMemoryRepository(db)
    assert await repo.erase_unlinked_sources("role", SCOPE, source_message_ids=("following",)) == 1
    result = await SourceMemoryService(repo, max_chars=16384, defer_budget=True).recall("role", SCOPE, QUERY)
    assert ids(result) == ["anchor", "later"]
    assert "数值37" not in result.context
    # Current visible order is not a semantic claim that NT481 belongs to VX642.
    assert result.diagnostics["window_semantic_relation"] == "not_inferred"


@pytest.mark.parametrize("revoked,expected", [("anchor", []), ("following", ["anchor"])])
async def test_fresh_grants_and_anchor_dependency_checked_after_window_selection(tmp_path, revoked, expected):
    db = SQLiteDB(tmp_path / "fresh.sqlite")
    populate(db)

    class Repo(DatabaseCharacterMemoryRepository):
        async def source_windows(self, *a, **kw):
            windows = await super().source_windows(*a, **kw)
            await self.erase_unlinked_sources(*a, source_message_ids=(revoked,))
            return windows

    result = await SourceMemoryService(Repo(db), max_chars=16384, defer_budget=True).recall("role", SCOPE, QUERY)
    assert (ids(result) if result.context else []) == expected
    assert result.diagnostics["fresh_recheck_omitted"] >= 1
    if revoked == "anchor":
        assert result.diagnostics["following_anchor_dependency_omitted"] == 1


@pytest.mark.parametrize("has_following", [True, False])
async def test_fact_quote_dedup_does_not_remove_explicitly_requested_anchor(tmp_path, has_following):
    db = SQLiteDB(tmp_path / "dedup.sqlite")
    if has_following:
        populate(db)
    else:
        put(db, "anchor", "VX642：原始记录，限制：禁用。")
    row = db.list_memory_sources(**FIELDS, source_message_ids=("anchor",))[0]
    item = MemoryItem(
        "1", "user_fact", "selected unconditional reference", evidence=(row["body"],), observed_at=row["observed_at"]
    )
    result = await SourceMemoryService(
        DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True
    ).recall("role", SCOPE, QUERY, memories=(item,))
    assert ids(result) == (["anchor", "following"] if has_following else ["anchor"])


async def test_adapter_without_direction_metadata_fails_unknown_instead_of_guessing(tmp_path):
    db = SQLiteDB(tmp_path / "adapter.sqlite")
    populate(db)

    class Repo(DatabaseCharacterMemoryRepository):
        async def source_windows(self, *a, **kw):
            windows = await super().source_windows(*a, **kw)
            for window in windows:
                window.pop("following_rows")
            return windows

    result = await SourceMemoryService(Repo(db), max_chars=16384, defer_budget=True).recall("role", SCOPE, QUERY)
    assert not result.context and result.diagnostics["status"] == "retrieval_error"


async def test_long_following_condition_never_leaves_only_short_anchor(tmp_path):
    db = SQLiteDB(tmp_path / "budget.sqlite")
    put(db, "anchor", "VX642：原始记录，限制：禁用。", 0)
    tail = "完整补充。" * 200 + "限制：此任务彻底取消。"
    put(db, "following", tail, 1)
    result = await SourceMemoryService(DatabaseCharacterMemoryRepository(db), max_chars=400, defer_budget=True).recall(
        "role", SCOPE, QUERY
    )
    assert not result.context and result.diagnostics["status"] == "budget_omitted"
    records = json.loads(result.candidate_context)["records"]
    assert [r["source_id"] for r in records] == ["anchor", "following"] and records[1]["text"] == tail


async def test_missing_visible_successor_is_recorded_without_inventing_one(tmp_path):
    db = SQLiteDB(tmp_path / "missing.sqlite")
    put(db, "anchor", "VX642：原始记录，限制：禁用。")
    result = await SourceMemoryService(
        DatabaseCharacterMemoryRepository(db), max_chars=16384, defer_budget=True
    ).recall("role", SCOPE, QUERY)
    assert ids(result) == ["anchor"] and result.diagnostics["following_missing_for_anchors"] == ["anchor"]
