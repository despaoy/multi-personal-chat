"""Legacy field reads require the exact authorized, complete source."""

from datetime import datetime, timedelta, timezone

import pytest

from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, MemoryItem, UserScope
from db.database import SQLiteDB
from inference.memory_response import read_memory_fields, render_memory_response
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("web", "test", "alice", "alice", "private")
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


async def seed(repo, value, quote, *, source=None, source_id="original", **kwargs):
    if source is not None:
        await repo.capture_source("kisaki", SCOPE, source_message_id=source_id, body=source, observed_at=NOW)
    return await repo.append_claim(
        "kisaki",
        SCOPE,
        MemoryItem("", "user_fact", "用户明确提到：" + quote, 0.9),
        memory_key="fact_" + value,
        evidence=(quote,),
        confidence=0.98,
        source_message_id=source_id,
        **kwargs,
    )


async def authorize_legacy_sources(repo, claim, expected):
    """Seed trusted legacy links; model citation IDs alone never grant access."""
    from db import memory_source

    scope = memory_source.source_scope(
        "kisaki", SCOPE.platform, SCOPE.adapter, SCOPE.sender_id, SCOPE.conversation_type, SCOPE.conversation_id
    )
    connection = repo._database.get_connection()
    with connection:
        cursor = connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        memory_source.run_sqlite(cursor, memory_source.lock_owner(scope))
        for source_id in expected:
            identity = memory_source.source_identity(scope, source_id)
            memory_source.run_sqlite(cursor, memory_source.link_plan(identity, claim["id"]))
    linked = await repo.linked_sources("kisaki", SCOPE, memory_ids=(claim["id"],))
    assert {row["source_message_id"]: row["body"] for row in linked} == expected


async def read(repo, query, *, scope=SCOPE, **kwargs):
    items, _, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        "kisaki",
        scope,
        query,
        reference_time=NOW + timedelta(seconds=1),
        **kwargs,
    )
    assert trace["status"] != "retrieval_error"
    context = CompiledCharacterContext(
        "",
        "",
        "参考",
        tuple(item.memory_id for item in items),
        memory_status="available",
        memory_packets=items,
        memory_field_presence=tuple(trace["field_presence"].items()),
    )
    return items, trace, context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,source,field,query",
    [
        ("岚舟", "我叫岚舟。", "name", "我的名字是什么？"),
        ("丽水", "我目前住在丽水。", "residence", "我的居住地是什么？"),
        ("宣城", "我来自宣城。", "origin", "我的籍贯是什么？"),
    ],
)
async def test_complete_legacy_source_is_read_as_the_proven_field_without_storage_rewrite(
    tmp_path, value, source, field, query
):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "legacy.sqlite"))
    record = await seed(repo, value, source, source=source)
    before = await repo.list_memory_records("kisaki", SCOPE)
    items, trace, context = await read(repo, query)
    assert trace["field_presence"][field] is True
    (result,) = read_memory_fields(query, context)
    assert result.status == "known" and result.value == value
    assert items[0].memory_id == str(record["id"])
    assert items[0].source_message_ids == ("original",)
    assert await repo.list_memory_records("kisaki", SCOPE) == before


@pytest.mark.asyncio
async def test_full_multifield_source_does_not_promote_collateral_fields(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "multi.sqlite"))
    source = "我叫岚舟，我来自宣城，我目前住在丽水，我的专业是环境工程。"
    await seed(repo, "岚舟", "我叫岚舟", source=source)
    items, trace, _ = await read(repo, "我的名字和居住地是什么？")
    assert trace["field_presence"]["name"] is True
    assert trace["field_presence"]["residence"] is False
    assert len(items) == 1 and items[0].memory_key == "user_name"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        None,
        "假设我目前住在丽水，那只是小说里的设想，并非我的真实住址。",
        "我朋友说：“我目前住在丽水。”这是朋友的住址，不是我的。",
    ],
)
async def test_missing_or_contradicting_complete_source_is_unverified_not_absent_or_current(tmp_path, source):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "unverified.sqlite"))
    await seed(repo, "丽水", "我目前住在丽水", source=source)
    items, trace, context = await read(repo, "我的居住地是什么？")
    assert trace["field_presence"]["residence"] is None
    (result,) = read_memory_fields("我的居住地是什么？", context)
    assert result.status == "unverified" and result.value is None
    assert "没有你的" not in (render_memory_response("你保存了我的住址吗？", context) or "")
    assert items and all(item.memory_key != "user_residence" for item in items)
    if source is not None:
        assert source in items[0].content


@pytest.mark.asyncio
async def test_generic_supersession_and_historical_window_keep_the_original_version_chain(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "versions.sqlite"))
    old = await seed(
        repo, "丽水", "我目前住在丽水。", source="我目前住在丽水。", valid_from="2026-09-01T00:00:00+00:00"
    )
    source = "我现在住在舟山。"
    await repo.capture_source("kisaki", SCOPE, source_message_id="update", body=source, observed_at=NOW)
    await repo.append_claim(
        "kisaki",
        SCOPE,
        MemoryItem("", "user_fact", "用户明确提到：" + source, 0.9),
        memory_key="fact_丽水",
        relation_type="SUPERSEDE",
        supersedes_memory_id=old["id"],
        evidence=(source,),
        confidence=0.98,
        source_message_id="update",
        valid_from=NOW.isoformat(),
    )
    _, trace, context = await read(repo, "我的居住地是什么？")
    assert trace["field_presence"]["residence"] is True
    (result,) = read_memory_fields("我的居住地是什么？", context)
    assert result.status == "known" and result.value == "舟山"
    historical, _, _ = await read(repo, "我2026年9月的居住地是什么？", include_historical=True)
    assert any(item.historical and "丽水" in item.content for item in historical)
    records = await repo.list_memory_records("kisaki", SCOPE, include_inactive=True)
    assert {r["memory_key"] for r in records} == {"fact_丽水"}
    assert {r["status"] for r in records} == {"active", "superseded"}


@pytest.mark.asyncio
async def test_foreign_owner_source_cannot_complete_a_legacy_claim_with_the_same_source_id(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "owner.sqlite"))
    await seed(repo, "岚舟", "我叫岚舟。")
    other = UserScope("web", "test", "bob", "bob", "private")
    await repo.capture_source("kisaki", other, source_message_id="original", body="我叫岚舟。", observed_at=NOW)
    items, trace, context = await read(repo, "我的名字是什么？")
    assert trace["field_presence"]["name"] is None
    assert items and items[0].memory_key == "fact_岚舟"
    (result,) = read_memory_fields("我的名字是什么？", context)
    assert result.status == "unverified"


@pytest.mark.asyncio
async def test_legacy_projection_never_erases_an_explicit_future_validity_boundary(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "future.sqlite"))
    await seed(repo, "丽水", "我目前住在丽水。", source="我目前住在丽水。", valid_from="2027-01-01T00:00:00+00:00")
    items, trace, _ = await read(repo, "我的居住地是什么？")
    assert not items and trace["field_presence"]["residence"] is False


@pytest.mark.asyncio
async def test_conflicting_linked_sources_cannot_certify_one_current_value(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "conflict.sqlite"))
    await repo.capture_source("kisaki", SCOPE, source_message_id="one", body="我叫岚舟。", observed_at=NOW)
    await repo.capture_source("kisaki", SCOPE, source_message_id="two", body="我叫若澄。", observed_at=NOW)
    claim = await repo.append_claim(
        "kisaki",
        SCOPE,
        MemoryItem("", "user_fact", "用户明确提到：我叫岚舟。", 0.9),
        memory_key="fact_岚舟",
        evidence=("我叫岚舟。", "我叫若澄。"),
        source_message_id="one",
        source_message_ids=("one", "two"),
        confidence=0.98,
    )
    await authorize_legacy_sources(repo, claim, {"one": "我叫岚舟。", "two": "我叫若澄。"})
    items, trace, context = await read(repo, "我的名字是什么？")
    assert trace["field_presence"]["name"] is None
    assert items and "若澄" in items[0].content and "岚舟" in items[0].content
    (result,) = read_memory_fields("我的名字是什么？", context)
    assert result.status == "unverified"


@pytest.mark.asyncio
async def test_temporal_observation_keeps_complete_source_when_the_quote_was_clipped(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "observation.sqlite"))
    source = "假设我目前住在丽水，那只是小说里的设想，并非我的真实住址。"
    await seed(
        repo,
        "丽水",
        "我目前住在丽水",
        source=source,
        observed_at=NOW.isoformat(),
        metadata={
            "temporal_provenance": {"version": 1, "producer": "semantic_memory", "validity_authority": "unspecified"}
        },
    )
    items, trace, context = await read(repo, "我的居住地是什么？")
    assert trace["field_presence"]["residence"] is None
    assert items and source in items[0].content and source in items[0].evidence
    assert items[0].source_observation and items[0].temporal_mode == "observation"
    assert read_memory_fields("我的居住地是什么？", context)[0].value is None


@pytest.mark.asyncio
async def test_linked_source_failure_stops_partial_field_recall(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "failure.sqlite"))
    await seed(repo, "岚舟", "我叫岚舟。", source="我叫岚舟。")
    await repo.append_claim(
        "kisaki",
        SCOPE,
        MemoryItem("", "user_fact", "用户说自己居住在舟山", 0.9),
        memory_key="user_residence",
        evidence=("我目前住在舟山。",),
        source_message_id="typed",
        confidence=0.98,
    )

    async def unavailable(*args, **kwargs):
        raise RuntimeError("source reader unavailable")

    repo.linked_source_receipts = unavailable
    with pytest.raises(RuntimeError, match="source reader unavailable"):
        await read(repo, "我的姓名和居住地是什么？")


@pytest.mark.asyncio
async def test_uncertain_legacy_source_reaches_compiler_with_its_trusted_receipt(tmp_path):
    from character.context_builder import compile_reference_context

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "receipt.sqlite"))
    source = "假设我目前住在丽水，那只是小说里的设想，并非我的真实住址。"
    claim = await seed(repo, "丽水", "我目前住在丽水", source=source)
    items, trace, _ = await read(repo, "我的居住地是什么？")
    assert items and items[0].observed_at == NOW.isoformat()
    text, used = compile_reference_context(items, complete_evidence=True, max_chars=6000, observation_semantics=True)
    assert used == (str(claim["id"]),) and source in text
    assert '"content_semantics":"quoted_source"' in text
    assert trace["field_presence"]["residence"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_source,confirmed",
    [
        ("我叫岚舟。", True),
        ("更正：我以前说的‘我叫岚舟’只是小说角色台词，不是我的真实姓名。", False),
    ],
)
async def test_every_linked_source_must_support_the_same_current_field(tmp_path, second_source, confirmed):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "every-source.sqlite"))
    for source_id, body in (("first", "我叫岚舟。"), ("second", second_source)):
        await repo.capture_source("kisaki", SCOPE, source_message_id=source_id, body=body, observed_at=NOW)
    claim = await repo.append_claim(
        "kisaki",
        SCOPE,
        MemoryItem("", "user_fact", "用户明确提到：我叫岚舟", 0.9),
        memory_key="fact_岚舟",
        evidence=("我叫岚舟",),
        confidence=0.98,
        source_message_id="first",
        source_message_ids=("first", "second"),
    )
    await authorize_legacy_sources(repo, claim, {"first": "我叫岚舟。", "second": second_source})
    items, trace, context = await read(repo, "我的名字是什么？")
    assert trace["field_presence"]["name"] is (True if confirmed else None)
    (result,) = read_memory_fields("我的名字是什么？", context)
    assert result.status == ("known" if confirmed else "unverified")
    if not confirmed:
        assert items and second_source in items[0].content
