"""Current field authority comes from each claim's complete authorized source."""

from datetime import datetime, timedelta, timezone

import pytest

from character.context_builder import compile_reference_context
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, MemoryItem, UserScope
from db.database import SQLiteDB
from inference.memory_response import read_memory_fields
from repositories.character_memory import DatabaseCharacterMemoryRepository

SCOPE = UserScope("web", "test", "alice", "alice", "private")
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
PROFILE = "请跨会话记住，我叫岚舟，我的老家在宣城，目前住在丽水，大学专业是环境工程。这四项资料都是我本人的。"
FIELDS = {
    "user_name": ("name", "我的姓名是什么？"),
    "user_residence": ("residence", "我的居住地是什么？"),
    "user_origin": ("origin", "我的籍贯是什么？"),
    "user_major": ("major", "我的专业是什么？"),
    "user_workplace": ("workplace", "我的工作单位是什么？"),
}


async def seed(repo, key, quote, *, source=None, source_id="profile", source_ids=()):
    if source is not None:
        await repo.capture_source("kisaki", SCOPE, source_message_id=source_id, body=source, observed_at=NOW)
    return await repo.append_claim(
        "kisaki",
        SCOPE,
        MemoryItem("", "user_fact", "模型任意摘要，不提供字段权限", 0.9),
        memory_key=key,
        evidence=(quote,),
        confidence=0.98,
        source_message_id=source_id,
        source_message_ids=source_ids,
        observed_at=NOW.isoformat(),
        valid_from=NOW.isoformat(),
        valid_to=NOW.isoformat(),
        metadata={
            "temporal_provenance": {"version": 1, "producer": "semantic_memory", "validity_authority": "unverified"}
        },
    )


async def read(repo, key):
    field, query = FIELDS[key]
    items, _, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
        "kisaki", SCOPE, query, reference_time=NOW + timedelta(seconds=1)
    )
    context = CompiledCharacterContext(
        "",
        "",
        "",
        tuple(i.memory_id for i in items),
        memory_status="available",
        memory_packets=items,
        memory_field_presence=tuple(trace["field_presence"].items()),
    )
    (result,) = read_memory_fields(query, context)
    return field, items, trace, result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,source,quote,value",
    [
        ("user_origin", "我的老家在宣城。", "我的老家在宣城", "宣城"),
        ("user_origin", "我的故乡是崇左。", "我的故乡是崇左", "崇左"),
        ("user_origin", "我家乡是绍兴。", "我家乡是绍兴", "绍兴"),
        ("user_major", "我的大学专业是环境工程。", "我的大学专业是环境工程", "环境工程"),
        ("user_major", PROFILE, "大学专业是环境工程", "环境工程"),
        ("user_origin", PROFILE, "我的老家在宣城", "宣城"),
        ("user_name", PROFILE, "我叫岚舟", "岚舟"),
        ("user_major", "我叫岚舟，来自宣城，住在丽水，专业是环境工程。", "专业是环境工程", "环境工程"),
        (
            "user_workplace",
            "我叫岚舟，我来自宣城，我住在丽水，我的专业是环境工程，我是大二学生，我在观测站工作。",
            "我在观测站工作",
            "观测站",
        ),
        ("user_major", "我叫岚舟。我的大学专业是环境工程。", "我的大学专业是环境工程", "环境工程"),
    ],
)
async def test_complete_self_source_recovers_only_the_recorded_field_without_storage_rewrite(
    tmp_path, key, source, quote, value
):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "complete.sqlite"))
    claim = await seed(repo, key, quote, source=source)
    before = await repo.list_memory_records("kisaki", SCOPE, include_inactive=True)
    field, items, trace, result = await read(repo, key)
    assert trace["field_presence"][field] is True
    assert result.status == "known" and result.value == value
    assert len(items) == 1 and items[0].memory_key == key and items[0].memory_id == str(claim["id"])
    compiled, used = compile_reference_context(
        items, complete_evidence=True, observation_semantics=True, max_chars=6000
    )
    assert str(claim["id"]) in used and source in compiled
    assert await repo.list_memory_records("kisaki", SCOPE, include_inactive=True) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,quote",
    [
        (None, "大学专业是环境工程"),
        ("假设我的大学专业是环境工程，但这只是虚构背景。", "我的大学专业是环境工程"),
        ("我朋友说：“我的大学专业是环境工程。”这是朋友的资料。", "我的大学专业是环境工程"),
        ("我叫岚舟，大学专业是环境工程。这些资料其实是朋友的。", "大学专业是环境工程"),
        ("我叫岚舟，大学专业是环境工程。上面那些资料并非事实。", "大学专业是环境工程"),
        ("我叫岚舟，大学专业是环境工程，但下月可能换专业。", "大学专业是环境工程"),
        ("我叫岚舟，大学专业是环境工程，我还没决定是否采用这个设定。", "大学专业是环境工程"),
        ("我叫岚舟，我的大学专业是环境工程。我的大学专业是海洋工程。", "我的大学专业是环境工程"),
        (PROFILE, "我叫岚舟"),
    ],
)
async def test_clipped_quote_cannot_drop_complete_owner_retraction_or_conflict(tmp_path, source, quote):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "uncertain.sqlite"))
    await seed(repo, "user_major", quote, source=source)
    field, items, trace, result = await read(repo, "user_major")
    assert trace["field_presence"][field] is None
    assert result.status == "unverified" and result.value is None
    assert items and all(i.temporal_mode == "observation" for i in items)
    if source is not None:
        assert source in items[0].content


@pytest.mark.asyncio
async def test_batch_source_reader_keeps_actual_claim_association(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "association.sqlite"))
    actual = await seed(repo, "user_name", "我叫岚舟", source=PROFILE)
    forged = await seed(repo, "user_major", "大学专业是环境工程", source_id="missing", source_ids=("profile",))
    linked = await repo.linked_sources("kisaki", SCOPE, memory_ids=(actual["id"], forged["id"]))
    assert linked and {r["memory_id"] for r in linked} == {actual["id"]}
    _, _, trace, result = await read(repo, "user_major")
    assert trace["field_presence"]["major"] is None and result.status == "unverified"


@pytest.mark.asyncio
async def test_other_owner_source_with_same_id_is_not_authority(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "owner.sqlite"))
    await seed(repo, "user_major", "大学专业是环境工程")
    await repo.capture_source(
        "kisaki",
        UserScope("web", "test", "bob", "bob", "private"),
        source_message_id="profile",
        body=PROFILE,
        observed_at=NOW,
    )
    _, _, trace, result = await read(repo, "user_major")
    assert trace["field_presence"]["major"] is None and result.status == "unverified"


@pytest.mark.asyncio
async def test_complete_source_reader_failure_is_visible_and_never_promotes_a_quote(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "failure.sqlite"))
    await seed(repo, "user_major", "大学专业是环境工程", source=PROFILE)

    async def unavailable(*args, **kwargs):
        raise RuntimeError("source read unavailable")

    repo.linked_source_receipts = unavailable
    with pytest.raises(RuntimeError, match="source read unavailable"):
        await read(repo, "user_major")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,quote,value",
    [
        ("user_residence", "我目前住在舟山，已经不住丽水了", "舟山"),
        ("user_origin", "老家仍是宣城", "宣城"),
        ("user_major", "大学专业仍是环境工程", "环境工程"),
    ],
)
async def test_complete_source_understands_explicit_update_and_unchanged_self_fields(tmp_path, key, quote, value):
    source = "请跨会话记住，我目前住在舟山，已经不住丽水了。老家仍是宣城，大学专业仍是环境工程。只更新目前居住地。"
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "controls.sqlite"))
    await seed(repo, key, quote, source=source)
    field, items, trace, result = await read(repo, key)
    assert trace["field_presence"][field] is True and result.status == "known" and result.value == value
    assert items and source in items[0].evidence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "老家仍是宣城，大学专业仍是环境工程。只更新目前居住地。",
        "我目前住在舟山。我朋友的大学专业仍是环境工程。只更新专业。",
    ],
)
async def test_source_control_does_not_invent_an_owner_or_cross_a_subject_reset(tmp_path, source):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "controls-owner.sqlite"))
    await seed(repo, "user_major", "大学专业仍是环境工程", source=source)
    _, _, trace, result = await read(repo, "user_major")
    assert trace["field_presence"]["major"] is None and result.status == "unverified"


@pytest.mark.asyncio
async def test_legacy_citation_cannot_borrow_a_source_linked_only_to_another_claim(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "legacy-association.sqlite"))
    await seed(repo, "user_name", "我叫岚舟", source="我叫岚舟。我的大学专业是环境工程。")
    await repo.append_claim(
        "kisaki",
        SCOPE,
        MemoryItem("", "user_fact", "用户明确提到：我的大学专业是环境工程", 0.9),
        memory_key="fact_环境工程",
        evidence=("我的大学专业是环境工程",),
        confidence=0.98,
        source_message_ids=("profile",),
        observed_at=NOW.isoformat(),
    )
    _, _, trace, result = await read(repo, "user_major")
    assert trace["field_presence"]["major"] is None and result.status == "unverified"
