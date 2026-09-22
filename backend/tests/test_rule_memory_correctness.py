"""M1 regression probes; all persistence tests use disposable SQLite databases."""

import asyncio
from types import SimpleNamespace

import pytest

from character.context_builder import build_user_scope
from character.memory_extractor import extract_memories, extract_preferred_address
from character.models import CompiledCharacterContext, MemoryItem, RelationshipState
from character.natural_relationship import relationship_write_blocked
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService, PreparedCharacterTurn, TurnInput


def scope(sender="alice", conversation="g1"):
    return build_user_scope(
        platform="qq", adapter="nonebot", sender_id=sender, conversation_id=conversation, conversation_type="group"
    )


@pytest.fixture
def repo(tmp_path):
    return DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "rules.db"))


async def save(repo, text, sender="alice", source="m1"):
    for item in extract_memories(text):
        await write_rule_memory(repo, "kisaki", scope(sender), item, source)


@pytest.mark.parametrize(
    "text",
    [
        "我叫小明？",
        "我住在上海？",
        "叫我小明？",
        "我妈说我喜欢咖啡。",
        "我妈妈以为我喜欢咖啡。",
        "不是我喜欢咖啡。",
        "我叫小明，但不是我的真名。",
        "假设我喜欢咖啡。",
        "小说台词：我叫小明。",
        "我在小说中说我喜欢咖啡。",
        "我以前喜欢咖啡，现在不喝了。",
        "今天我喜欢咖啡。",
    ],
)
def test_non_assertions_never_become_facts(text):
    assert not extract_memories(text)
    assert extract_preferred_address(text) is None


def test_hobby_is_not_fiction_and_questions_do_not_erase_other_sentences():
    assert not relationship_write_blocked("我喜欢读小说。")
    assert extract_memories("我喜欢读小说。")[0].memory_key == "preference_读小说"
    assert [m.memory_key for m in extract_memories("我叫小明？我喜欢咖啡。")] == ["preference_咖啡"]


def test_qualifiers_and_exact_source_are_preserved():
    text = "我喜欢咖啡，但晚上不喝。"
    item = extract_memories(text)[0]
    assert "晚上不喝" in item.content and item.evidence == text
    assert dict(item.qualifiers)["context"] == text
    assert item.memory_key == "preference_咖啡"
    assert not extract_memories("我喜欢咖啡，" + "非常" * 100 + "但晚上不喝。")
    assert extract_memories("我喜欢咖啡但晚上不喝。")[0].memory_key == "preference_咖啡"
    repeated = extract_memories("我喜欢咖啡，但晚上不喝。我喜欢喝咖啡。")
    assert len(repeated) == 1 and "晚上不喝" in repeated[0].content


def test_only_allowlisted_objects_are_normalized():
    assert extract_memories("我喜欢喝咖啡。")[0].memory_key == "preference_咖啡"
    assert extract_memories("我喜欢吃苦。")[0].memory_key == "preference_吃苦"
    assert "明天" in extract_memories("明天我会去面试。")[0].content


@pytest.mark.asyncio
async def test_origin_and_residence_do_not_overwrite_each_other(repo):
    await save(repo, "我来自成都。我现在住在上海。")
    rows = await repo.list_memory_records("kisaki", scope())
    assert {r["memory_key"] for r in rows} == {"user_origin", "user_residence"}
    assert all(r["evidence"] and r["source_message_ids"] == ["m1"] for r in rows)
    await save(repo, "我现在住在南京。", source="m2")
    rows = await repo.list_memory_records("kisaki", scope())
    assert any("成都" in r["content"] for r in rows)
    assert any("南京" in r["content"] for r in rows)
    history = await repo.list_memory_records("kisaki", scope(), include_inactive=True)
    assert next(r for r in history if "上海" in r["content"])["status"] == "superseded"


@pytest.mark.asyncio
async def test_preference_changes_are_versioned_not_appended_as_two_current_facts(repo):
    await save(repo, "我喜欢喝咖啡。")
    await save(repo, "我不喜欢咖啡。", source="m2")
    current = await repo.list_memory_records("kisaki", scope())
    assert len(current) == 1 and "不喜欢" in current[0]["content"]
    history = await repo.list_memory_records("kisaki", scope(), include_inactive=True)
    assert len(history) == 2 and {r["status"] for r in history} == {"active", "superseded"}
    await save(repo, "我不喜欢咖啡。", source="m3")
    assert len(await repo.list_memory_records("kisaki", scope(), include_inactive=True)) == 2


@pytest.mark.asyncio
async def test_conditions_survive_repetition_and_uncertain_changes_are_pending(repo):
    await save(repo, "我喜欢咖啡，但晚上不喝。")
    await save(repo, "我喜欢喝咖啡。", source="m2")
    assert "晚上不喝" in (await repo.list_memory_records("kisaki", scope()))[0]["content"]
    await save(repo, "我喜欢咖啡，但不加糖。", source="m3")
    rows = await repo.list_memory_records("kisaki", scope(), include_inactive=True)
    assert len(rows) == 2
    assert (
        next(r for r in rows if r["status"] == "pending")["metadata"]["review_reason"]
        == "conditional_change_requires_review"
    )
    await save(repo, "我不喜欢咖啡。", source="m4")
    assert "不喜欢" in (await repo.list_memory_records("kisaki", scope()))[0]["content"]


@pytest.mark.asyncio
async def test_legacy_revision_zero_is_superseded_and_other_scopes_unchanged(repo):
    for sender in ("alice", "bob"):
        await repo.add_or_update_memory(
            "kisaki",
            scope(sender),
            MemoryItem("", "user_fact", "用户说喜欢喝咖啡", 0.6),
            memory_key="preference_喝咖啡",
        )
    await save(repo, "我不喜欢咖啡。")
    assert "不喜欢" in (await repo.list_memory_records("kisaki", scope()))[0]["content"]
    assert "喜欢喝" in (await repo.list_memory_records("kisaki", scope("bob")))[0]["content"]
    assert not await repo.list_memory_records("kisaki", scope(conversation="g2"))
    assert not await repo.list_memory_records("other", scope())


@pytest.mark.asyncio
async def test_concurrent_rules_leave_one_current_version(repo):
    await asyncio.gather(save(repo, "我喜欢咖啡。", source="a"), save(repo, "我不喜欢咖啡。", source="b"))
    assert len(await repo.list_memory_records("kisaki", scope())) == 1
    assert len(await repo.list_memory_records("kisaki", scope(), include_inactive=True)) == 2


@pytest.mark.asyncio
async def test_stale_update_cannot_overwrite_a_newer_version(repo):
    await save(repo, "我喜欢咖啡。")
    old = (await repo.list_memory_records("kisaki", scope()))[0]
    await save(repo, "我不喜欢咖啡。")
    with pytest.raises(ValueError, match="changed concurrently"):
        await repo.append_claim(
            "kisaki",
            scope(),
            MemoryItem("", "user_fact", "stale", 0.5),
            memory_key=old["memory_key"],
            relation_type="SUPERSEDE",
            supersedes_memory_id=old["id"],
            metadata={"origin": "rule_v2"},
        )
    assert len(await repo.list_memory_records("kisaki", scope())) == 1


@pytest.mark.asyncio
async def test_ambiguous_legacy_aliases_remain_untouched_and_pending_deduplicates(repo):
    for key in ("preference_咖啡", "preference_喝咖啡"):
        await repo.add_or_update_memory(
            "kisaki",
            scope(),
            MemoryItem("", "user_fact", "旧偏好", 0.6),
            memory_key=key,
        )
    await save(repo, "我不喜欢咖啡。")
    await save(repo, "我不喜欢咖啡。", source="m2")
    rows = await repo.list_memory_records("kisaki", scope(), include_inactive=True)
    assert len(rows) == 3
    assert len([row for row in rows if row["status"] == "active"]) == 2
    candidate = next(row for row in rows if row["status"] == "pending")
    assert candidate["metadata"]["review_reason"] == "ambiguous_legacy_aliases"


@pytest.mark.asyncio
async def test_chat_completion_persists_user_evidence_without_model_calls(repo, monkeypatch):
    monkeypatch.setattr(
        "character.memory_llm.get_memory_enrichment_scheduler",
        lambda: SimpleNamespace(enabled=False),
    )
    service = CharacterContextService(object(), repo, object())
    prepared = PreparedCharacterTurn(
        character_id="kisaki",
        user_scope=scope(),
        compiled=CompiledCharacterContext("", "", "", ()),
        history=(),
        relationship=RelationshipState(),
        memory_candidates=0,
        interaction_count=0,
        reply_guard=None,
    )
    for message_id, message in (("m1", "我喜欢喝咖啡。"), ("m2", "我不喜欢咖啡。")):
        outcome = await service.complete_turn(
            prepared,
            TurnInput(message, "qq", "nonebot", "alice", "g1", "group"),
            "我猜用户住在北京。",
            source_message_id=message_id,
        )
        assert outcome.new_memories == 1
        assert outcome.memory_enrichment_mode == "rules"
        assert not outcome.memory_enrichment_scheduled
    rows = await repo.list_memory_records("kisaki", scope(), include_inactive=True)
    assert len(rows) == 2
    current = next(row for row in rows if row["status"] == "active")
    assert current["evidence"] == ["我不喜欢咖啡。"]
    assert current["source_message_ids"] == ["m2"]
    assert all("北京" not in str(row) for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["rule_v2", "rule_candidate"])
async def test_old_adapters_cannot_silently_discard_evidence_or_pending_status(origin):
    repo = DatabaseCharacterMemoryRepository(object())
    with pytest.raises(RuntimeError, match="版本与证据"):
        await repo.append_claim(
            "kisaki",
            scope(),
            MemoryItem("", "user_fact", "候选", 0.6),
            memory_key="preference_咖啡",
            relation_type="PENDING",
            metadata={"origin": origin},
        )
