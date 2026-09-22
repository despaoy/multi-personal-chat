"""Event rules use frozen times and disposable databases, never user data."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.context_builder import build_user_scope
from character.event_memory import event_reference_content, parse_event
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, MemoryItem, RelationshipState
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService, PreparedCharacterTurn, TurnInput

NOW = datetime(2026, 9, 22, 16, 30, tzinfo=timezone.utc)  # Sep 23 in +08:00.


def scope(sender="alice", conversation="g1"):
    return build_user_scope(
        platform="qq", adapter="nonebot", sender_id=sender, conversation_id=conversation, conversation_type="group"
    )


@pytest.fixture
def repo(tmp_path):
    return DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "events.db"))


async def save(repo, text, sender="alice", source="m1"):
    count = 0
    for item in extract_memories(text, reference_time=NOW):
        count += await write_rule_memory(repo, "kisaki", scope(sender), item, source)
    return count


async def current(repo):
    return await repo.list_memory_records("kisaki", scope())


@pytest.mark.parametrize(
    ("text", "state", "scheduled"),
    [
        ("我明天要面试。", "planned", "2026-09-24"),
        ("后天我会去体检。", "planned", "2026-09-25"),
        ("我2026-10-03要参加会议。", "planned", "2026-10-03"),
        ("我正在准备毕业答辩。", "ongoing", ""),
        ("面试已经结束了。", "completed", ""),
        ("我完成了论文。", "completed", ""),
        ("面试取消了。", "cancelled", ""),
        ("面试推迟到明天了。", "postponed", "2026-09-24"),
        ("面试延期了。", "postponed", ""),
    ],
)
def test_explicit_event_extraction(text, state, scheduled):
    items = extract_memories(text, reference_time=NOW)
    assert len(items) == 1
    assert items[0].memory_type == "shared_event"  # A personal plan is not a promise to the character.
    assert items[0].event.state == state
    assert items[0].event.scheduled_date == scheduled
    assert items[0].evidence == text


@pytest.mark.parametrize(
    "text",
    [
        "面试结束了吗？",
        "面试还没结束。",
        "面试可能结束了。",
        "如果面试结束了。",
        "我朋友的面试结束了。",
        "那个结束了。",
        "不要记住我明天要面试。",
        "小说台词：我明天要面试。",
        "面试不是取消了。",
        "我以为面试结束了。",
    ],
)
def test_no_guessed_event_updates(text):
    assert not any(item.event for item in extract_memories(text, reference_time=NOW))


def test_invalid_dates_and_naive_time_are_not_interpreted():
    assert parse_event("我2026-02-30要面试。", NOW) is None
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_event("我明天要面试。", datetime(2026, 9, 22))


@pytest.mark.asyncio
async def test_completed_event_replaces_plan_with_evidence_and_history(repo):
    assert await save(repo, "我明天要面试。") == 1
    assert await save(repo, "面试已经结束了。", source="m2") == 1
    rows = await current(repo)
    assert len(rows) == 1 and "已完成" in rows[0]["content"]
    assert rows[0]["metadata"]["event"]["scheduled_date"] == "2026-09-24"
    assert rows[0]["evidence"] == ["面试已经结束了。"]
    assert rows[0]["source_message_ids"] == ["m2"]
    history = await repo.list_memory_records("kisaki", scope(), include_inactive=True)
    assert len(history) == 2
    assert {r["status"] for r in history} == {"superseded", "active"}
    assert await save(repo, "面试已经结束了。") == 0
    assert await save(repo, "我明天要面试。") == 0  # No implicit reopening.


@pytest.mark.asyncio
async def test_postpone_changes_date_and_unspecified_postpone_clears_old_date(repo):
    await save(repo, "我明天要面试。")
    await save(repo, "面试推迟到后天了。")
    assert (await current(repo))[0]["metadata"]["event"]["scheduled_date"] == "2026-09-25"
    await save(repo, "面试延期了。")
    assert (await current(repo))[0]["metadata"]["event"]["scheduled_date"] == ""
    await save(repo, "面试取消了。")
    assert "已取消" in (await current(repo))[0]["content"]


@pytest.mark.asyncio
async def test_ambiguous_or_unmatched_updates_change_nothing(repo):
    assert await save(repo, "面试结束了。") == 0
    await save(repo, "我明天要面试。")
    await save(repo, "我后天要面试。")
    assert len(await current(repo)) == 2
    assert await save(repo, "面试结束了。") == 0
    assert await save(repo, "面试取消了。", sender="bob") == 0
    assert all(r["metadata"]["event"]["state"] == "planned" for r in await current(repo))
    assert not await repo.list_memory_records("kisaki", scope(conversation="g2"))
    assert not await repo.list_memory_records("other", scope())


@pytest.mark.asyncio
async def test_unique_legacy_goal_can_close_but_relationship_notes_are_untouched(repo):
    for key in ("goal_面试", "relationship:shared_event:面试"):
        await repo.add_or_update_memory(
            "kisaki", scope(), MemoryItem("", "shared_event", "准备面试", 0.7), memory_key=key
        )
    assert await save(repo, "面试结束了。") == 1
    rows = await current(repo)
    assert len(rows) == 2
    assert next(r for r in rows if r["memory_key"] == "goal_面试")["metadata"]["event"]["state"] == "completed"
    assert next(r for r in rows if r["memory_key"].startswith("relationship:"))["content"] == "准备面试"


@pytest.mark.asyncio
async def test_same_message_plan_then_completion_is_ordered(repo):
    assert await save(repo, "我明天要面试。面试取消了。") == 2
    assert "已取消" in (await current(repo))[0]["content"]


@pytest.mark.asyncio
async def test_overdue_is_unknown_not_completed_and_does_not_write(repo):
    await save(repo, "我明天要面试。")
    row = (await current(repo))[0]
    later = datetime(2026, 9, 25, tzinfo=timezone.utc)
    assert "结果未知" in event_reference_content(row, later)
    assert "结果未知" not in event_reference_content(row, NOW)
    assert (await current(repo))[0]["metadata"]["event"]["state"] == "planned"
    await save(repo, "面试结束了。")
    assert "已完成" in event_reference_content((await current(repo))[0], later)


@pytest.mark.asyncio
async def test_actual_retrieval_labels_overdue_and_keeps_completion(repo):
    await save(repo, "我明天要面试。")
    service = CharacterMemoryService(repo, semantic_enabled=False)
    later = datetime(2026, 10, 1, tzinfo=timezone.utc)
    items, _ = await service.load_relevant_memories("kisaki", scope(), "我的面试", reference_time=later)
    assert len(items) == 1 and "结果未知" in items[0].content
    await save(repo, "面试结束了。")
    items, _ = await service.load_relevant_memories("kisaki", scope(), "我的面试", reference_time=later)
    assert len(items) == 1 and "已完成" in items[0].content
    assert "结果未知" not in items[0].content
    history = await repo.list_memory_records("kisaki", scope(), include_inactive=True)
    old = next(row for row in history if row["status"] == "superseded")
    assert "结果未知" not in event_reference_content(old, later)


@pytest.mark.asyncio
async def test_old_date_after_reschedule_is_not_silently_reopened(repo):
    await save(repo, "我明天要面试。")
    await save(repo, "面试推迟到后天了。")
    assert await save(repo, "我明天要面试。") == 0
    rows = await current(repo)
    assert len(rows) == 1
    assert rows[0]["metadata"]["event"]["scheduled_date"] == "2026-09-25"


@pytest.mark.asyncio
async def test_chat_completion_routes_event_without_scheduler(repo, monkeypatch):
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", lambda: SimpleNamespace(enabled=False))
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
    for text in ("我明天要面试。", "面试取消了。"):
        outcome = await service.complete_turn(
            prepared,
            TurnInput(text, "qq", "nonebot", "alice", "g1", "group"),
            "祝面试成功",
            source_message_id=text,
        )
        assert outcome.new_memories == 1
        assert outcome.memory_enrichment_mode == "rules"
    assert "已取消" in (await current(repo))[0]["content"]
