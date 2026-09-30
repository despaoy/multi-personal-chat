"""Preserve conditional speech without promoting speculation into facts."""
from datetime import datetime, timedelta, timezone

import pytest

from character.memory_extractor import extract_memories, fictional_memory_context
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from character.natural_relationship import hypothetical_source_only, relationship_write_blocked
from character.profile_registry import CharacterProfileRegistry
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput


@pytest.mark.parametrize("text", [
    "如果获得批准，我才会去参加交流活动。现在还没通过。",
    "要是项目结束，我可能搬到外地。",
    "假如下个月有空，我就学木工。",
    "假设：我毕业后换一份工作。",
    "如果我是世界首富，我就买一座岛。",
])
def test_hypothetical_source_does_not_relax_fact_grammar(text):
    assert fictional_memory_context(text) and relationship_write_blocked(text)
    assert hypothetical_source_only(text)
    assert not extract_memories(text)


@pytest.mark.parametrize("text", [
    "如果小说中的主人公离开学校，他会搬到海边。",
    "如果我说‘我搬家了’，那只是台词。",
    "假如这是角色扮演，我喜欢咖啡。",
    "例如我叫李明。", "假装我是医生。",
    "如果你同意，交流偏好：叫我大王。",
    "我喜欢散步。",
])
def test_explicit_fiction_notes_and_ordinary_assertions_keep_separate_paths(text):
    assert not hypothetical_source_only(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,captured", [
    ("如果获得批准，我才会去参加交流活动。现在还没通过。", True),
    ("假如以后搬家，我想住得离单位近一些。", True),
    ("要是我有这笔钱，我会买一艘船。", True),
    ("如果拿到通知我才会去，但是不要记住这件事。", False),
    ("如果登录成功，我会修改密码。", False),
    ("如果小说中的人物搬家，那只是故事里的安排。", False),
])
async def test_actual_complete_stores_source_without_model_or_fact_writes(tmp_path, monkeypatch, text, captured):
    import character.memory_llm as memory_llm

    class NoModel:
        async def complete(self, messages):
            raise AssertionError("Source-only retention must not invoke a model")

        async def close(self):
            pass

    database = SQLiteDB(tmp_path / "hypothetical.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    registry = CharacterProfileRegistry()
    registry.load_profiles()
    service = CharacterContextService(registry, repo, DatabaseMessageRepository(database))
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "unused"), completion=NoModel())
    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: scheduler)
    turn = TurnInput(text, "web", "test", "u", "u", "private")
    prepared = await service.prepare_turn(turn, "tsukiyashiro_kisaki")
    try:
        await service.complete_turn(prepared, turn, "明白。", source_message_id="actual-input")
        assert await scheduler.flush_memory(timeout=3)
        rows = await repo.list_sources(prepared.character_id, prepared.user_scope)
        assert [row["body"] for row in rows] == ([text] if captured else [])
        assert await repo.list_memory_records(prepared.character_id, prepared.user_scope) == []
        assert scheduler.status.failed == 0
        if captured:
            assert scheduler.status.recent_results[-1]["status"] == "source_only"
    finally:
        await scheduler.shutdown(timeout=3)


@pytest.mark.asyncio
async def test_source_only_cannot_cross_erasure_fence(tmp_path):
    database = SQLiteDB(tmp_path / "fence.sqlite")
    fields = dict(character_id="role", platform="test", adapter="test", sender_id="u",
                  conversation_type="private", conversation_id="u")
    database.clear_character_memories(**fields)

    class NoModel:
        async def complete(self, messages):
            raise AssertionError("No model call allowed")

        async def close(self):
            pass

    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "unused"), completion=NoModel())
    try:
        assert scheduler.schedule(repository=DatabaseCharacterMemoryRepository(database), character_id="role",
            user_scope=UserScope("test", "test", "u", "u", "private"),
            message="如果通过申请，我会搬家。", rule_hints=[], source_message_id="late", source_only=True,
            observed_at=datetime.now(timezone.utc) - timedelta(seconds=10))
        assert await scheduler.flush_memory(timeout=3)
        assert database.list_memory_sources(**fields) == []
        assert scheduler.status.recent_results[-1]["reason"] == "source_stale"
    finally:
        await scheduler.shutdown(timeout=3)
