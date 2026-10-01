"""Complete non-fiction quoted speech survives without factual promotion."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from character.memory_extractor import extract_memories
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.natural_relationship import quoted_source_only, relationship_write_blocked
from character.profile_registry import CharacterProfileRegistry
from character.source_memory import SourceMemoryService
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

FIXTURE = json.loads((Path(__file__).parent / "fixtures/deepseek_memory_quoted_source_cases.json").read_text())
SOURCE = FIXTURE["new_native_source"]


@pytest.mark.parametrize(
    "text",
    [
        SOURCE,
        "朋友说：“我喜欢茶。”我只是在转述他的原话。",
        "我今天收到课程确认。历史引用是「我喜欢茶」。",
        "我今天收到课程确认。历史引用是『我喜欢茶』。",
        "我今天收到课程确认。历史引用是‘我喜欢茶’。",
        '我今天收到课程确认。历史引用是"我喜欢茶"。',
        "我今天收到课程确认。他曾说：“她说『我喜欢茶』。”",
        '我今天收到课程确认。历史引用是"我写了\\"备注\\""。',
    ],
)
def test_balanced_nonfiction_reference_can_be_original_source_without_becoming_a_fact(text):
    assert quoted_source_only(text) and relationship_write_blocked(text)
    assert not extract_memories(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "我喜欢茶。",
        "“我喜欢茶。”",
        "“我喜欢茶。”？！",
        "我收到确认。“未闭合。",
        "我收到确认。”反向关闭。",
        "我收到确认。“混合关闭」",
        "我收到确认。“「错序”」",
        "小说台词：“我喜欢茶。”",
        "故事里他说：“我喜欢茶。”",
        "角色扮演，我说：“我喜欢茶。”",
        "假设我说：“我喜欢茶。”",
        "比如：“我喜欢茶。”",
        "交流偏好：引用“叫我老师”。",
        "我转述“交流偏好：叫我老师”。",
        "不要保存这条引用“我喜欢茶”。",
        "我引用“密码是合成测试占位符”。",
    ],
)
def test_fiction_note_optout_sensitive_and_malformed_sources_keep_their_admission_gates(text):
    assert not quoted_source_only(text)


class NoModel:
    async def complete(self, messages):
        raise AssertionError("Source-only retention must not promote quoted speech through a fact model")

    async def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,captured",
    [
        (SOURCE, True),
        ("朋友说：“我喜欢茶。”我只是在转述他的原话。", True),
        ("我今天收到课程确认。历史引用是「我喜欢茶」。", True),
        ("小说台词：“我喜欢茶。”", False),
        ("我收到确认。“未闭合。", False),
        ("不要保存这条引用“我喜欢茶”。", False),
        ("我引用“密码是合成测试占位符”。", False),
        ("我转述“交流偏好：叫我老师”。", False),
    ],
)
async def test_actual_completion_captures_full_source_receipt_and_recall_without_fact_or_address_promotion(
    tmp_path, monkeypatch, text, captured
):
    import character.memory_llm as memory_llm

    database = SQLiteDB(tmp_path / "quoted.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    registry = CharacterProfileRegistry()
    registry.load_profiles()
    service = CharacterContextService(registry, repo, DatabaseMessageRepository(database))
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "unused"), completion=NoModel())
    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: scheduler)
    stamp = datetime.now(timezone.utc)
    turn = TurnInput(text, "web", "quote-test", "owner", "owner", "private", received_at=stamp)
    prepared = await service.prepare_turn(turn, "tsukiyashiro_kisaki")
    receipt = asyncio.get_running_loop().create_future()
    try:
        outcome = await service.complete_turn(
            prepared, turn, "收到。", source_message_id="quoted-input", memory_receipt=receipt
        )
        assert await scheduler.flush_memory(timeout=3)
        sources = await repo.list_sources(prepared.character_id, prepared.user_scope)
        assert [r["body"] for r in sources] == ([text] if captured else [])
        assert await repo.list_memory_records(prepared.character_id, prepared.user_scope) == []
        assert (await repo.get_relationship_record(prepared.character_id, prepared.user_scope)).get(
            "preferred_address", ""
        ) == ""
        assert scheduler.status.failed == 0
        if captured:
            assert outcome.memory_enrichment_scheduled and outcome.source_capture == "recorded"
            assert receipt.done() and receipt.result()["status"] == "source_only"
            assert receipt.result()["accepted"] == receipt.result()["persisted"] == 0
            assert datetime.fromisoformat(sources[0]["observed_at"]) == prepared.received_at
            recall = await SourceMemoryService(repo).recall(
                prepared.character_id, prepared.user_scope, "课程确认" if "课程" in text else "朋友说我喜欢茶"
            )
            packet = json.loads(recall.context)
            assert packet["described_subject"] == packet["current_validity"] == "not_resolved"
            assert packet["speaker_role"] == "user" and packet["records"][0]["text"] == text
        else:
            assert not sources and not receipt.done()
    finally:
        await scheduler.shutdown(timeout=3)
        database.close_connection()


@pytest.mark.asyncio
async def test_quoted_source_job_cannot_restore_pre_erasure_work(tmp_path):
    from character.models import UserScope

    database = SQLiteDB(tmp_path / "erasure.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "quote-test", "owner", "owner", "private")
    fields = ("role", "web", "quote-test", "owner", "private", "owner")
    database.clear_character_memories(*fields)
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "unused"), completion=NoModel())
    receipt = asyncio.get_running_loop().create_future()
    try:
        assert scheduler.schedule(
            repository=repo,
            character_id="role",
            user_scope=scope,
            message=SOURCE,
            rule_hints=(),
            source_message_id="late-quote",
            source_only=True,
            observed_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            receipt=receipt,
            immediate=True,
        )
        assert await scheduler.flush_memory(timeout=3)
        assert receipt.done() and receipt.result()["reason"] == "source_stale"
        assert database.list_memory_sources(*fields) == []
        assert await repo.list_memory_records("role", scope) == []
    finally:
        await scheduler.shutdown(timeout=3)
        database.close_connection()
