"""Generation/queue delays must not move the user's relative-time reference."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import CompiledCharacterContext, RelationshipState, UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService, PreparedCharacterTurn, TurnInput

RECEIVED = datetime(2026, 1, 31, 15, 59, tzinfo=timezone.utc)  # 23:59 in the existing event timezone.
SCOPE = UserScope("test", "clock", "user", "user", "private")


@pytest.mark.asyncio
async def test_background_model_receives_source_clock_not_processing_clock(tmp_path):
    class Completion:
        payload = None

        async def complete(self, messages):
            self.payload = json.loads(messages[1]["content"])
            return '{"memories":[]}'

        async def close(self):
            pass

    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "unused", "unused"), completion=completion)
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "model-clock.sqlite"))
    assert scheduler.schedule(repository=repo, character_id="role", user_scope=SCOPE,
                              message="我明天要面试。", rule_hints=[], observed_at=RECEIVED)
    await scheduler.shutdown(timeout=3)
    assert datetime.fromisoformat(completion.payload["current_time_utc"]) == RECEIVED
    assert completion.payload["current_time_local"] == "2026-01-31T23:59:00+08:00"


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["我明天要面试。", "我喜欢红茶。"])
@pytest.mark.parametrize("received,expected_date", [
    (RECEIVED, "2026-02-01"),
    (datetime(2028, 2, 28, 15, 59, tzinfo=timezone.utc), "2028-02-29"),
    (datetime(2026, 12, 31, 15, 59, tzinfo=timezone.utc), "2027-01-01"),
    (datetime.fromisoformat("2026-04-30T23:59:00+08:00"), "2026-05-01"),
    (datetime(2026, 1, 31, 16, 1, tzinfo=timezone.utc), "2026-02-02"),
])
async def test_rules_use_received_time_for_event_and_observation(tmp_path, monkeypatch, message, received, expected_date):
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler",
                        lambda: SimpleNamespace(enabled=False))
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "rule-clock.sqlite"))
    service = CharacterContextService(object(), repo, object())
    prepared = PreparedCharacterTurn("role", SCOPE, CompiledCharacterContext("", "", ""), (),
                                     RelationshipState(), 0, 0, None, received_at=received)
    turn = TurnInput(message, "test", "clock", "user", "user", "private")
    outcome = await service.complete_turn(prepared, turn, "好的", source_message_id="original")
    assert outcome.new_memories == 1
    row, = await repo.list_memory_records("role", SCOPE, limit=None)
    assert datetime.fromisoformat(row["observed_at"]) == received
    if "面试" in message:
        assert row["metadata"]["event"]["scheduled_date"] == expected_date


def test_naive_model_reference_clock_is_rejected():
    from character.memory_llm import build_memory_llm_messages

    with pytest.raises(ValueError, match="timezone"):
        build_memory_llm_messages("我明天要面试。", (), (), (), 2000, .8,
                                  observed_at=datetime(2026, 1, 1))
