"""Observation time is server provenance, never a model-generated fact."""

import json
from datetime import datetime, timezone

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.asyncio
@pytest.mark.parametrize("model_time", ["2099-01-01T00:00:00Z", "1900-01-01", "not-a-date", ""])
@pytest.mark.parametrize("source_time", [None, "2026-01-02T08:30:00+08:00"])
async def test_model_cannot_set_observation_time(tmp_path, model_time, source_time):
    class Completion:
        async def complete(self, messages):
            return json.dumps({"memories": [{"attributed_to": "user",
                "operation": "ADD", "kind": "study_stage", "value": "大三", "evidence": "今年刚升大三",
                "confidence": 0.96, "observed_at": model_time,
                "valid_from": "2025-09-01", "valid_to": "2026-07-01",
            }]})

        async def close(self):
            pass

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / "clock.db"))
    scope = UserScope("qq", "test", "user", "user", "private")
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="stub"),
        completion=Completion(),
    )
    before = datetime.now(timezone.utc)
    assert scheduler.schedule(repository=repo, character_id="test", user_scope=scope,
                              message="今年刚升大三", rule_hints=[], source_message_id="source",
                              observed_at=datetime.fromisoformat(source_time) if source_time else None)
    after = datetime.now(timezone.utc)
    await scheduler.shutdown(timeout=3)
    records = await repo.list_memory_records("test", scope)
    assert len(records) == 1
    observed = datetime.fromisoformat(records[0]["observed_at"])
    if source_time:
        assert observed == datetime.fromisoformat(source_time)
        assert observed.utcoffset().total_seconds() == 0
    else:
        assert before <= observed <= after
    # Event validity is independent of the system observation clock.
    assert records[0]["valid_from"].startswith("2025-09-01")
    assert records[0]["valid_to"].startswith("2026-07-01")


def test_naive_observation_time_rejected_before_enqueue():
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(enabled=True, base_url="http://unused", model="stub"),
        completion=object(),
    )
    with pytest.raises(ValueError, match="timezone"):
        scheduler.schedule(repository=object(), character_id="test",
                           user_scope=UserScope("qq", "test", "user", "user", "private"),
                           message="今年刚升大三", rule_hints=[], observed_at=datetime(2026, 1, 1))
