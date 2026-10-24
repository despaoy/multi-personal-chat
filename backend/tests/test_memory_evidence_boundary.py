"""Model quote bounds precede admission; derived complete observations preserve sources."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, parse_llm_proposals
from character.models import UserScope
from db.database import SQLiteDB

HISTORY = ({"role": "user", "content": "我最近在准备保研面试。"},)
CURRENT = "还是上次那个方向。"


def candidate(evidence, **changes):
    return dict(
        dict(kind="goal", value="保研面试", evidence=evidence, operation="ADD", confidence=0.95, attributed_to="user"),
        **changes,
    )


def parse(raw, source=CURRENT, history=HISTORY):
    return parse_llm_proposals(json.dumps(dict(memories=[raw])), source_message=source, history=history)


def test_copied_historical_quote_is_not_rewritten_into_new_evidence():
    assert parse(candidate(HISTORY[0]["content"])) == []


def test_current_quote_can_still_use_user_history_to_resolve_value():
    (proposal,) = parse(candidate(CURRENT))
    assert proposal.evidence == CURRENT and proposal.memory.content == "用户正在进行或准备：保研面试"
    with pytest.raises(MemoryResponseError, match="value"):
        parse(candidate(CURRENT), history=())
    with pytest.raises(MemoryResponseError, match="value"):
        parse(candidate(CURRENT), history=({"role": "assistant", "content": HISTORY[0]["content"]},))


@pytest.mark.parametrize("confidence", [0.1, 0.95])
def test_oversized_model_quote_fails_before_business_filtering(confidence):
    quote = "我计划学习陶艺，" + "甲" * 113
    assert len(quote) == 121
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(quote, kind="shared_event", value="陶艺", confidence=confidence), source=quote, history=())
    assert caught.value.field == "evidence" and quote not in str(caught.value)


def test_exact_quote_limit_and_complete_long_source_observation_remain_supported():
    quote = "我计划学习陶艺，" + "甲" * 112
    assert len(quote) == 120
    source = quote + "；当前只是计划，只有收到书面录取后才考虑出发，目前尚未收到。"
    (proposal,) = parse(candidate(quote, kind="shared_event", value="陶艺"), source=source, history=())
    assert proposal.evidence == source and proposal.source_observation
    assert "完整内容见证据" in proposal.memory.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,expected,reason",
    [
        ("copied", "no_change", "no_eligible_memory"),
        ("empty", "no_change", "no_new_information"),
        ("current", "saved", None),
        ("oversized", "failed", "invalid_model_response"),
    ],
)
async def test_receipt_distinguishes_quote_failure_admission_and_empty_extraction(tmp_path, case, expected, reason):
    database = SQLiteDB(tmp_path / "evidence.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "evidence-boundary", "reader", "reader", "private")
    memories = [] if case == "empty" else [candidate(HISTORY[0]["content"] if case == "copied" else CURRENT)]
    if case == "oversized":
        memories.append(candidate("甲" * 121))
    completion = SimpleNamespace(
        complete=AsyncMock(return_value=json.dumps(dict(memories=memories))), close=AsyncMock()
    )
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message=CURRENT,
            history=HISTORY,
            rule_hints=(),
            source_message_id="current",
        )
        assert receipt["status"] == expected and receipt["source_capture"] == "recorded"
        records = await repository.list_memory_records("role", scope)
        if case == "current":
            assert receipt["persisted"] == len(records) == 1
        else:
            assert receipt["reason"] == reason and receipt["accepted"] == receipt["persisted"] == 0 and not records
        if case == "oversized":
            assert receipt["error"] == "MemoryResponseError" and receipt["field"] == "evidence"
            assert receipt["stage"] == "proposal_validation" and receipt["retryable"] is False
        assert len(await repository.list_sources("role", scope)) == 1
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()
