"""Invalid writing fields fail before admission; target-only destructive actions remain valid."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, parse_llm_proposals
from character.models import UserScope
from db.database import SQLiteDB

SOURCE = "我的专业是物理学。"
OLD = dict(
    id="7", memory_key="user_major", memory_type="user_fact", content="用户说自己的专业是物理学", status="active"
)


def candidate(**changes):
    return dict(
        dict(kind="major", value="物理学", evidence=SOURCE, confidence=0.96, operation="ADD", attributed_to="user"),
        **changes,
    )


def parse(raw, source=SOURCE, existing=()):
    return parse_llm_proposals(json.dumps(dict(memories=[raw])), source_message=source, existing_memories=existing)


@pytest.mark.parametrize("operation", ["ADD", "MERGE", "SUPERSEDE", "COEXIST", "PENDING"])
@pytest.mark.parametrize("field,bad", [("kind", ""), ("kind", "private-unknown"), ("value", ""), ("value", "甲" * 49)])
def test_invalid_writing_field_cannot_be_hidden_by_low_confidence_or_missing_target(operation, field, bad):
    raw = candidate(operation=operation, confidence=0.1, **{field: bad})
    with pytest.raises(MemoryResponseError) as caught:
        parse(raw)
    assert caught.value.field == field and "private-" not in str(caught.value)


def test_unknown_kind_cannot_be_guessed_from_valid_old_target():
    source = "刚才说错了，我的专业是化学。"
    raw = candidate(
        kind="private-unknown",
        value="化学",
        evidence=source,
        operation="SUPERSEDE",
        target_memory_id="7",
        target_memory_key="user_major",
    )
    with pytest.raises(MemoryResponseError, match="kind"):
        parse(raw, source, (OLD,))


@pytest.mark.parametrize(
    "operation,source",
    [("RETRACT", "我撤回之前关于我的专业是物理学的说法。"), ("ERASE", "请把你记住的我的专业从记忆里删除。")],
)
def test_target_only_retraction_and_erasure_do_not_need_new_fact_fields(operation, source):
    raw = candidate(evidence=source, operation=operation, target_memory_id="7", target_memory_key="user_major")
    del raw["kind"]
    del raw["value"]
    proposals = parse(raw, source, (OLD,))
    assert len(proposals) == 1 and proposals[0].operation == operation
    assert proposals[0].target_memory_id == "7" and proposals[0].target_memory_key == "user_major"


@pytest.mark.parametrize("length", [48, 49])
def test_value_contract_uses_declared_index_limit(length):
    value = "甲" * length
    source = "我的专业是" + value + "。"
    raw = candidate(value=value, evidence=source)
    if length == 49:
        with pytest.raises(MemoryResponseError, match="value"):
            parse(raw, source)
    else:
        proposals = parse(raw, source)
        assert len(proposals) == 1 and value in proposals[0].memory.content


def test_valid_low_confidence_and_unsupported_evidence_remain_normal_rejection():
    assert parse(candidate(confidence=0.1)) == []
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(value="化学"))
    assert caught.value.field == "value"


@pytest.mark.asyncio
@pytest.mark.parametrize("field,bad", [("kind", "private-unknown"), ("value", ""), ("value", "甲" * 49)])
async def test_invalid_second_candidate_never_persists_valid_first_candidate(tmp_path, field, bad):
    database = SQLiteDB(tmp_path / "writing.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "write-contract", "reader", "reader", "private")
    raw = json.dumps(dict(memories=[candidate(), candidate(**{field: bad})]))
    completion = SimpleNamespace(complete=AsyncMock(return_value=raw), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message=SOURCE,
            rule_hints=(),
            source_message_id="source",
        )
        assert receipt["status"] == "failed" and receipt["stage"] == "proposal_validation"
        assert receipt["error"] == "MemoryResponseError" and receipt["reason"] == "invalid_model_response"
        assert receipt["field"] == field and receipt["retryable"] is False
        assert receipt["accepted"] == receipt["persisted"] == 0
        assert receipt["source_capture"] == "recorded" and not await repository.list_memory_records("role", scope)
        assert "private-" not in json.dumps(receipt)
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()
