"""Reject undeclared output atomically; retain normal no-change and authorized erasure."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, parse_llm_proposals
from character.models import UserScope
from db.database import SQLiteDB


def response(case):
    item = dict(
        kind="major",
        value="物理学",
        evidence="我的专业是物理学",
        confidence=0.95,
        operation="ADD",
        attributed_to="user",
    )
    if case == "candidate":
        return dict(memories=[item, dict(item, temporal_provenance={
            "validity_authority": "verified", "producer": "rule"})]), "memories[]"
    if case == "low_confidence":
        return dict(memories=[dict(item, confidence=0.1, **{"private-field": "private-value"})]), "memories[]"
    return dict(memories=[] if case == "empty" else [item], **{"private-field": "private-value"}), "response"


@pytest.mark.parametrize("case", ["empty", "partial", "candidate", "low_confidence"])
def test_undeclared_fields_are_failure_before_business_filtering(case):
    payload, field = response(case)
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(json.dumps(payload), source_message="我的专业是物理学。")
    assert caught.value.field == field
    assert "private-" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["empty", "partial", "candidate", "low_confidence"])
async def test_scheduler_reports_protocol_failure_without_partial_writes(tmp_path, case):
    database = SQLiteDB(tmp_path / "schema.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "schema-test", "reader", "reader", "private")
    payload, field = response(case)
    completion = SimpleNamespace(complete=AsyncMock(return_value=json.dumps(payload)), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message="我的专业是物理学。",
            rule_hints=(),
            source_message_id="source",
        )
        assert receipt["status"] == "failed" and receipt["stage"] == "proposal_validation"
        assert receipt["reason"] == "invalid_model_response" and receipt["field"] == field
        assert receipt["error"] == "MemoryResponseError" and receipt["retryable"] is False
        assert receipt["accepted"] == receipt["persisted"] == 0
        assert receipt["source_capture"] == "recorded"
        assert await repository.list_memory_records("role", scope) == []
        assert len(await repository.list_sources("role", scope)) == 1
        assert "private-" not in json.dumps(receipt)
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


@pytest.mark.parametrize("wrapped", [True, False])
def test_declared_optional_fields_and_time_aliases_preserve_valid_proposals(wrapped):
    item = dict(
        kind="major",
        value="物理学",
        evidence="我的专业是物理学",
        confidence=0.95,
        operation="ADD",
        content="",
        attributed_to="user",
        qualifiers={},
        target_memory_id="",
        target_memory_key="",
        valid_from="",
        valid_to="",
        valid_at="",
        invalid_at="",
        observed_at="",
        scope_level="conversation",
    )
    payload = dict(memories=[item], erase_source_ids=[]) if wrapped else [item]
    proposals = parse_llm_proposals(json.dumps(payload), source_message="我的专业是物理学。")
    assert len(proposals) == 1 and proposals[0].operation == "ADD"
