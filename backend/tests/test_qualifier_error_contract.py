"""Qualifier protocol errors must identify their field and prevent partial writes."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, _sanitize_qualifiers
from character.models import UserScope
from db.database import SQLiteDB

SOURCE = "我喜欢蓝莓茶；只有收到书面录取后才去陶艺班，目前还没有收到。"
QUOTE = "我喜欢蓝莓茶"


@pytest.mark.parametrize(
    "raw",
    [
        True,
        7,
        {"private-key": "private-value"},
        {"condition": {"private-key": "private-value"}},
        {"condition": ""},
        {"condition": "甲" * 49},
        {"condition": "收到书面录取"},
        {"condition": "我喜欢", " Condition ": "蓝莓茶"},
        ["我喜欢", "蓝莓茶"],
    ],
)
def test_invalid_qualifier_contract_is_specific_safe_error(raw):
    with pytest.raises(MemoryResponseError) as caught:
        _sanitize_qualifiers(raw, evidence=QUOTE)
    assert caught.value.field == "qualifiers"
    assert "private-" not in str(caught.value)


@pytest.mark.parametrize("raw", [None, "", {}, []])
def test_absent_optional_qualifiers_remain_valid(raw):
    assert _sanitize_qualifiers(raw, evidence=QUOTE) == ()


def test_literal_qualifiers_and_supported_pending_flag_remain_valid():
    assert _sanitize_qualifiers({"condition": "收到书面录取"}, evidence=SOURCE) == (("condition", "收到书面录取"),)
    assert _sanitize_qualifiers({"certainty": False}, evidence=QUOTE, pending=True) == (("certainty", "false"),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "qualifiers",
    [{"private-key": "private-value"}, {"condition": "收到书面录取"}, {"condition": {}}, {"condition": "甲" * 49}],
)
async def test_malformed_second_candidate_cannot_hide_as_no_change_or_partially_save(tmp_path, qualifiers):
    database = SQLiteDB(tmp_path / "qualifier-error.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "qualifier-error", "reader", "reader", "private")
    item = dict(
        kind="like",
        value="蓝莓茶",
        evidence=QUOTE,
        confidence=0.96,
        operation="ADD",
        attributed_to="user",
        qualifiers={},
    )
    completion = SimpleNamespace(
        complete=AsyncMock(return_value=json.dumps(dict(memories=[item, dict(item, qualifiers=qualifiers)]))),
        close=AsyncMock(),
    )
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        result = await worker.schedule_and_wait(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message=SOURCE,
            rule_hints=(),
            source_message_id="source",
        )
        assert result["status"] == "failed" and result["reason"] == "invalid_model_response"
        assert result["field"] == "qualifiers" and result["stage"] == "proposal_validation"
        assert "qualifiers" in result["message"]
        assert result["error"] == "MemoryResponseError" and result["retryable"] is False
        assert result["accepted"] == result["persisted"] == 0 and result["source_capture"] == "recorded"
        assert not await repository.list_memory_records("role", scope)
        assert (await repository.list_sources("role", scope))[0]["body"] == SOURCE
        assert "private-" not in json.dumps(result)
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


def test_excess_qualifier_count_has_explicit_error():
    with pytest.raises(MemoryResponseError, match="数量") as caught:
        _sanitize_qualifiers({str(index): "private-value" for index in range(9)}, evidence=QUOTE)
    assert caught.value.field == "qualifiers"


def test_actual_json_boundary_already_normalizes_object_keys_to_strings():
    from character.memory_llm import parse_llm_proposals

    item = dict(
        kind="like",
        value="蓝莓茶",
        evidence=QUOTE,
        confidence=0.96,
        operation="ADD",
        attributed_to="user",
        qualifiers={1: "private-value"},
    )
    with pytest.raises(MemoryResponseError, match="qualifiers") as caught:
        parse_llm_proposals(json.dumps(dict(memories=[item])), source_message=SOURCE)
    assert caught.value.field == "qualifiers" and "private-value" not in str(caught.value)
