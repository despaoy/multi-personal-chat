"""Balanced evaluation must use the requested service contract without degradation."""
from types import SimpleNamespace

import numpy as np
import pytest
from experiments import evaluate_cahm_balanced as evaluator


def provider():
    return SimpleNamespace(model_id="explicit-test-embedding", dimension=2,
                           embed_texts=lambda texts: np.ones((len(texts), 2), dtype=np.float32))


@pytest.mark.parametrize("historical", [False, True])
async def test_actual_service_receives_history_control(historical, monkeypatch):
    seen = []
    original = evaluator._CaseRepository.list_memory_records
    async def read(self, character_id, user_scope, limit=100, *, include_inactive=False):
        seen.append(include_inactive)
        return await original(self, character_id, user_scope, limit, include_inactive=include_inactive)
    monkeypatch.setattr(evaluator._CaseRepository, "list_memory_records", read)
    case = dict(id="history-control", query="我喜欢红茶吗？", include_historical=historical,
                gold_ids=["tea"], records=[dict(id="tea", status="active", memory_type="user_fact",
                memory_key="preference_红茶", content="用户说喜欢红茶", importance=0.9, confidence=1.0)])
    result = await evaluator._evaluate_retrieval_variant([case], evaluator.RETRIEVAL_VARIANTS[1], provider())
    assert seen == [historical]
    assert result["successfully_processed_cases"] == 1
    assert result["failures"] == []
    assert result["recall_at_5"] == 1
    assert result["unsupported_service_switches"] == []
    assert result["historical_query_control_supported"] is True


async def test_repository_failure_is_not_counted_as_completed(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("synthetic storage unavailable")
    monkeypatch.setattr(evaluator._CaseRepository, "list_memory_records", fail)
    result = await evaluator._evaluate_retrieval_variant(
        [dict(id="storage-failure", query="我喜欢红茶吗？", gold_ids=[], records=[])],
        evaluator.RETRIEVAL_VARIANTS[1], provider())
    assert result["successfully_processed_cases"] == 0
    assert result["failures"][0]["error"] == "RuntimeError: synthetic storage unavailable"


def test_unsupported_service_parameters_fail_instead_of_disappearing(monkeypatch):
    class ObsoleteService:
        def __init__(self, repository, *, embedding_provider):
            raise AssertionError("must fail argument binding first")
    monkeypatch.setattr(evaluator, "CharacterMemoryService", ObsoleteService)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        evaluator._service_for_variant(evaluator._CaseRepository(), evaluator.RETRIEVAL_VARIANTS[1], provider(),
                                       min_hybrid_score=0.35, candidate_limit=100)



@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("canonical", [False, True])
def test_case_repository_decodes_complete_evidence_at_its_boundary(balanced, canonical):
    from copy import deepcopy
    from experiments.evaluate_cahm import _CaseRepository

    body = "完整虚构原文：仅本人可申请，周末不受理，不得自动续借。"
    record = dict(id="fixture", memory_type="user_fact", content=body, source_message_id="first")
    if canonical:
        record.update(evidence=[body], source_message_ids=["second"],
                      evidence_json='["旧摘要"]', source_message_ids_json='["旧来源"]')
    else:
        import json
        record.update(evidence_json=json.dumps([body]), source_message_ids_json='["second"]')
    original = deepcopy(record)
    adapter = evaluator._CaseRepository if balanced else _CaseRepository
    row = adapter([record]).records[0]
    assert row["evidence"] == [body] and row["source_message_ids"] == ["first", "second"]
    assert record == original


@pytest.mark.parametrize("balanced", [False, True])
def test_case_repository_rejects_corrupt_raw_evidence(balanced):
    from experiments.evaluate_cahm import _CaseRepository
    from character.memory_read_authority import MemoryRecordError

    adapter = evaluator._CaseRepository if balanced else _CaseRepository
    with pytest.raises(MemoryRecordError, match="evidence_json"):
        adapter([dict(id="fixture", content="完整虚构原话", evidence_json='["未结束')])



@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("field", ["evidence", "source_message_ids"])
def test_case_repository_rejects_null_canonical_evidence(balanced, field):
    from experiments.evaluate_cahm import _CaseRepository
    from character.memory_read_authority import MemoryRecordError

    adapter = evaluator._CaseRepository if balanced else _CaseRepository
    with pytest.raises(MemoryRecordError) as caught:
        adapter([dict(id="fixture", content="完整虚构原话", **{field: None})])
    assert caught.value.field == field



@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("source,expected", [
    ({"status": " superseded ", "relation_type": " coexist ", "memory_status": "active"}, ("superseded", "COEXIST")),
    ({"memory_status": " archived ", "relation": " merge "}, ("archived", "MERGE")),
    ({"memory_status": " current "}, ("active", "ADD")),
    ({}, ("active", "ADD")),
])
def test_case_repository_normalizes_its_documented_lifecycle_format(balanced, source, expected):
    from copy import deepcopy
    from experiments.evaluate_cahm import _CaseRepository

    original = deepcopy(source)
    adapter = evaluator._CaseRepository if balanced else _CaseRepository
    row = adapter([dict(id="fixture", content="完整虚构声明", **source)]).records[0]
    assert (row["status"], row["relation_type"]) == expected
    assert source == original


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("field", ["status", "relation_type"])
def test_case_repository_rejects_null_lifecycle_instead_of_using_alias(balanced, field):
    from experiments.evaluate_cahm import _CaseRepository
    from character.memory_read_authority import MemoryRecordError

    adapter = evaluator._CaseRepository if balanced else _CaseRepository
    with pytest.raises(MemoryRecordError) as caught:
        adapter([dict(id="fixture", content="完整虚构声明", memory_status="active", relation="ADD", **{field: None})])
    assert caught.value.field == field


@pytest.mark.parametrize("field,alias", [("status", "memory_status"), ("relation_type", "relation")])
async def test_recall_rejects_missing_standard_lifecycle_field(field, alias):
    from unittest.mock import AsyncMock
    from character.memory_service import CharacterMemoryService
    from character.models import UserScope

    row = dict(id="fixture", memory_type="user_fact", memory_key="preference_红茶", content="用户说喜欢红茶",
               status="active", relation_type="ADD", evidence=["我喜欢红茶。"], source_message_ids=[])
    row[alias] = row.pop(field)
    repository = SimpleNamespace(list_memory_records=AsyncMock(return_value=[row]))
    with pytest.raises(KeyError) as caught:
        await CharacterMemoryService(repository, semantic_enabled=False).load_relevant_memories(
            "fixture-role", UserScope("test", "lifecycle", "owner", "room", "private"), "我喜欢红茶吗？")
    assert caught.value.args == (field,)


async def test_evaluator_counts_lifecycle_leakage_for_integer_zero_identity():
    record = dict(id=0, memory_type="user_fact", memory_key="preference_红茶", content="用户以前喜欢红茶",
                  status="superseded", relation_type="ADD", importance=1)
    case = dict(id="zero-identity", query="我喜欢红茶吗？", gold_ids=[], records=[record])
    result = await evaluator._evaluate_retrieval_variant([case], evaluator.RETRIEVAL_VARIANTS[0], provider())
    assert result["successfully_processed_cases"] == result["injected_memories"] == 1
    assert result["lifecycle_leakage"]["superseded_count"] == 1


async def test_evaluator_rejects_a_selected_identity_outside_its_fixture(monkeypatch):
    from unittest.mock import AsyncMock
    from character.models import MemoryItem

    record = dict(id="valid", memory_type="user_fact", memory_key="preference_红茶", content="用户喜欢红茶",
                  status="active", relation_type="ADD")
    case = dict(id="unbound-identity", query="我喜欢红茶吗？", gold_ids=[], records=[record])
    monkeypatch.setattr(evaluator.CharacterMemoryService, "load_relevant_memories",
                        AsyncMock(return_value=((MemoryItem("unbound", "user_fact", "无来源的结果"),), 1)))
    with pytest.raises(KeyError) as caught:
        await evaluator._evaluate_retrieval_variant([case], evaluator.RETRIEVAL_VARIANTS[0], provider())
    assert caught.value.args == ("unbound",)
