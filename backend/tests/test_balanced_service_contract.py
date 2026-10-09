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
