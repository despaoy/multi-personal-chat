"""Invalid evaluation settings fail before model-backed work."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from experiments import evaluate_cahm_balanced as evaluator


def arguments(tmp_path, **overrides):
    dataset = tmp_path / "complete.jsonl"
    dataset.write_text(json.dumps(dict(task="retrieval_v2", id="tea", query="我喜欢红茶吗？",
        gold_ids=["tea"], records=[dict(id="tea", memory_key="preference_红茶", memory_type="user_fact",
        content="用户说喜欢红茶", importance=0.9, confidence=1.0, status="active")])) , encoding="utf-8")
    return SimpleNamespace(**(dict(dataset=dataset, memory_llm_base_url="", memory_llm_model="",
        min_hybrid_score=0.35, candidate_limit=100) | overrides))


@pytest.mark.parametrize("override", [dict(candidate_limit=0), dict(candidate_limit=1.5),
    dict(min_hybrid_score=float("nan")), dict(min_hybrid_score=1.2)])
async def test_invalid_settings_precede_relationship_calls_and_embedding(tmp_path, monkeypatch, override):
    relations = AsyncMock(side_effect=AssertionError("must not invoke relation evaluation"))
    provider = Mock()
    factory = Mock(side_effect=AssertionError("must not construct provider"))
    monkeypatch.setattr(evaluator, "_evaluate_relations", relations)
    monkeypatch.setattr(evaluator, "get_default_embedding_provider", factory)
    with pytest.raises(ValueError, match="candidate_limit|min_hybrid_score"):
        await evaluator.evaluate(arguments(tmp_path, **override), embedding_provider=provider)
    assert provider.mock_calls == []
    relations.assert_not_called()
    factory.assert_not_called()


async def test_valid_settings_run_both_actual_retrieval_variants(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")
    provider = SimpleNamespace(model_id="explicit-test-embedding", dimension=2,
        embed_texts=lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    report = await evaluator.evaluate(arguments(tmp_path), embedding_provider=provider)
    assert report["relation"]["evaluated"] is False
    assert len(report["retrieval"]) == 2
    for result in report["retrieval"].values():
        assert result["successfully_processed_cases"] == 1
        assert result["failures"] == []
        assert result["effective_configuration"]["candidate_limit"] == 100
        assert result["effective_configuration"]["min_hybrid_score"] == 0.35
