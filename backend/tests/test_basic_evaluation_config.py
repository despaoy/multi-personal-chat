"""Basic evaluation rejects incomplete configuration before model work."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from experiments import evaluate_cahm as evaluator


@pytest.mark.parametrize("base_url,model,score,error", [
    ("https://model.invalid/v1", "", 0.35, "provided together"),
    ("", "example-model", 0.35, "provided together"),
    ("https://model.invalid/v1", "  ", 0.35, "provided together"),
    ("  ", "example-model", 0.35, "provided together"),
    ("", "", float("nan"), "min_hybrid_score"),
    ("", "", -0.1, "min_hybrid_score"),
])
async def test_bad_configuration_stops_before_model_work(tmp_path, monkeypatch, base_url, model, score, error):
    dataset = tmp_path / "case.jsonl"
    dataset.write_text(json.dumps(dict(id="name", task="extraction", message="我叫林澈。", gold_keys=["user_name"])), encoding="utf-8")
    context = AsyncMock(side_effect=AssertionError("must not enter model extraction"))
    provider = Mock(side_effect=AssertionError("must not initialize embedding"))
    monkeypatch.setattr(evaluator, "_context_predictions", context)
    monkeypatch.setattr(evaluator, "get_default_embedding_provider", provider)
    args = SimpleNamespace(dataset=dataset, memory_llm_base_url=base_url, memory_llm_model=model,
                           min_hybrid_score=score, memory_llm_api_key="", memory_llm_confidence_threshold=0.85)
    with pytest.raises(ValueError, match=error):
        await evaluator.evaluate(args)
    context.assert_not_called()
    provider.assert_not_called()
