"""Offline research routes cannot conceal dependency/configuration failures."""

import sys
from unittest.mock import Mock, patch

import pytest

from inference.lora_router import LoRARouter


@pytest.mark.parametrize("failure", ["import", "classifier", "missing_adapter", "invalid_adapter", "empty_adapter", "incomplete_intent", "nonnumeric_intent"])
def test_offline_failure_is_not_replaced_by_a_route(failure):
    config = dict(enabled=True, mode="intent", default_adapter="default",
                  persona_keywords={"fixture": ["合成角色"]}, persona_adapters={"fixture": "fixture"})
    router = LoRARouter(config)
    if failure == "import":
        with patch.dict(sys.modules, {"knowledge.intent_detector": None}), pytest.raises(ImportError):
            router.route("合成角色")
    elif failure == "classifier":
        error = TimeoutError("synthetic classifier unavailable")
        router._intent_detector = Mock(side_effect=error)
        with pytest.raises(TimeoutError) as caught:
            router.route("合成角色")
        assert caught.value is error
        router._intent_detector.assert_called_once()
    elif failure == "incomplete_intent":
        with pytest.raises(ValueError):
            router.route("合成角色", intent_result=(True,))
    elif failure == "nonnumeric_intent":
        with pytest.raises(ValueError):
            router.route("合成角色", intent_result=(True, "reason instead of score", "fixture"))
    else:
        router.mode = "rule"
        router.persona_adapters = ({"fixture": "../invalid"} if failure == "invalid_adapter" else
                                  {"fixture": ""} if failure == "empty_adapter" else {})
        with pytest.raises(KeyError if failure == "missing_adapter" else ValueError):
            router.route("合成角色")


def test_router_uses_classifier_score_without_promoting_it(monkeypatch):
    from knowledge import intent_detector

    load = Mock()
    predict = Mock(return_value=(True, "controlled classifier result", .63, "fixture"))
    monkeypatch.setattr(intent_detector, "_load_ml_model", load)
    monkeypatch.setattr(intent_detector, "_ml_predict", predict)
    router = LoRARouter(dict(enabled=True, mode="intent", rag_confidence_threshold=.5,
                            persona_keywords={}, persona_adapters={}))
    result = router.route("请查询合成资料的预约时间和未付款条件。")
    assert result.target == "rag_required" and result.confidence == .63
    router.rag_confidence_threshold = .8
    assert router.route("请查询合成资料的预约时间和未付款条件。").target == "base_chat"
    assert load.call_count == predict.call_count == 2
