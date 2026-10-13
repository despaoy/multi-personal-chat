"""Optional intent classifiers may decide routes, but failed predictions cannot."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from db.schemas import MessageRequest
from knowledge import intent_detector

QUERY = "核对虚构夏屿展览的费用、开放时段与雨天例外；无对应来源则保持未知。"


@pytest.mark.parametrize("ml_positive", [True, False])
@pytest.mark.parametrize("confidence", [0.6, 0.65, 0.99])
@pytest.mark.parametrize("rule_positive", [True, False])
def test_valid_predictions_preserve_hybrid_route_without_redundant_rule_call(monkeypatch, ml_positive, confidence, rule_positive):
    calls = []
    name = "虚构夏屿展览" if ml_positive else None

    def predict(message):
        assert message == QUERY
        calls.append("model")
        return ml_positive, "declared classifier decision", confidence, name

    def rule(message, context):
        assert message == QUERY and context == {"turn": "complete"}
        calls.append("rule")
        return rule_positive, "declared rule decision"

    monkeypatch.setattr(intent_detector, "_ML_AVAILABLE", True)
    monkeypatch.setattr(intent_detector, "_ml_predict", predict)
    monkeypatch.setattr(intent_detector, "get_intent_detector", lambda: SimpleNamespace(needs_rag=rule))
    pred, reason, selected = intent_detector.needs_rag(QUERY, {"turn": "complete"})
    assert pred == (ml_positive or rule_positive) and selected == name
    assert reason == ("declared classifier decision" if ml_positive else "declared rule decision")
    assert calls == (["model"] if ml_positive else ["model", "rule"])


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
def test_loaded_classifier_failure_is_not_replaced_by_rules(monkeypatch, error_type):
    error = error_type("Declared isolated prediction failure")

    def predict(_message):
        raise error

    def forbidden():
        pytest.fail("A prediction failure cannot switch to a rule result")

    monkeypatch.setattr(intent_detector, "_ML_AVAILABLE", True)
    monkeypatch.setattr(intent_detector, "_ml_predict", predict)
    monkeypatch.setattr(intent_detector, "get_intent_detector", forbidden)
    with pytest.raises(error_type) as caught:
        intent_detector.needs_rag(QUERY)
    assert caught.value is error


@pytest.mark.parametrize("rule_positive", [True, False])
def test_no_optional_classifier_keeps_explicit_rule_result(monkeypatch, rule_positive):
    monkeypatch.setattr(intent_detector, "_ML_AVAILABLE", False)
    monkeypatch.setattr(intent_detector, "_load_ml_model", lambda: None)
    monkeypatch.setattr(intent_detector, "_ml_predict", lambda _: pytest.fail("No classifier was loaded"))
    monkeypatch.setattr(intent_detector, "get_intent_detector", lambda: SimpleNamespace(
        needs_rag=lambda message, context: (rule_positive, "rules without optional classifier"),
    ))
    assert intent_detector.needs_rag(QUERY) == (rule_positive, "rules without optional classifier", None)


@pytest.mark.asyncio
async def test_prediction_failure_stops_chat_before_retrieval_planning_or_answer(monkeypatch):
    from api import generate
    from knowledge import retrieval_query_plan

    error = RuntimeError("Declared isolated classifier failure")

    def predict(_message):
        raise error

    def forbidden(*args, **kwargs):
        pytest.fail("A failed classifier cannot proceed to rules, planning, retrieval or answer")

    monkeypatch.setattr(intent_detector, "_ML_AVAILABLE", True)
    monkeypatch.setattr(intent_detector, "_ml_predict", predict)
    monkeypatch.setattr(intent_detector, "get_intent_detector", forbidden)
    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", forbidden)
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", forbidden)
    monkeypatch.setattr(generate, "_get_system_prompt", forbidden)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(
            MessageRequest(message=QUERY), None, runtime_config=dict(useKnowledgeBase=True), model_generate=forbidden,
        )
    assert caught.value.status_code == 503 and caught.value.__cause__ is error
