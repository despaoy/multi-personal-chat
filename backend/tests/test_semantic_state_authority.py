"""Full semantic review owns non-safety meaning through the actual compiler."""

import json

import pytest
from services.character_context import build_character_context_service
from test_semantic_character_context import _service, _turn

from character.models import InteractionState, WeightedSignal
from character.semantic_state_estimator import REVIEW_FULL_ABLATION, SemanticStateEstimator
from character.situation_analyzer import SituationAnalyzer
from inference.generation_request import GenerationRequest, build_generation_request
from inference.provider_context import ProviderContextBudget

QUERY = "这是语法题：请把“晚安，再见”转换成第三人称转述。这句话只是待分析的材料，我没有结束当前对话。"


def factual_result():
    return dict(
        state=dict(
            primary_situation="factual",
            situation_scores=dict(factual=0.95),
            user_acts=dict(information_request=0.95),
            user_needs=dict(information=0.95),
            valence=0.0,
            arousal=0.2,
            warmth=0.0,
            face_threat=0.0,
            conversation_phase="sustaining",
            confidence=0.95,
        )
    )


async def reviewed(estimator, query, original, payload):
    calls = []

    async def reviewer(messages):
        wire = json.loads(messages[-1]["content"])
        assert wire["current_message"] == query and wire["recent_history"] == []
        assert wire["review_reasons"] == [REVIEW_FULL_ABLATION]
        calls.append(messages)
        return json.dumps(payload)

    estimator._reviewer = reviewer
    outcome = await estimator.refine_with_diagnostics(query, [], original)
    assert len(calls) == 1 and outcome.status == "applied"
    return outcome


async def test_quoted_farewell_does_not_survive_complete_semantic_review():
    original = SituationAnalyzer().estimate(QUERY, [])
    assert original.conversation_phase == "closing" and any(s.signal_id == "closing" for s in original.user_acts)
    result = await reviewed(
        SemanticStateEstimator(None, review_mode="all_non_safety"), QUERY, original, factual_result()
    )
    assert result.state.conversation_phase == "sustaining" and result.state.primary_situation == "factual"
    assert result.state.user_acts == (WeightedSignal("information_request", 0.95),)
    assert result.state.user_needs == (WeightedSignal("information", 0.95),)


@pytest.mark.parametrize(
    "old_primary,old_act,old_phase",
    [
        ("meta", "gratitude", "closing"),
        ("emotional", "advice_boundary", "repairing"),
        ("conflict", "repair_bid", "repairing"),
    ],
)
async def test_full_result_does_not_merge_conflicting_rule_hypotheses(old_primary, old_act, old_phase):
    original = InteractionState(
        primary_situation=old_primary,
        situation_scores=(WeightedSignal(old_primary, 0.99),),
        user_acts=(WeightedSignal(old_act, 0.99),),
        user_needs=(WeightedSignal("autonomy", 0.99),),
        valence=-0.7,
        conversation_phase=old_phase,
        confidence=0.99,
    )
    result = await reviewed(
        SemanticStateEstimator(None, review_mode="all_non_safety"), QUERY, original, factual_result()
    )
    assert result.state.primary_situation == "factual" and result.state.conversation_phase == "sustaining"
    assert result.state.situation_scores == (WeightedSignal("factual", 0.95),)
    assert result.state.user_acts == (WeightedSignal("information_request", 0.95),)
    assert result.state.user_needs == (WeightedSignal("information", 0.95),) and result.state.valence == 0.0


async def test_full_review_does_not_run_keyword_reason_classifier(monkeypatch):
    from character import semantic_state_estimator

    def forbidden(*_args):
        pytest.fail("Full review must not classify ambiguity by keywords")

    monkeypatch.setattr(semantic_state_estimator, "semantic_review_reasons", forbidden)
    original = InteractionState(
        primary_situation="daily", situation_scores=(WeightedSignal("daily", 0.99),), confidence=0.99
    )
    result = await reviewed(
        SemanticStateEstimator(None, review_mode="all_non_safety"),
        "请计算完整题设：7加8的结果。",
        original,
        factual_result(),
    )
    assert result.state.primary_situation == "factual"


async def test_an_actual_model_reviewed_farewell_is_retained():
    query = "事情已经处理完了，晚安，结束这次聊天。"
    raw = factual_result()
    raw["state"].update(
        primary_situation="daily",
        situation_scores=dict(daily=0.95),
        user_acts=dict(closing=0.95),
        user_needs={},
        conversation_phase="closing",
    )
    result = await reviewed(
        SemanticStateEstimator(None, review_mode="all_non_safety"), query, SituationAnalyzer().estimate(query, []), raw
    )
    assert result.state.conversation_phase == "closing" and result.state.user_acts == (WeightedSignal("closing", 0.95),)


async def test_production_factory_semantics_reach_final_dynamic_prompt(monkeypatch):
    service = build_character_context_service(object(), context_budget=ProviderContextBudget(65536))
    assert service._semantic_estimator._review_mode == "all_non_safety"
    estimator = service._semantic_estimator

    async def reviewer(messages):
        assert json.loads(messages[-1]["content"])["current_message"] == QUERY
        return json.dumps(factual_result())

    estimator._reviewer = reviewer
    # Complete, declared source controls isolate the semantic state and compiler;
    # no stored memories or shared history are required by this grammar task.
    prepared = await _service(estimator).prepare_turn(_turn(QUERY), "tsukiyashiro_kisaki")
    assert prepared.semantic_review_status == "applied" and prepared.interaction.conversation_phase == "sustaining"
    assert prepared.decision.strategy_ids == ("respond_directly",)
    plan = build_generation_request(GenerationRequest(message=QUERY, character_context=prepared.compiled))
    system = "\n".join(row["content"] for row in plan.messages if row["role"] == "system")
    assert "自然接住告别" not in system and QUERY not in system
    assert QUERY in plan.messages[-1]["content"]


async def test_full_provider_failure_is_not_replaced_by_original_state():
    error = RuntimeError("synthetic semantic provider unavailable")
    calls = []

    async def broken(messages):
        calls.append(messages)
        raise error

    estimator = SemanticStateEstimator(broken, review_mode="all_non_safety")
    with pytest.raises(RuntimeError) as caught:
        await estimator.refine_with_diagnostics(QUERY, [], SituationAnalyzer().estimate(QUERY, []))
    assert caught.value is error and len(calls) == 1
