"""Configured semantic selection has no dependency on discarded optional scores."""

import json
from unittest.mock import Mock

import pytest

from character.context_builder import compile_dynamic_context
from character.contextual_policy import ContextualDecisionPolicy
from character.decision_policy import DecisionPolicy
from character.models import CharacterProfile, InteractionState, RelationshipState, SituationState, WeightedSignal

PROFILE = CharacterProfile("role", "测试角色", response_preferences=("brief_self_disclosure",))
STATE = InteractionState(
    primary_situation="factual",
    situation_scores=(WeightedSignal("factual", 0.9),),
    user_acts=(WeightedSignal("information_request", 0.95),),
    confidence=0.9,
)
QUERY = "已知2、4、6，请计算它们的平均值。"


@pytest.mark.asyncio
async def test_normal_semantic_path_skips_unused_ranking_and_keeps_final_context():
    legacy = DecisionPolicy().decide(PROFILE, RelationshipState(), "factual", interaction=STATE)
    deferred = DecisionPolicy(defer_optional_strategies=True)
    deferred._soft_decision = Mock(side_effect=AssertionError("Discarded optional scorer must not run"))
    baseline = deferred.decide(PROFILE, RelationshipState(), "factual", interaction=STATE)
    assert baseline.selection_source == "pending_semantic" and baseline.strategy_ids == ()

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and payload["profile"]["response_preferences"] == ["brief_self_disclosure"]
        assert payload["required_strategies"] == ["respond_directly"]
        return '{"strategy_ids":["respond_directly","brief_self_disclosure"]}'

    kwargs = dict(
        query=QUERY,
        history=(),
        profile=PROFILE,
        relationship=RelationshipState(),
        interaction=STATE,
        has_relevant_memory=False,
    )
    policy = ContextualDecisionPolicy(reviewer)
    old = await policy.refine(legacy, **kwargs)
    new = await policy.refine(baseline, **kwargs)
    assert new.plan.strategy_ids == old.plan.strategy_ids == ("respond_directly", "brief_self_disclosure")
    context = lambda plan: compile_dynamic_context(RelationshipState(), SituationState(), plan, STATE)
    assert context(new.plan) == context(old.plan)
    deferred._soft_decision.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_failure_is_not_replaced_by_optional_scores():
    decision = DecisionPolicy(defer_optional_strategies=True)
    decision._soft_decision = Mock(side_effect=AssertionError("No fallback scoring"))
    baseline = decision.decide(PROFILE, RelationshipState(), "factual", interaction=STATE)
    failure = RuntimeError("synthetic semantic failure")

    async def reviewer(messages):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        await ContextualDecisionPolicy(reviewer).refine(
            baseline,
            query=QUERY,
            history=(),
            profile=PROFILE,
            relationship=RelationshipState(),
            interaction=STATE,
            has_relevant_memory=False,
        )
    assert caught.value is failure
    decision._soft_decision.assert_not_called()


@pytest.mark.parametrize("score", [0.2, 0.6])
def test_safety_related_signals_keep_existing_selection(score):
    state = InteractionState(
        primary_situation="emotional",
        situation_scores=(WeightedSignal("emotional", 0.8),),
        user_needs=(WeightedSignal("safety_clarification", score),),
        confidence=0.8,
    )
    legacy = DecisionPolicy().decide(PROFILE, RelationshipState(), "emotional", interaction=state)
    deferred = DecisionPolicy(defer_optional_strategies=True)
    original = deferred._soft_decision
    deferred._soft_decision = Mock(wraps=original)
    assert deferred.decide(PROFILE, RelationshipState(), "emotional", interaction=state) == legacy
    deferred._soft_decision.assert_called_once()


@pytest.mark.parametrize("defer", [False, True])
def test_hard_safety_retains_priority_even_with_invalid_primary(defer):
    state = InteractionState(primary_situation="invalid", safety_triggered=True)
    plan = DecisionPolicy(defer_optional_strategies=defer).decide(
        PROFILE, RelationshipState(), "invalid", interaction=state
    )
    assert plan.strategy_ids == ("ensure_safety",)


@pytest.mark.parametrize("defer", [False, True])
def test_unknown_non_safety_primary_does_not_become_daily(defer):
    with pytest.raises(KeyError):
        DecisionPolicy(defer_optional_strategies=defer).decide(PROFILE, RelationshipState(), "invalid")


def test_default_public_mode_keeps_existing_optional_selection():
    policy = DecisionPolicy()
    original = policy._soft_decision
    policy._soft_decision = Mock(wraps=original)
    plan = policy.decide(PROFILE, RelationshipState(), "factual", interaction=STATE)
    assert plan.strategy_ids and plan.selection_source != "pending_semantic"
    policy._soft_decision.assert_called_once()
