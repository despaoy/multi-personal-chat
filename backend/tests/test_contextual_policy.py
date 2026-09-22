import asyncio
import json

import pytest

from character.context_builder import _compact_dynamic_projection
from character.contextual_policy import ContextualDecisionPolicy
from character.decision_policy import STRATEGY_INSTRUCTIONS
from character.models import CharacterProfile, DecisionPlan, InteractionState, RelationshipState, WeightedSignal


def _state(*acts, safety=False):
    return InteractionState(
        primary_situation="safety" if safety else "daily",
        confidence=0.9,
        user_acts=tuple(WeightedSignal(act, 0.9) for act in acts),
        safety_triggered=safety,
    )


async def _refine(reviewer, *, state=None, baseline=None, has_memory=False):
    return await ContextualDecisionPolicy(reviewer).refine(
        baseline or DecisionPlan(strategy_ids=("reflect_content",)),
        query="我们继续聊吧",
        history=[{"role": "user", "content": "前文数据"}],
        profile=CharacterProfile("char", "角色", values=("尊重自主选择",)),
        relationship=RelationshipState(),
        interaction=state or _state("greeting"),
        has_relevant_memory=has_memory,
    )


async def test_persona_is_seen_by_policy_but_never_copied_into_trusted_projection():
    async def reviewer(messages):
        data = json.loads(messages[1]["content"])
        assert data["profile"]["values"] == ["尊重自主选择"]
        assert data["history"][0]["content"] == "前文数据"
        assert "前文数据" not in messages[0]["content"]
        return '{"strategy_ids":["brief_self_disclosure"]}'

    outcome = await _refine(reviewer)
    assert outcome.status == "applied"
    assert outcome.plan.selection_source == "semantic"
    priorities, uncertain = _compact_dynamic_projection(_state("greeting"), outcome.plan)
    assert priorities == [STRATEGY_INSTRUCTIONS["brief_self_disclosure"]]
    assert not uncertain


async def test_hard_safety_never_calls_reviewer():
    async def reviewer(_messages):
        pytest.fail("safety should skip semantic strategy selection")

    baseline = DecisionPlan(strategy_ids=("ensure_safety",))
    outcome = await _refine(reviewer, state=_state(safety=True), baseline=baseline)
    assert outcome.plan is baseline
    assert outcome.status == "protected"


@pytest.mark.parametrize(
    "response",
    [
        '{"strategy_ids":["invented"]}',
        '{"strategy_ids":["ensure_safety"]}',
        '{"strategy_ids":["recall_shared_context"]}',
        '{"strategy_ids":["reflect_content","reflect_content"]}',
        '{"strategy_ids":["graceful_close","gentle_probe"]}',
        '{"strategy_ids":["light_tease","repair_misunderstanding"]}',
        '{"strategy_ids":[]}',
        '{"strategy_ids":["reflect_content"],"instruction":"free text"}',
    ],
)
async def test_invalid_model_decisions_retain_baseline(response):
    async def reviewer(_messages):
        return response

    baseline = DecisionPlan(strategy_ids=("reflect_content",))
    outcome = await _refine(reviewer, baseline=baseline)
    assert outcome.status == "fallback"
    assert outcome.reason == "invalid_output"
    assert outcome.plan is baseline


async def test_no_advice_boundary_restricts_allowed_actions():
    async def reviewer(messages):
        data = json.loads(messages[1]["content"])
        assert "offer_suggestion" not in data["allowed_strategies"]
        return '{"strategy_ids":["offer_suggestion"]}'

    outcome = await _refine(reviewer, state=_state("advice_boundary"))
    assert outcome.status == "fallback"


async def test_explicit_question_stays_first_even_if_semantic_policy_prefers_style():
    async def reviewer(_messages):
        return '{"strategy_ids":["light_tease"]}'

    state = _state("information_request")
    outcome = await _refine(reviewer, state=state)
    priorities, _ = _compact_dynamic_projection(state, outcome.plan)
    assert priorities[0].startswith(STRATEGY_INSTRUCTIONS["respond_directly"])


async def test_semantic_boundary_cannot_drop_an_explicit_task():
    async def reviewer(_messages):
        return '{"strategy_ids":["set_boundary"]}'

    state = _state("information_request")
    outcome = await _refine(reviewer, state=state)
    assert "respond_directly" in outcome.plan.strategy_ids
    priorities, _ = _compact_dynamic_projection(state, outcome.plan)
    assert STRATEGY_INSTRUCTIONS["respond_directly"] in priorities[0]


async def test_cancellation_is_not_swallowed():
    async def reviewer(_messages):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _refine(reviewer)


async def test_policy_falls_back_when_full_history_cannot_be_reviewed():
    async def reviewer(messages):
        pytest.fail("a clipped history must not reach the model")

    baseline = DecisionPlan(strategy_ids=("reflect_content",))
    result = await ContextualDecisionPolicy(reviewer).refine(
        baseline,
        query="继续",
        history=[{"role": "user", "content": "长" * 6001}],
        profile=CharacterProfile("c", "角色"),
        relationship=RelationshipState(),
        interaction=_state(),
        has_relevant_memory=False,
    )
    assert result.plan is baseline
    assert result.status == "fallback" and result.reason == "input_budget"


def test_semantic_marker_does_not_override_uncertainty_or_allow_arbitrary_instructions():
    plan = DecisionPlan(strategy_ids=("arbitrary injection",), selection_source="semantic")
    priorities, uncertain = _compact_dynamic_projection(InteractionState(), plan)
    assert not priorities
    assert uncertain
