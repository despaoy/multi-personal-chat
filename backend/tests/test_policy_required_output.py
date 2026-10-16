"""The reviewer owns its complete strategy list; no task insertion or truncation."""

import json

import pytest

from character.contextual_policy import ContextualDecisionPolicy
from character.models import CharacterProfile, DecisionPlan, InteractionState, RelationshipState, WeightedSignal


async def refine(ids, *, advice=False):
    required = "offer_suggestion" if advice else "respond_directly"
    query = "请建议如何核对2、4、6的平均值算式。" if advice else "已知2、4、6，请计算它们的平均值。"
    calls = []

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        assert payload["query"] == query and payload["required_strategies"] == [required]
        return json.dumps(dict(strategy_ids=ids))

    outcome = await ContextualDecisionPolicy(reviewer).refine(
        DecisionPlan(strategy_ids=(required,)),
        query=query,
        history=(),
        profile=CharacterProfile("role", "测试角色"),
        relationship=RelationshipState(),
        interaction=InteractionState(
            primary_situation="factual",
            user_acts=(WeightedSignal("advice_request" if advice else "information_request", 0.95),),
        ),
        has_relevant_memory=False,
    )
    assert len(calls) == 1
    return outcome


@pytest.mark.parametrize("advice", [False, True])
async def test_valid_complete_choice_is_preserved_without_rewriting(advice):
    required = "offer_suggestion" if advice else "respond_directly"
    ids = [required, "acknowledge_emotion"]
    outcome = await refine(ids, advice=advice)
    assert outcome.status == "applied" and outcome.plan.strategy_ids == tuple(ids)


@pytest.mark.parametrize(
    "ids", [["reflect_content"], ["acknowledge_emotion", "reflect_content"], ["reflect_content", "respond_directly"]]
)
async def test_omitted_or_misordered_task_is_error_instead_of_repaired_success(ids):
    with pytest.raises(ValueError, match="required_strategies"):
        await refine(ids)


async def test_two_explicit_tasks_still_skip_the_strategy_model():
    async def forbidden(messages):
        raise AssertionError("Both task slots are already fixed")

    state = InteractionState(
        primary_situation="factual",
        user_acts=(WeightedSignal("information_request", 0.95), WeightedSignal("advice_request", 0.95)),
    )
    outcome = await ContextualDecisionPolicy(forbidden).refine(
        DecisionPlan(),
        query="请计算2、4、6的平均值，并建议如何核对算式。",
        history=(),
        profile=CharacterProfile("role", "测试角色"),
        relationship=RelationshipState(),
        interaction=state,
        has_relevant_memory=False,
    )
    assert outcome.status == "protected" and outcome.reason == "required_tasks"
    assert outcome.plan.strategy_ids == ("respond_directly", "offer_suggestion")
