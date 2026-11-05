"""Advice refusal cannot cancel a simultaneous explicit information task."""

import pytest
from test_interaction_policy import _profile

from character.context_builder import compile_dynamic_context
from character.decision_policy import STRATEGY_INSTRUCTIONS, DecisionPolicy
from character.models import InteractionState, RelationshipState, SituationState, WeightedSignal
from character.situation_analyzer import SITUATION_EMOTIONAL, SITUATION_FACTUAL, SITUATION_META, SITUATION_SAFETY

SCOPED = "只回答本轮仍明确要求的问题；不追加用户已拒绝的建议、分析或追问，也不以追问变相推进这些内容。"


def project(primary, *, information, safety=False):
    query = "完整虚构票价：三张票，每张两元。请解释计算过程；只拒绝额外健康建议，不拒绝这道题的解释。"
    if primary == SITUATION_META:
        query = "先说明你公开的人物身份，再解释完整虚构题：三张票、每张两元；不索取内部资料，只拒绝额外健康建议。"
    elif safety:
        query = "虚构紧急场景：当前存在明确自伤危险，需要确认即时安全；拒绝建议不能覆盖即时安全处理。"
    elif not information:
        query = "我今天有点累，只想安静待一会儿；不要建议、不分析，也别追问。"
    acts = (WeightedSignal("advice_boundary", 0.96),)
    if information:
        acts += (WeightedSignal("information_request", 0.92),)
    state = InteractionState(
        primary_situation=primary,
        situation_scores=(WeightedSignal(primary, 0.94),),
        user_acts=acts,
        user_needs=(WeightedSignal("information" if information else "companionship", 0.9),),
        confidence=0.94,
        safety_triggered=safety,
    )
    relationship = RelationshipState(stage="familiar")
    plan = DecisionPolicy().decide(_profile(), relationship, primary, interaction=state)
    rendered = compile_dynamic_context(
        relationship,
        SituationState(topic=query, emotion_hint=query, response_goal=query),
        plan,
        state,
    )
    return query, state, plan, rendered


@pytest.mark.parametrize("primary", [SITUATION_FACTUAL, SITUATION_META])
def test_explicit_information_keeps_its_scope_in_both_policy_and_projection(primary):
    query, state, plan, rendered = project(primary, information=True)
    assert SCOPED in plan.avoid and SCOPED in rendered
    assert "不得提供建议、分析方案" not in rendered
    assert STRATEGY_INSTRUCTIONS["respond_directly"] in rendered
    assert STRATEGY_INSTRUCTIONS["stay_present"] not in rendered
    assert "证据不足就明确说明" in rendered and "不得编造事实" in rendered
    assert {signal.signal_id for signal in state.user_acts} == {"advice_boundary", "information_request"}
    assert query not in rendered


@pytest.mark.parametrize("safety", [False, True], ids=["quiet-company", "immediate-safety"])
def test_quiet_refusal_and_immediate_safety_do_not_lose_their_existing_constraints(safety):
    primary = SITUATION_SAFETY if safety else SITUATION_EMOTIONAL
    query, _, plan, rendered = project(primary, information=safety, safety=safety)
    assert SCOPED not in rendered and query not in rendered
    if safety:
        assert plan.strategy_ids == ("ensure_safety",)
        assert STRATEGY_INSTRUCTIONS["ensure_safety"] in rendered
        assert "不提供伤害自己的具体操作信息" in rendered
    else:
        assert "stay_present" in plan.strategy_ids
        assert "不得提供建议、分析方案" in plan.avoid and "不得提供建议、分析方案" in rendered
        assert "本轮不得追问" in plan.avoid and "本轮不得追问" in rendered
        assert STRATEGY_INSTRUCTIONS["respond_directly"] not in rendered
