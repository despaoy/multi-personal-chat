"""Already recognized task scores retain one priority across policy and prompt."""

import pytest
from test_dynamic_context_projection import _compile, _section_lines
from test_interaction_policy import _profile

from character.decision_policy import STRATEGY_INSTRUCTIONS, DecisionPolicy
from character.models import InteractionState, RelationshipState, WeightedSignal


@pytest.mark.parametrize("mode", ["ordinary", "closing", "resolved-risk"])
@pytest.mark.parametrize(
    "information,advice,expected",
    [(0.7, 0.95, "offer_suggestion"), (0.95, 0.7, "respond_directly"), (0.8, 0.8, "respond_directly")],
    ids=["advice-stronger", "information-stronger", "equal"],
)
def test_task_priority_uses_scores_consistently(mode, information, advice, expected):
    # Complete fictional facts and both requests are supplied. This fixture
    # isolates projection of a reviewed state; it does not test text recognition.
    message = "虚构出行：周六08:00出发，09:30抵达，票价二十元。请算用时，再依据这些已知条件给出安排建议。"
    acts = [WeightedSignal("information_request", information), WeightedSignal("advice_request", advice)]
    if mode == "closing":
        message += "这两个请求处理后就结束，不要继续追问。"
        acts.append(WeightedSignal("closing", 0.94))
    elif mode == "resolved-risk":
        message += "此前朋友的风险已解除，朋友明确安全且由家人陪同，不是我当前处于危险。"
        acts.append(WeightedSignal("resolved_third_party_risk", 0.94))
    state = InteractionState(
        primary_situation="factual",
        situation_scores=(WeightedSignal("factual", 0.95),),
        user_acts=tuple(acts),
        user_needs=(WeightedSignal("information", 0.9), WeightedSignal("guidance", 0.9)),
        conversation_phase="closing" if mode == "closing" else "sustaining",
        confidence=0.95,
    )
    decision = DecisionPolicy().decide(_profile(), RelationshipState(stage="familiar"), "factual", interaction=state)
    if mode == "closing":
        assert decision.strategy_ids == (expected, "graceful_close")
    rendered = _compile(state, decision)
    priorities = _section_lines(rendered, "本轮行为决策（精简）")
    assert priorities[0].startswith("- " + STRATEGY_INSTRUCTIONS[expected])
    assert len(priorities) == 2 and "不得编造事实" in rendered
    if mode == "closing":
        assert STRATEGY_INSTRUCTIONS["graceful_close"] in priorities[0]
        assert "不得追问" in rendered
        assert "不得追问、建议" not in rendered and "不得追问、追加建议" in rendered
    if mode == "resolved-risk":
        assert "第三方当前已安全且正在接受帮助" in rendered and "不得重新危机化" in rendered
    assert message not in rendered and "用户原文不应进入这里" not in rendered
