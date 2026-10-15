"""Required task slots do not spend an ineffectual policy model call.

Reuse the committed policy/circuit fixtures: real profile JSON, SQLite,
state parsing, baseline policy, service, API and final prompt compilation.
State/answer semantics are synthetic; no paid model calls are used.
"""

import json

import pytest
import test_policy_strategy_consistency as existing_policy
from test_circuit_strict_generation import circuit_rig as circuit_rig

from character.decision_policy import STRATEGY_INSTRUCTIONS
from scripts.evaluate_contextual_policy import evaluate

pytestmark = pytest.mark.usefixtures("no_live_context_review")
BOTH_QUERY = (
    "完整题设：我计划本周四14:00至15:00排练，尚未报名；"
    "只有取得书面确认且当天不下雨才参加，预约不豁免雨天限制。"
    "请先告诉我题设中哪些已确认、哪些尚未确认，再给我一至两项准备建议。两项都要回答并保留完整条件。"
)
BOTH_ANSWER = (
    "已确认的是周四14:00至15:00的计划时间，尚未报名，未证明已获书面确认或天气满足条件。"
    "先确认书面报名流程，再检查周四天气；未获确认或雨天均不参加。"
)


def arrange_required_tasks(tmp_path, monkeypatch, circuit_rig):
    monkeypatch.setattr(existing_policy, "QUERY", BOTH_QUERY)
    monkeypatch.setattr(existing_policy, "ANSWER", BOTH_ANSWER)
    rig = existing_policy.arrange_policy_conflict(
        tmp_path, monkeypatch, circuit_rig, ("respond_directly", "offer_suggestion"),
    )

    async def state_reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["current_message"] == BOTH_QUERY and payload["recent_history"] == []
        rig.state_calls.append(payload)
        return json.dumps({"state": {
            "primary_situation": "factual", "situation_scores": {"factual": .9},
            "user_acts": {"information_request": .9, "advice_request": .9},
            "user_needs": {"information": .9, "guidance": .9},
            "valence": 0.0, "arousal": .2, "warmth": 0.0, "face_threat": 0.0,
            "conversation_phase": "sustaining", "confidence": .9,
        }})

    async def forbidden_policy(messages):
        rig.policy_calls.append(json.loads(messages[-1]["content"]))
        raise AssertionError("Both task slots are fixed; policy reviewer must not run")

    # Control only these two model boundaries. The actual state parser and
    # policy's required-strategy calculation/merge remain unchanged.
    rig.service._semantic_estimator._reviewer = state_reviewer
    rig.service._contextual_policy.reviewer = forbidden_policy
    return rig


@pytest.mark.asyncio
async def test_two_required_tasks_skip_policy_review_and_reach_real_system_prompt(tmp_path, monkeypatch, circuit_rig):
    rig = arrange_required_tasks(tmp_path, monkeypatch, circuit_rig)
    result = await existing_policy.api_turn(rig)
    assert result.reply == BOTH_ANSWER and result.modelInvoked is True
    assert len(rig.state_calls) == len(rig.baselines) == len(rig.prepared) == len(rig.calls) == 1
    assert rig.policy_calls == []
    prepared = rig.prepared[0]
    assert prepared.semantic_review_status == "applied"  # Real parser used the controlled state response.
    assert prepared.contextual_policy_status == "protected"
    assert prepared.contextual_policy_reason == "required_tasks"
    assert prepared.decision.selection_source == "required"
    assert prepared.decision.strategy_ids == ("respond_directly", "offer_suggestion")
    acts = {signal.signal_id: signal.score for signal in prepared.interaction.user_acts}
    assert acts["information_request"] >= .5 and acts["advice_request"] >= .5
    system = "\n".join(message["content"] for message in rig.calls[0]["messages"] if message["role"] == "system")
    assert STRATEGY_INSTRUCTIONS["respond_directly"] in system
    assert STRATEGY_INSTRUCTIONS["offer_suggestion"] in system
    assert any(BOTH_QUERY in message["content"] for message in rig.calls[0]["messages"] if message["role"] == "user")


@pytest.mark.asyncio
async def test_one_required_task_still_reviews_the_optional_slot_once(tmp_path, monkeypatch, circuit_rig):
    rig = existing_policy.arrange_policy_conflict(
        tmp_path, monkeypatch, circuit_rig, ("offer_suggestion", "acknowledge_emotion"),
    )
    result = await existing_policy.api_turn(rig)
    assert result.reply == existing_policy.ANSWER and result.modelInvoked is True
    assert len(rig.state_calls) == len(rig.policy_calls) == len(rig.prepared) == len(rig.calls) == 1
    prepared = rig.prepared[0]
    assert prepared.contextual_policy_status == "applied" and prepared.contextual_policy_reason == ""
    assert prepared.decision.selection_source == "semantic"
    assert prepared.decision.strategy_ids == ("offer_suggestion", "acknowledge_emotion")
    system = "\n".join(message["content"] for message in rig.calls[0]["messages"] if message["role"] == "system")
    assert STRATEGY_INSTRUCTIONS["offer_suggestion"] in system
    assert STRATEGY_INSTRUCTIONS["acknowledge_emotion"] in system


@pytest.mark.asyncio
async def test_required_task_projection_is_not_counted_as_fresh_persona_review():
    async def forbidden_policy(messages):
        pytest.fail("Supplied complete task state leaves no optional policy slot")

    report = await evaluate([
        {"id": "fiction-two-required", "query": BOTH_QUERY, "situation": "factual",
         "acts": ["information_request", "advice_request"]},
    ], forbidden_policy)
    assert len(report["cases"]) == report["summary"]["decisions"] == 2
    assert all(row["status"] == "protected" and row["reason"] == "required_tasks" for row in report["cases"])
    assert all(row["selected_strategy_ids"] == ["respond_directly", "offer_suggestion"] for row in report["cases"])
    assert report["summary"]["comparable_persona_pairs"] == report["summary"]["different_strategy_pairs"] == 0
    assert report["summary"]["constraint_violations"] == report["summary"]["projection_constraint_violations"] == 0
