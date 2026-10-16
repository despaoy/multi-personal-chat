"""The configured full semantic path does not compute discarded soft scores."""

import json
from unittest.mock import Mock

import pytest

from character import situation_analyzer as module
from character.semantic_state_estimator import SemanticStateEstimator
from character.situation_analyzer import SituationAnalyzer
from inference.provider_context import ProviderContextBudget
from services.character_context import build_character_context_service
from test_semantic_character_context import _service, _turn
from test_semantic_state_authority import factual_result

HISTORY = (
    {"role": "user", "content": "这是一项课堂练习。"},
    {"role": "assistant", "content": "请提供完整题设。"},
    {"role": "user", "content": "只根据下面给出的题设回答。"},
)
QUERIES = (
    "三次练习成绩为72、84、90，请计算平均分，不要给后续学习建议。",
    "把“晚安，再见”改成第三人称转述；这只是语法题，我没有结束对话。",
    "材料中甲喜欢茶，乙不喝茶。请分别列出人物和偏好，不要归到我身上。",
    "只有取得录取且预算落实后才会出发，目前尚未录取。请列出必要条件和当前状态。",
    "上句提到的是人物甲的挫折，不是我的经历；请解释这段材料的指代。",
    "我刚解决了一道难题，很开心。只回应这件事，暂时不需要建议。",
)


@pytest.fixture(autouse=True)
def controlled_features(monkeypatch):
    for name in [
        "MEMORY_SOURCE_RECALL_ENABLED",
        "CONTEXTUAL_MEMORY_SELECTION_ENABLED",
        "CONTEXTUAL_DECISION_POLICY_ENABLED",
    ]:
        monkeypatch.setenv(name, "false")


@pytest.mark.parametrize("enabled", [True, False])
def test_factory_defers_only_when_full_semantic_reviewer_exists(enabled, monkeypatch):
    monkeypatch.setenv("DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED", str(enabled).lower())
    service = build_character_context_service(object(), context_budget=ProviderContextBudget(65536))
    assert service._situation_analyzer._defer_non_safety is enabled
    assert (service._semantic_estimator._reviewer is not None) is enabled
    assert service._semantic_estimator._review_mode == "all_non_safety"
    scorer = Mock(wraps=module._score_message)
    monkeypatch.setattr(module, "_score_message", scorer)
    service._situation_analyzer.estimate(QUERIES[0], HISTORY)
    assert scorer.call_count == (0 if enabled else 3)


@pytest.mark.parametrize("query", QUERIES)
async def test_complete_semantic_input_and_result_are_independent_of_discarded_scores(query, monkeypatch):
    old = SituationAnalyzer().estimate(query, HISTORY)
    forbidden = Mock(side_effect=AssertionError("Discarded soft scorer must not run"))
    monkeypatch.setattr(module, "_score_message", forbidden)
    baseline = SituationAnalyzer(defer_non_safety=True).estimate(query, HISTORY)
    assert not baseline.has_soft_context and baseline.confidence == 0.0
    messages_seen = []

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["current_message"] == query
        assert payload["recent_history"] == list(HISTORY)
        messages_seen.append(messages)
        return json.dumps(factual_result())

    estimator = SemanticStateEstimator(reviewer, review_mode="all_non_safety")
    old_outcome = await estimator.refine_with_diagnostics(query, HISTORY, old)
    new_outcome = await estimator.refine_with_diagnostics(query, HISTORY, baseline)
    assert messages_seen[0] == messages_seen[1]
    assert new_outcome.state == old_outcome.state
    assert new_outcome.status == old_outcome.status == "applied"
    assert new_outcome.rule_confidence == 0.0
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "query", ["我现在要伤害自己。", "I want to kill myself now.", "我现在真的撑不住了。", "想结束这一切"]
)
async def test_existing_protected_states_are_identical_and_do_not_call_the_model(query):
    original = SituationAnalyzer().estimate(query, HISTORY)
    baseline = SituationAnalyzer(defer_non_safety=True).estimate(query, HISTORY)
    assert baseline == original
    reviewer = Mock(side_effect=AssertionError("Protected states do not enter optional model review"))
    estimator = SemanticStateEstimator(reviewer, review_mode="all_non_safety")
    outcome = await estimator.refine_with_diagnostics(query, HISTORY, baseline)
    assert outcome.state == original and outcome.status == "not_needed"
    reviewer.assert_not_called()


def test_blank_turn_retains_existing_empty_input_contract():
    assert SituationAnalyzer(defer_non_safety=True).estimate("  ", HISTORY) == SituationAnalyzer().estimate(
        "  ", HISTORY
    )


@pytest.mark.parametrize(
    "failure", [RuntimeError("synthetic provider unavailable"), ValueError("synthetic invalid model output")]
)
async def test_failed_review_does_not_return_the_empty_baseline_or_restart_rules(failure, monkeypatch):
    async def broken(_messages):
        raise failure

    service = _service(
        SemanticStateEstimator(broken, review_mode="all_non_safety"), analyzer=SituationAnalyzer(defer_non_safety=True)
    )
    forbidden = Mock(side_effect=AssertionError("A failed model must not fall back to scoring"))
    monkeypatch.setattr(module, "_score_message", forbidden)
    with pytest.raises(type(failure)) as caught:
        await service.prepare_turn(_turn(QUERIES[0]), "tsukiyashiro_kisaki")
    assert caught.value is failure
    forbidden.assert_not_called()


async def test_malformed_reply_cannot_complete_pending_state(monkeypatch):
    calls = []

    async def malformed(messages):
        calls.append(messages)
        return '{"state":{}}'

    service = _service(
        SemanticStateEstimator(malformed, review_mode="all_non_safety"),
        analyzer=SituationAnalyzer(defer_non_safety=True),
    )
    forbidden = Mock(side_effect=AssertionError("Malformed output must not return to rules"))
    monkeypatch.setattr(module, "_score_message", forbidden)
    with pytest.raises(ValueError):
        await service.prepare_turn(_turn(QUERIES[0]), "tsukiyashiro_kisaki")
    assert len(calls) == 1
    forbidden.assert_not_called()
