import json

import pytest
from scripts.evaluate_contextual_policy import constraint_violations, evaluate


def test_constraints_do_not_treat_persona_difference_as_correctness():
    assert constraint_violations(["reflect_content"], [], safety_required=True) == ["required_safety_route_missing"]
    assert constraint_violations(["ensure_safety"], [], safety_required=True) == []
    assert constraint_violations(["reflect_content"], []) == []
    assert constraint_violations(["light_tease"], []) == []
    assert constraint_violations(["gentle_probe"], ["advice_boundary"]) == ["no_advice_boundary_violated"]
    assert constraint_violations(["reflect_content"], ["information_request"]) == ["explicit_question_dropped"]
    assert constraint_violations(["gentle_probe"], ["closing"]) == ["conversation_close_violated"]


async def test_two_profiles_are_evaluated_without_gold_or_expected_strategy():
    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        assert "expected" not in payload
        assert "gold" not in payload
        return '{"strategy_ids":["reflect_content"]}'

    result = await evaluate([{"id": "q", "query": "你好", "situation": "daily", "acts": ["greeting"]}], reviewer)
    assert result["summary"]["decisions"] == 2
    assert result["summary"]["comparable_persona_pairs"] == 1
    assert result["summary"]["different_strategy_pairs"] == 0
    assert result["summary"]["constraint_violations"] == 0
    with pytest.raises(ValueError, match="unknown"):
        await evaluate([{"id": "bad", "query": "q", "situation": "daily", "acts": ["invented"]}], reviewer)


async def test_inferred_state_does_not_receive_fixture_gold():
    async def reviewer(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["state_hint"]["situation"] != "conflict"
        assert "advice_request" not in payload["state_hint"]["acts"]
        return '{"strategy_ids":["reflect_content"]}'

    # Deliberately conflicting gold stays evaluator-only; the rule path sees a greeting.
    result = await evaluate(
        [{"id": "q", "query": "你好", "situation": "conflict", "acts": ["advice_request"]}], reviewer, "rules"
    )
    assert result["supplied_state_hints"] is False
    assert result["summary"]["projection_constraint_violations"] == 2
    assert result["summary"]["baseline_projection_constraint_violations"] == 2
    assert result["state_diagnostics"]["case_count"] == 1
    assert result["state_diagnostics"]["cases_missing_required_acts"] == 1
    assert result["state_diagnostics"]["situation_matches"] == 0
    assert result["state_cases"][0]["missing_required_acts_at_0_5"] == ["advice_request"]


async def test_required_act_coverage_does_not_count_personas_twice():
    async def reviewer(messages):
        return '{"strategy_ids":["reflect_content"]}'

    result = await evaluate(
        [{"id": "q", "query": "你好", "situation": "daily", "acts": ["greeting"]}], reviewer
    )
    assert result["summary"]["decisions"] == 2
    assert result["state_diagnostics"]["case_count"] == 1
    assert result["state_diagnostics"]["cases_missing_required_acts"] == 0
    assert result["state_diagnostics"]["reviewed_cases"] == 0
