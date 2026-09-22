import json

import pytest
from scripts.evaluate_semantic_state import evaluate, validate_cases


async def test_state_fixture_gold_is_not_model_input_and_failure_is_preserved():
    seen = []

    async def reviewer(messages):
        seen.append(messages)
        assert "fixture_only_marker" not in json.dumps(messages)
        return "invalid"

    report = await evaluate(
        [{"id": "fixture_only_marker", "query": "你好", "situation": "factual", "acts": ["advice_request"]}],
        reviewer,
        include_model_output=True,
    )
    assert len(seen) == 1
    assert report["summary"]["fallbacks"] == 1
    assert report["summary"]["cases_missing_required_acts"] == 1
    assert report["cases"][0]["reviewed"] == report["cases"][0]["rules"]
    assert report["cases"][0]["model_final_outputs"] == ["invalid"]


async def test_safety_gate_bypasses_even_full_state_experiment():
    async def reviewer(messages):
        pytest.fail("safety must not call a reviewer")

    report = await evaluate(
        [
            {
                "id": "s",
                "query": "我现在有立即伤害自己的打算。",
                "situation": "safety",
                "acts": [],
                "safety_triggered": True,
            }
        ],
        reviewer,
    )
    assert report["summary"]["required_safety_misses"] == 0
    assert report["summary"]["reviewed_cases"] == 0
    assert "model_final_outputs" not in report["cases"][0]


async def test_unknown_required_labels_are_rejected_before_review():
    async def reviewer(messages):
        pytest.fail("invalid fixture must fail before inference")

    with pytest.raises(ValueError, match="unknown"):
        await evaluate([{"id": "x", "query": "q", "situation": "daily", "acts": ["unknown"]}], reviewer)


@pytest.mark.parametrize(
    "patch", [{"query": ""}, {"safety_triggered": "false"}, {"history": [{}]}, {"acts": ["greeting", "greeting"]}]
)
def test_input_validation_is_available_before_loading_model(patch):
    case = {"id": "x", "query": "q", "situation": "daily", "acts": ["greeting"], **patch}
    with pytest.raises(ValueError):
        validate_cases([case])
