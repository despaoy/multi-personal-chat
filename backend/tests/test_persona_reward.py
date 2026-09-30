import copy

import pytest

from training.persona_reward import DIMENSIONS, VIOLATIONS, aggregate_reward, calibrate_pairs


def judgment(value=0.8):
    return {
        "judge_id": "calibrated-later-v1",
        "scores": dict.fromkeys(DIMENSIONS, value),
        "violations": dict.fromkeys(VIOLATIONS, False),
    }


def test_hard_violation_cannot_be_offset_by_perfect_style():
    row = judgment(1)
    assert aggregate_reward(row)["reward"] == pytest.approx(1)
    row["violations"]["fabricated_fact"] = True
    assert aggregate_reward(row)["reward"] == -1
    assert aggregate_reward(row)["hard_failures"] == ["fabricated_fact"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.8"])
def test_invalid_scores_fail_closed(value):
    row = judgment()
    row["scores"]["naturalness"] = value
    with pytest.raises(ValueError, match="finite"):
        aggregate_reward(row)


def test_missing_violation_checks_are_not_assumed_safe():
    row = judgment()
    del row["violations"]["fabricated_fact"]
    with pytest.raises(ValueError, match="checks"):
        aggregate_reward(row)


def calibration():
    return {
        "id": "pair1",
        "split": "validation",
        "human_approved": True,
        "source_group": "scene1",
        "human_preference": "a",
        "a": judgment(0.9),
        "b": judgment(0.2),
    }


def test_calibration_counts_human_disagreement_and_ties_without_approving_ppo():
    a, b, c = [calibration() for _ in range(3)]
    b.update(id="pair2", source_group="scene2", human_preference="b")
    c.update(id="pair3", source_group="scene3", human_preference="tie", b=judgment(0.9))
    result = calibrate_pairs([a, b, c])
    assert result["agreement"] == pytest.approx(2 / 3)
    assert result["judge_ties"] == 1
    assert result["status"] == "measured_not_approved_for_ppo"


def test_calibration_rejects_unreviewed_or_test_pairs_and_mixed_judges():
    row = calibration()
    for change in ({"human_approved": False}, {"split": "test"}):
        with pytest.raises(ValueError, match="human-approved"):
            calibrate_pairs([{**row, **change}])
    row["b"]["judge_id"] = "other"
    with pytest.raises(ValueError, match="same judge"):
        calibrate_pairs([row])


def test_reward_weights_are_explicit_and_do_not_mutate_judgment():
    row = judgment()
    original = copy.deepcopy(row)
    weights = dict.fromkeys(DIMENSIONS, 0)
    weights["grounding"] = 1
    assert aggregate_reward(row, weights)["reward"] == pytest.approx(0.6)
    assert original == row
    with pytest.raises(ValueError, match="weights"):
        aggregate_reward(row, dict.fromkeys(DIMENSIONS, 0))
