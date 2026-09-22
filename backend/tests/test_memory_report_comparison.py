import copy

import pytest
from scripts.compare_memory_reports import compare, validated_observations

CASES = [{"id": "a", "memories": [{"id": "m"}], "gold_ids": ["m"]}]
REPORT = {
    "input_sha256": "hash",
    "cases": [{"id": "a", "gold_ids": ["m"], "candidate_ids": ["m"], "selected_ids": ["m"], "status": "selected"}],
}


def test_comparison_recomputes_from_actual_outputs():
    wrong = copy.deepcopy(REPORT)
    wrong["cases"][0]["selected_ids"] = []
    wrong["metrics"] = {"exact_set_accuracy": 1}  # Do not trust summary fields.
    result = compare(CASES, {"old": wrong, "new": REPORT}, "hash")
    assert result["variants"]["old"]["metrics"]["exact_set_accuracy"] == 0
    assert result["comparisons"]["new"]["improved_case_ids"] == ["a"]


@pytest.mark.parametrize("change", ["hash", "gold", "coverage", "candidate"])
def test_incomparable_report_is_rejected(change):
    report = copy.deepcopy(REPORT)
    if change == "hash":
        report["input_sha256"] = "other"
    elif change == "gold":
        report["cases"][0]["gold_ids"] = []
    elif change == "coverage":
        report["cases"] = []
    else:
        report["cases"][0]["candidate_ids"] = ["invented"]
    with pytest.raises(ValueError):
        validated_observations(CASES, report, "hash")
