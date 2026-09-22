import copy

import pytest
from scripts.aggregate_memory_seeds import aggregate

CASES = [{"id": "q", "memories": [{"id": "m"}], "gold_ids": ["m"]}]


def report(seed, correct=True):
    return {
        "input_sha256": "hash",
        "inference": {"seed": seed, "decoding": "sampled"},
        "selection_instruction_sha256": "prompt",
        "embedding_enabled": False,
        "cases": [
            {
                "id": "q",
                "gold_ids": ["m"],
                "candidate_ids": ["m"],
                "selected_ids": ["m"] if correct else [],
                "status": "selected",
            }
        ],
    }


def test_repeated_seed_runs_do_not_inflate_distinct_case_count():
    result = aggregate(CASES, [report(42), report(43, False), report(44)], "hash")
    assert result["distinct_cases"] == 1
    assert result["repeated_case_evaluations"] == 3
    assert result["mean_successful_exact_accuracy"] == pytest.approx(2 / 3)
    assert result["correct_all_seeds"] == 0
    assert result["varying_case_ids"] == ["q"]
    assert result["posthoc_majority_vote"]["metrics"]["successful_exact_set_accuracy"] == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "recipe", "recall", "candidate_view"])
def test_seed_cherry_picking_or_recipe_changes_rejected(change):
    rows = [report(42), report(43), report(44)]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[2] = copy.deepcopy(rows[0])
    elif change == "recipe":
        rows[2]["inference"]["enable_thinking"] = True
    elif change == "candidate_view":
        rows[2]["candidate_view"] = "status_neutral"
    else:
        rows[2]["cases"][0].update(candidate_ids=[], selected_ids=[])
    with pytest.raises(ValueError):
        aggregate(CASES, rows, "hash")


def test_one_good_vote_cannot_override_two_completed_negative_votes():
    result = aggregate(CASES, [report(42), report(43, False), report(44, False)], "hash")
    voted = result["posthoc_majority_vote"]
    assert voted["cases"][0]["selected_ids"] == []
    assert voted["metrics"]["provider_failures"] == 0
    assert voted["metrics"]["false_negative"] == 1


def test_two_failed_runs_are_not_counted_as_successful_consensus_abstention():
    cases = [{"id": "q", "memories": [{"id": "m"}], "gold_ids": []}]
    rows = [report(42, False), report(43, False), report(44, False)]
    for row in rows:
        row["cases"][0]["gold_ids"] = []
    for row in rows[1:]:
        row["cases"][0]["status"] = "fallback"
    result = aggregate(cases, rows, "hash")["posthoc_majority_vote"]
    assert result["metrics"]["provider_failures"] == 1
    assert result["metrics"]["successful_exact_set_accuracy"] == 0


def test_two_valid_matching_runs_can_survive_one_failed_run():
    rows = [report(42), report(43, False), report(44)]
    rows[1]["cases"][0]["status"] = "fallback"
    result = aggregate(CASES, rows, "hash")["posthoc_majority_vote"]
    assert result["metrics"]["provider_failures"] == 0
    assert result["cases"][0]["selected_ids"] == ["m"]
    assert result["cases"][0]["completed_runs"] == 2


def test_exact_set_vote_does_not_synthesize_a_never_proposed_combination():
    cases = [{"id": "q", "memories": [{"id": key} for key in "abc"], "gold_ids": ["a"]}]
    rows = [report(42), report(43), report(44)]
    for row, selected in zip(rows, [["a", "b"], ["a", "c"], ["b", "c"]], strict=True):
        row["cases"][0].update(candidate_ids=list("abc"), gold_ids=["a"], selected_ids=selected)
    result = aggregate(cases, rows, "hash")
    assert result["posthoc_majority_vote"]["cases"][0]["selected_ids"] == list("abc")
    strict = result["posthoc_exact_set_vote"]
    assert strict["cases"][0]["status"] == "fallback"
    assert strict["metrics"]["consensus_failures"] == 1
    assert "provider_failures" not in strict["metrics"]
    assert strict["cases"][0]["fallback_reason"] == "no_set_consensus"
    assert strict["additional_model_calls_beyond_itemwise_replay"] == 0


def test_exact_set_empty_consensus_differs_from_failed_consensus():
    cases = [{"id": "q", "memories": [{"id": "m"}], "gold_ids": []}]
    rows = [report(42, False), report(43, False), report(44)]
    for row in rows:
        row["cases"][0]["gold_ids"] = []
    result = aggregate(cases, rows, "hash")["posthoc_exact_set_vote"]
    assert result["cases"][0]["status"] == "selected"
    assert result["metrics"]["successful_exact_set_accuracy"] == 1
    rows[1]["cases"][0]["status"] = "fallback"
    result = aggregate(cases, rows, "hash")["posthoc_exact_set_vote"]
    assert result["cases"][0]["status"] == "fallback"
    assert result["metrics"]["successful_exact_set_accuracy"] == 0


def test_vote_count_cannot_exceed_final_item_budget_and_costs_remain_auditable():
    cases = [{"id": "q", "memories": [{"id": str(i)} for i in range(7)], "gold_ids": []}]
    rows = [report(42), report(43), report(44)]
    for row, selected in zip(
        rows, [["0", "1", "2", "3", "4"], ["0", "1", "2", "5", "6"], ["3", "4", "5", "6"]], strict=True
    ):
        row["cases"][0].update(candidate_ids=[str(i) for i in range(7)], gold_ids=[], selected_ids=selected)
        row["inference_calls"] = [{"latency_seconds": 2.5}]
    result = aggregate(cases, rows, "hash")["posthoc_majority_vote"]
    assert len(result["cases"][0]["selected_ids"]) == 5
    assert result["model_calls_across_all_seeds"] == 3
    assert result["summed_generation_seconds"] == 7.5
    reversed_result = aggregate(cases, list(reversed(rows)), "hash")["posthoc_majority_vote"]
    assert result == reversed_result
