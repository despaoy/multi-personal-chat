import pytest

from evaluation.evidence_selection import SelectionObservation, paired_selection_metrics, selection_metrics


def test_abstaining_on_everything_cannot_appear_successful():
    rows = [
        SelectionObservation("positive", frozenset({"a"}), (), "fallback"),
        SelectionObservation("negative", frozenset(), (), "selected"),
    ]
    metrics = selection_metrics(rows)
    assert metrics["negative_case_injection_rate"] == 0
    assert metrics["answerable_miss_rate"] == 1
    assert metrics["recall_micro"] == 0
    assert metrics["precision_micro"] is None
    assert metrics["provider_failure_rate"] == 0.5


def test_micro_counts_and_empty_denominators():
    metrics = selection_metrics(
        [
            SelectionObservation("1", frozenset({"a", "b"}), ("a", "wrong")),
            SelectionObservation("2", frozenset(), ("wrong",)),
        ]
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["precision_micro"] == pytest.approx(1 / 3)
    assert metrics["recall_micro"] == 0.5
    assert metrics["negative_case_injection_rate"] == 1
    assert selection_metrics([])["exact_set_accuracy"] is None


def test_duplicate_cases_and_selected_ids_rejected():
    row = SelectionObservation("1", frozenset(), ())
    with pytest.raises(ValueError):
        selection_metrics([row, row])
    with pytest.raises(ValueError):
        selection_metrics([SelectionObservation("1", frozenset(), ("a", "a"))])


def test_recall_loss_is_separated_from_selector_loss():
    row = SelectionObservation("case", frozenset({"a", "b", "c"}), ("a",), candidate_ids=("a", "b", "noise"))
    metrics = selection_metrics([row])
    assert metrics["candidate_gold_recall"] == pytest.approx(2 / 3)
    assert metrics["selector_recall_given_available_gold"] == 0.5
    assert metrics["gold_missing_before_selection"] == 1
    assert metrics["available_gold_dropped_by_selection"] == 1


def test_negative_provider_failure_does_not_count_as_successful_exact():
    metrics = selection_metrics([SelectionObservation("case", frozenset(), (), "fallback")])
    assert metrics["exact_set_accuracy"] == 1
    assert metrics["successful_exact_set_accuracy"] == 0


@pytest.mark.parametrize("candidates", [("a", "a"), ("b",)])
def test_invalid_recall_diagnostics_rejected(candidates):
    with pytest.raises(ValueError):
        selection_metrics([SelectionObservation("case", frozenset(), ("a",), candidate_ids=candidates)])


def test_pair_accuracy_requires_both_correct_not_merely_different():
    rows = [SelectionObservation("a", frozenset({"x"}), ("wrong",)), SelectionObservation("b", frozenset(), ())]
    result = paired_selection_metrics(rows, {"a": "pair", "b": "pair"})
    assert result["both_correct_rate"] == 0
    assert result["details"][0]["expected_change"] is True
    assert result["details"][0]["observed_change"] is True


def test_invariance_pair_and_failure_are_distinguished():
    rows = [SelectionObservation("a", frozenset(), ()), SelectionObservation("b", frozenset(), ())]
    assert paired_selection_metrics(rows, {"a": "pair", "b": "pair"})["both_correct_rate"] == 1
    rows[1] = SelectionObservation("b", frozenset(), (), "fallback")
    assert paired_selection_metrics(rows, {"a": "pair", "b": "pair"})["both_correct_rate"] == 0


@pytest.mark.parametrize("mapping", [{"a": "p"}, {"a": ""}, {"absent": "p"}])
def test_incomplete_or_unknown_pair_rejected(mapping):
    with pytest.raises(ValueError):
        paired_selection_metrics([SelectionObservation("a", frozenset(), ())], mapping)
