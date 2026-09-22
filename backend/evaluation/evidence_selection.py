"""Evidence selection metrics, with explicit abstention and failure accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SelectionObservation:
    case_id: str
    gold_ids: frozenset[str]
    selected_ids: tuple[str, ...]
    status: str = "selected"
    category: str = "general"
    candidate_ids: tuple[str, ...] | None = None


def selection_metrics(observations: Sequence[SelectionObservation]) -> dict:
    if len({row.case_id for row in observations}) != len(observations):
        raise ValueError("duplicate evaluation case ID")
    true_positive = false_positive = false_negative = 0
    exact = 0
    negative_cases = false_injections = answerable = missed_answerable = failures = 0
    available_gold = total_retrieval_gold = selected_available_gold = 0
    retrieval_cases = successful_exact = 0
    for row in observations:
        if len(row.selected_ids) != len(set(row.selected_ids)):
            raise ValueError("duplicate selected ID")
        predicted = set(row.selected_ids)
        if row.status not in {"selected", "empty", "fallback", "error"}:
            raise ValueError("unknown selection status")
        if row.candidate_ids is not None:
            candidates = set(row.candidate_ids)
            if len(candidates) != len(row.candidate_ids) or not predicted <= candidates:
                raise ValueError("invalid candidate IDs or selected IDs outside recall")
            retrieval_cases += 1
            available_gold += len(row.gold_ids & candidates)
            total_retrieval_gold += len(row.gold_ids)
            selected_available_gold += len(predicted & row.gold_ids & candidates)
        true_positive += len(predicted & row.gold_ids)
        false_positive += len(predicted - row.gold_ids)
        false_negative += len(row.gold_ids - predicted)
        exact += predicted == row.gold_ids
        failures += row.status in {"fallback", "error"}
        successful_exact += predicted == row.gold_ids and row.status not in {"fallback", "error"}
        if not row.gold_ids:
            negative_cases += 1
            false_injections += bool(predicted)
        else:
            answerable += 1
            missed_answerable += not bool(predicted & row.gold_ids)

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    return {
        "cases": len(observations),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "exact_set_accuracy": ratio(exact, len(observations)),
        "irrelevant_selection_rate": ratio(false_positive, true_positive + false_positive),
        "negative_cases": negative_cases,
        "negative_case_injection_rate": ratio(false_injections, negative_cases),
        "answerable_cases": answerable,
        "answerable_miss_rate": ratio(missed_answerable, answerable),
        "provider_failures": failures,
        "provider_failure_rate": ratio(failures, len(observations)),
        "successful_exact_set_accuracy": ratio(successful_exact, len(observations)),
        "retrieval_diagnostic_cases": retrieval_cases,
        "candidate_gold_recall": ratio(available_gold, total_retrieval_gold),
        "selector_recall_given_available_gold": ratio(selected_available_gold, available_gold),
        "gold_missing_before_selection": total_retrieval_gold - available_gold,
        "available_gold_dropped_by_selection": available_gold - selected_available_gold,
    }


def paired_selection_metrics(observations: Sequence[SelectionObservation], pair_ids: dict[str, str]) -> dict:
    """Both members must succeed; merely changing the prediction earns no credit.

    Pairs are evaluator-only metadata and need not ask for opposite outcomes (e.g.
    adding an irrelevant injected instruction should preserve an empty selection).
    """
    selection_metrics(observations)  # Validate IDs/statuses with the same rules.
    by_id = {row.case_id: row for row in observations}
    if not set(pair_ids) <= set(by_id):
        raise ValueError("unknown paired case")
    groups = {}
    for case_id, pair_id in pair_ids.items():
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError("invalid pair ID")
        groups.setdefault(pair_id, []).append(by_id[case_id])
    if any(len(rows) != 2 for rows in groups.values()):
        raise ValueError("each pair must contain exactly two cases")
    details = []
    for pair_id, rows in sorted(groups.items()):
        both_correct = all(
            set(row.selected_ids) == row.gold_ids and row.status not in {"fallback", "error"} for row in rows
        )
        details.append(
            {
                "pair_id": pair_id,
                "case_ids": [row.case_id for row in rows],
                "expected_change": rows[0].gold_ids != rows[1].gold_ids,
                "observed_change": set(rows[0].selected_ids) != set(rows[1].selected_ids),
                "both_successfully_correct": both_correct,
            }
        )
    return {
        "pairs": len(details),
        "both_correct": sum(row["both_successfully_correct"] for row in details),
        "both_correct_rate": sum(row["both_successfully_correct"] for row in details) / len(details)
        if details
        else None,
        "details": details,
    }
