"""Small A/B harness shared by tests and offline experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ABCase:
    case_id: str
    query: str
    expected_ids: tuple[str, ...]
    raw_text: bool = False


def _rank(bundle: dict[str, Any] | None, expected: set[str]) -> int | None:
    if not bundle:
        return None
    for rank, item in enumerate(bundle.get("results") or [], 1):
        if str(item.get("id")) in expected:
            return rank
    return None


def evaluate_ab(
    cases: Iterable[ABCase],
    baseline_retrieve: Callable[..., dict[str, Any] | None],
    experimental_retrieve: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    """Compare retrieval only; it never promotes the experimental path."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected = set(case.expected_ids)
        started = time.perf_counter()
        baseline = baseline_retrieve(case.query)
        baseline_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        experiment = experimental_retrieve(case.query, raw_text=case.raw_text)
        experiment_ms = (time.perf_counter() - started) * 1000
        baseline_rank = _rank(baseline, expected)
        experiment_rank = _rank(experiment, expected)
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "baseline_rank": baseline_rank,
                "experiment_rank": experiment_rank,
                "baseline_hit": baseline_rank is not None,
                "experiment_hit": experiment_rank is not None,
                "baseline_rr": 1.0 / baseline_rank if baseline_rank else 0.0,
                "experiment_rr": 1.0 / experiment_rank if experiment_rank else 0.0,
                "baseline_ms": round(baseline_ms, 3),
                "experiment_ms": round(experiment_ms, 3),
                "raw_available": bool((experiment or {}).get("raw_excerpt")),
            }
        )

    count = len(rows)
    mean = lambda key: round(sum(float(row[key]) for row in rows) / count, 6) if count else 0.0
    return {
        "case_count": count,
        "baseline": {
            "recall_at_k": mean("baseline_hit"),
            "mrr": mean("baseline_rr"),
            "mean_latency_ms": mean("baseline_ms"),
        },
        "experiment": {
            "recall_at_k": mean("experiment_hit"),
            "mrr": mean("experiment_rr"),
            "mean_latency_ms": mean("experiment_ms"),
            "raw_available_rate": mean("raw_available"),
        },
        "rows": rows,
        "promotion_decision": "manual_review_required",
    }
