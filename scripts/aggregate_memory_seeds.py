"""Aggregate predeclared seed runs without treating repeats as independent cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.evidence_selection import (  # noqa: E402
    SelectionObservation,
    paired_selection_metrics,
    selection_metrics,
)
from scripts.compare_memory_reports import validated_observations  # noqa: E402


def majority_observations(cases, runs, *, max_items=5):
    """Post-hoc same-model voting, not calibrated confidence or independence.

    Require a strict majority of the originally declared runs for every selected
    item. Failed runs do not cast negative votes, and too few completed runs
    produce a failure, not a successful empty decision. No gold enters voting.
    """
    quorum = len(runs) // 2 + 1
    if not runs or max_items < 1:
        raise ValueError("voting needs runs and a positive item budget")
    observations = []
    diagnostics = []
    for case in cases:
        rows = [run[case["id"]] for run in runs.values()]
        valid = [row for row in rows if row.status not in {"fallback", "error"}]
        votes = {}
        ranks = {}
        for row in valid:
            for rank, key in enumerate(row.selected_ids):
                votes[key] = votes.get(key, 0) + 1
                ranks.setdefault(key, []).append(rank)
        completed = len(valid) >= quorum
        selected = (
            sorted(
                (key for key, count in votes.items() if count >= quorum),
                key=lambda key: (-votes[key], statistics.mean(ranks[key]), key),
            )[:max_items]
            if completed
            else []
        )
        observations.append(
            SelectionObservation(
                case["id"],
                frozenset(case["gold_ids"]),
                tuple(selected),
                "selected" if completed else "fallback",
                case.get("category", "general"),
                rows[0].candidate_ids,
            )
        )
        diagnostics.append(
            {
                "id": case["id"],
                "completed_runs": len(valid),
                "required_votes": quorum,
                "votes": votes,
                "selected_ids": selected,
                "status": "selected" if completed else "fallback",
            }
        )
    return observations, diagnostics


def exact_set_observations(cases, runs):
    """Require agreement on a whole set; never assemble an unseen combination.

    Item-wise majorities can combine mutually inconsistent proposals. A strict
    whole-set quorum is more conservative: lack of agreement is a failure to
    decide, not a valid empty selection. No gold influences the chosen set.
    """
    if not runs:
        raise ValueError("voting needs runs")
    quorum = len(runs) // 2 + 1
    observations, diagnostics = [], []
    for case in cases:
        rows = [run[case["id"]] for run in runs.values()]
        valid = [row for row in rows if row.status not in {"fallback", "error"}]
        votes = Counter(tuple(sorted(row.selected_ids)) for row in valid)
        agreed = [keys for keys, count in votes.items() if count >= quorum]
        completed = bool(agreed)
        selected = agreed[0] if completed else ()
        observations.append(
            SelectionObservation(
                case["id"],
                frozenset(case["gold_ids"]),
                selected,
                "selected" if completed else "fallback",
                case.get("category", "general"),
                rows[0].candidate_ids,
            )
        )
        diagnostics.append(
            {
                "id": case["id"],
                "completed_runs": len(valid),
                "required_votes": quorum,
                "set_votes": [{"selected_ids": list(keys), "count": count} for keys, count in sorted(votes.items())],
                "selected_ids": list(selected),
                "status": "selected" if completed else "fallback",
                "fallback_reason": ""
                if completed
                else "insufficient_completed_runs"
                if len(valid) < quorum
                else "no_set_consensus",
            }
        )
    return observations, diagnostics


def aggregate(cases, reports, input_sha256, expected_seeds=(42, 43, 44)):
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected seeds must be nonempty and unique")
    runs = {}
    call_logs = []
    reference_recipe = None
    for report in reports:
        inference = dict(report["inference"])
        seed = inference.pop("seed", None)
        if inference.get("decoding") != "sampled" or type(seed) is not int or seed in runs:
            raise ValueError("seed runs must be sampled and have unique integer seeds")
        recipe = {
            "inference": inference,
            "selection_protocol": report.get("selection_protocol", "direct"),
            "candidate_view": report.get("candidate_view", "original"),
            "candidate_view_sha256": report.get("candidate_view_sha256"),
            "selection_instruction_sha256": report["selection_instruction_sha256"],
            "planning_instruction_sha256": report.get("planning_instruction_sha256"),
            "embedding_enabled": report["embedding_enabled"],
            "embedding_model": report.get("embedding_model"),
        }
        if reference_recipe is None:
            reference_recipe = recipe
        elif recipe != reference_recipe:
            raise ValueError("seed comparison must retain the same inference/prompt recipe")
        observations = validated_observations(cases, report, input_sha256)
        runs[seed] = {row.case_id: row for row in observations}
        call_logs.append(report.get("inference_calls"))
    if set(runs) != set(expected_seeds):
        raise ValueError("all and only predeclared seeds must be supplied")
    details = []
    per_seed = {seed: 0 for seed in expected_seeds}
    for case in cases:
        correct = {}
        signatures = set()
        candidate_sets = set()
        for seed in expected_seeds:
            row = runs[seed][case["id"]]
            signature = (tuple(sorted(row.selected_ids)), row.status)
            signatures.add(signature)
            candidate_sets.add(tuple(sorted(row.candidate_ids or ())))
            correct[seed] = set(row.selected_ids) == row.gold_ids and row.status not in {"fallback", "error"}
            per_seed[seed] += correct[seed]
        if len(candidate_sets) != 1:
            raise ValueError("candidate recall changed between seeds")
        details.append(
            {
                "id": case["id"],
                "successful_exact_by_seed": correct,
                "correct_all_seeds": all(correct.values()),
                "wrong_all_seeds": not any(correct.values()),
                "selection_or_status_varies": len(signatures) > 1,
            }
        )
    if not details:
        raise ValueError("empty corpus")
    accuracies = [per_seed[seed] / len(details) for seed in expected_seeds]
    voted, vote_details = majority_observations(cases, runs)
    set_voted, set_vote_details = exact_set_observations(cases, runs)
    set_metrics = selection_metrics(set_voted)
    # The common metric helper uses provider_* names for any fallback. Here a
    # disagreement can occur with three successful model calls: name it honestly.
    set_metrics["consensus_failures"] = set_metrics.pop("provider_failures")
    set_metrics["consensus_failure_rate"] = set_metrics.pop("provider_failure_rate")
    return {
        "input_sha256": input_sha256,
        "recipe": reference_recipe,
        "expected_seeds": list(expected_seeds),
        "distinct_cases": len(details),
        "repeated_case_evaluations": len(details) * len(expected_seeds),
        "successful_exact_counts_by_seed": per_seed,
        "mean_successful_exact_accuracy": statistics.mean(accuracies),
        "min_successful_exact_accuracy": min(accuracies),
        "max_successful_exact_accuracy": max(accuracies),
        "correct_all_seeds": sum(row["correct_all_seeds"] for row in details),
        "wrong_all_seeds": [row["id"] for row in details if row["wrong_all_seeds"]],
        "varying_case_ids": [row["id"] for row in details if row["selection_or_status_varies"]],
        "cases": details,
        "posthoc_majority_vote": {
            "model_calls_across_all_seeds": sum(len(log) for log in call_logs)
            if all(isinstance(log, list) for log in call_logs)
            else None,
            "summed_generation_seconds": sum(call["latency_seconds"] for log in call_logs for call in log)
            if all(isinstance(log, list) for log in call_logs)
            else None,
            "metrics": selection_metrics(voted),
            "paired_metrics": paired_selection_metrics(
                voted, {case["id"]: case["pair_id"] for case in cases if "pair_id" in case}
            ),
            "cases": vote_details,
            "quality_claim": "Replay of all declared seeds; added inference cost, correlated same-model votes, no held-out validation or calibrated probability.",
        },
        "posthoc_exact_set_vote": {
            "metrics": set_metrics,
            "cases": set_vote_details,
            "additional_model_calls_beyond_itemwise_replay": 0,
            "quality_claim": "Same source runs, strict majority of complete sets; disagreement is failure, not successful abstention. Still post-hoc and not independently validated.",
        },
        "quality_claim": "Seed stability on the same correlated development cases, not independent test examples or a confidence interval.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    source = args.input.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    result = aggregate(
        cases,
        [json.loads(path.read_text(encoding="utf-8")) for path in args.report],
        hashlib.sha256(source.encode()).hexdigest(),
    )
    result["source_reports_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"cases", "recipe", "source_reports_sha256"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
