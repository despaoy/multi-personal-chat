"""Compare saved outputs without rerunning models or rewriting earlier results.

Checks fixture hashes, gold, candidate identity and coverage. Counts paired wins
and regressions, not significance: these small AI-authored probes are correlated
development data and cannot establish a general research improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.evidence_selection import (  # noqa: E402
    SelectionObservation,
    paired_selection_metrics,
    selection_metrics,
)


def validated_observations(cases, report, input_sha256):
    if report.get("input_sha256") != input_sha256:
        raise ValueError("report fixture hash differs")
    fixtures = {case["id"]: case for case in cases}
    if len(fixtures) != len(cases):
        raise ValueError("duplicate fixture IDs")
    rows = report["cases"]
    if len(rows) != len(fixtures) or {row["id"] for row in rows} != set(fixtures):
        raise ValueError("report must cover the complete fixture exactly once")
    observations = []
    for row in rows:
        case = fixtures[row["id"]]
        ids = {str(item["id"]) for item in case["memories"]}
        if set(row["gold_ids"]) != set(case["gold_ids"]):
            raise ValueError("report gold differs from fixture")
        if not set(row["candidate_ids"]) <= ids:
            raise ValueError("candidate outside fixture")
        observations.append(
            SelectionObservation(
                row["id"],
                frozenset(case["gold_ids"]),
                tuple(row["selected_ids"]),
                row["status"],
                case.get("category", "general"),
                tuple(row["candidate_ids"]),
            )
        )
    selection_metrics(observations)
    return observations


def compare(cases, reports, input_sha256):
    pair_ids = {case["id"]: case["pair_id"] for case in cases if "pair_id" in case}
    variants = {}
    correctness = {}
    for name, report in reports.items():
        observations = validated_observations(cases, report, input_sha256)
        correctness[name] = {
            row.case_id: set(row.selected_ids) == row.gold_ids and row.status not in {"fallback", "error"}
            for row in observations
        }
        variants[name] = {
            "metrics": selection_metrics(observations),
            "paired_metrics": paired_selection_metrics(observations, pair_ids),
            "failed_case_ids": [key for key, correct in correctness[name].items() if not correct],
            "inference_calls": len(report.get("inference_calls", [])),
            "generation_seconds": sum(row["latency_seconds"] for row in report.get("inference_calls", [])),
        }
    first = next(iter(variants), None)
    comparisons = {}
    for name in list(variants)[1:]:
        baseline, candidate = correctness[first], correctness[name]
        comparisons[name] = {
            "reference": first,
            "improved_case_ids": [key for key in baseline if not baseline[key] and candidate[key]],
            "regressed_case_ids": [key for key in baseline if baseline[key] and not candidate[key]],
            "both_correct": sum(baseline[key] and candidate[key] for key in baseline),
            "both_wrong": sum(not baseline[key] and not candidate[key] for key in baseline),
        }
    return {
        "input_sha256": input_sha256,
        "quality_claim": "Paired development diagnostics only; not independent research evidence or a deployment gate.",
        "variants": variants,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if len({path.stem for path in args.report}) != len(args.report):
        parser.error("report names must be unique")
    source = args.input.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    if not cases:
        parser.error("empty fixture")
    reports = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.report}
    result = compare(cases, reports, hashlib.sha256(source.encode()).hexdigest())
    result["source_reports_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
    print(json.dumps({name: row["metrics"] for name, row in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
