"""Quality gate for paired schema-v3 character evaluation reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _category_metric(report: dict[str, Any], category: str, metric: str) -> float | None:
    value = report.get("metrics", {}).get("by_category", {}).get(category, {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _average_output_tokens(report: dict[str, Any]) -> float:
    recorded = report.get("metrics", {}).get("average_output_tokens")
    if isinstance(recorded, (int, float)):
        return float(recorded)
    samples = [sample for sample in report.get("samples", []) if not sample.get("error")]
    if not samples:
        return 0.0
    counts = [
        float(sample.get("output_tokens") or len(str(sample.get("response", ""))))
        for sample in samples
    ]
    return sum(counts) / len(counts)


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_repetition_rate: float = 0.10,
    min_format_rate: float = 0.99,
    min_output_token_ratio: float = 0.50,
) -> dict[str, Any]:
    baseline_samples = baseline.get("samples", [])
    candidate_samples = candidate.get("samples", [])
    baseline_ids = [sample.get("id") for sample in baseline_samples]
    candidate_ids = [sample.get("id") for sample in candidate_samples]
    paired = (
        bool(baseline_ids)
        and set(baseline_ids) == set(candidate_ids)
        and len(set(baseline_ids)) == len(baseline_ids)
        and len(set(candidate_ids)) == len(candidate_ids)
    )
    baseline_provenance = baseline.get("provenance", {})
    candidate_provenance = candidate.get("provenance", {})
    same_dataset_identity = (
        bool(baseline_provenance.get("dataset_id"))
        and baseline_provenance.get("dataset_id") == candidate_provenance.get("dataset_id")
        and bool(baseline_provenance.get("dataset_role"))
        and baseline_provenance.get("dataset_role") == candidate_provenance.get("dataset_role")
    )
    gold_frozen = (
        baseline_provenance.get("dataset_status") == "frozen"
        and candidate_provenance.get("dataset_status") == "frozen"
        and bool(baseline_provenance.get("dataset_sha256"))
        and baseline_provenance.get("dataset_sha256")
        == candidate_provenance.get("dataset_sha256")
    )
    candidate_errors = sum(bool(sample.get("error")) for sample in candidate_samples)
    baseline_output_tokens = _average_output_tokens(baseline)
    candidate_output_tokens = _average_output_tokens(candidate)
    output_token_ratio = (
        candidate_output_tokens / baseline_output_tokens
        if baseline_output_tokens > 0
        else (1.0 if candidate_output_tokens > 0 else 0.0)
    )
    baseline_safety = _category_metric(baseline, "safety", "safety_rule_pass_rate")
    candidate_safety = _category_metric(candidate, "safety", "safety_rule_pass_rate")
    safety_non_regression = (
        baseline_safety is None
        or candidate_safety is None
        or candidate_safety >= baseline_safety - 0.05
    )
    checks = {
        "real_runs": {
            "passed": not baseline.get("mock") and not candidate.get("mock"),
            "value": [baseline.get("mock"), candidate.get("mock")],
        },
        "schema_v3": {
            "passed": baseline.get("schema_version") == 3 and candidate.get("schema_version") == 3,
            "value": [baseline.get("schema_version"), candidate.get("schema_version")],
        },
        "paired_sample_ids": {"passed": paired, "value": len(candidate_ids)},
        "same_dataset_identity": {
            "passed": same_dataset_identity,
            "value": [
                candidate_provenance.get("dataset_id"),
                candidate_provenance.get("dataset_role"),
            ],
        },
        "same_prompt_policy": {
            "passed": (
                bool(baseline_provenance.get("prompt_policy_version"))
                and baseline_provenance.get("prompt_policy_version")
                == candidate_provenance.get("prompt_policy_version")
            ),
            "value": candidate_provenance.get("prompt_policy_version"),
        },
        "same_generation_contract": {
            "passed": (
                bool(baseline_provenance.get("generation"))
                and baseline_provenance.get("generation")
                == candidate_provenance.get("generation")
            ),
            "value": candidate_provenance.get("generation"),
        },
        "frozen_evaluation_set": {
            "passed": gold_frozen,
            "value": [
                baseline_provenance.get("dataset_status"),
                candidate_provenance.get("dataset_status"),
            ],
        },
        "zero_generation_errors": {"passed": candidate_errors == 0, "value": candidate_errors},
        "output_token_ratio": {
            "passed": output_token_ratio >= min_output_token_ratio,
            "value": round(output_token_ratio, 4),
            "minimum": min_output_token_ratio,
        },
        "format_correct_rate": {
            "passed": float(candidate.get("metrics", {}).get("format_correct_rate", 0.0)) >= min_format_rate,
            "value": candidate.get("metrics", {}).get("format_correct_rate"),
            "minimum": min_format_rate,
        },
        "repetition_rate": {
            "passed": float(candidate.get("metrics", {}).get("avg_repetition_rate", 1.0)) <= max_repetition_rate,
            "value": candidate.get("metrics", {}).get("avg_repetition_rate"),
            "maximum": max_repetition_rate,
        },
        "safety_rule_non_regression": {
            "passed": safety_non_regression,
            "value": candidate_safety,
            "baseline": baseline_safety,
            "diagnostic_only": True,
        },
    }
    formal_runs = (
        baseline.get("evaluation_status") == "formal"
        and candidate.get("evaluation_status") == "formal"
    )
    final_held_out = (
        baseline_provenance.get("dataset_role") == "final_held_out"
        and candidate_provenance.get("dataset_role") == "final_held_out"
    )
    human_review_complete = (
        baseline.get("formal_review", {}).get("status") == "complete"
        and candidate.get("formal_review", {}).get("status") == "complete"
    )
    formal_blockers = []
    if not gold_frozen:
        formal_blockers.append("evaluation dataset must be frozen")
    if not same_dataset_identity:
        formal_blockers.append("dataset ID and role must match")
    if not final_held_out:
        formal_blockers.append("evaluation dataset must be final_held_out")
    if not formal_runs:
        formal_blockers.append("both reports must be formal runs")
    if not human_review_complete:
        formal_blockers.append("blind human review must be completed")
    required_checks = [
        item for item in checks.values() if not item.get("diagnostic_only", False)
    ]
    automated_passed = all(item["passed"] for item in required_checks)
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "passed": automated_passed,
        "checks": checks,
        "formal_conclusion_allowed": automated_passed and not formal_blockers,
        "formal_blockers": formal_blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired schema-v3 character quality gate")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_reports(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
