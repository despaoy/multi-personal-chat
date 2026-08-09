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
        and baseline_ids == candidate_ids
        and len(set(baseline_ids)) == len(baseline_ids)
    )
    baseline_provenance = baseline.get("provenance", {})
    candidate_provenance = candidate.get("provenance", {})
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
        "paired_sample_order": {"passed": paired, "value": len(candidate_ids)},
        "same_prompt_content": {
            "passed": (
                bool(baseline_provenance.get("prompt_content_sha256"))
                and baseline_provenance.get("prompt_content_sha256")
                == candidate_provenance.get("prompt_content_sha256")
            ),
            "value": candidate_provenance.get("prompt_content_sha256"),
        },
        "same_generation_contract": {
            "passed": (
                bool(baseline_provenance.get("generation_sha256"))
                and baseline_provenance.get("generation_sha256")
                == candidate_provenance.get("generation_sha256")
            ),
            "value": candidate_provenance.get("generation_sha256"),
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
    formal_blockers = ["blind human review must be completed"]
    if not gold_frozen:
        formal_blockers.insert(0, "evaluation dataset must be frozen")
    required_checks = [
        item for item in checks.values() if not item.get("diagnostic_only", False)
    ]
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "passed": all(item["passed"] for item in required_checks),
        "checks": checks,
        "formal_conclusion_allowed": False,
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
