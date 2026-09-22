"""Freeze a public Chinese character subset without generating or scoring replies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.charactereval_adapter import (  # noqa: E402
    SOURCE_SHA256,
    SUBSET_SEED,
    adapt_character_case,
    load_pinned_corpus,
    stable_character_subset,
)


def audit(rows, profiles, metric_map, *, roles=8, per_role=2):
    selected = stable_character_subset(rows, roles=roles, per_role=per_role)
    details = []
    for row in selected:
        detail = {"id": str(row["id"]), "role": row["role"], "novel_name": row.get("novel_name", "")}
        try:
            case = adapt_character_case(row, profiles, metric_map)
            detail.update(
                status="adapted",
                history_messages=len(case.history),
                context_sha256=hashlib.sha256(row["context"].encode()).hexdigest(),
                profile_sha256=hashlib.sha256(case.profile.encode()).hexdigest(),
                profile_chars=len(case.profile),
                context_chars=len(row["context"]),
                metric_ids=list(case.metric_ids),
            )
        except ValueError as error:
            detail.update(status="invalid", reason=str(error))
        details.append(detail)
    return {
        "status": "public_chinese_subset_frozen_no_generation_or_scores",
        "subset_seed": SUBSET_SEED,
        "requested_roles": roles,
        "per_role": per_role,
        "cases": details,
        "summary": {
            "corpus_cases": len(rows),
            "selected_cases": len(details),
            "invalid_selected_cases": sum(row["status"] == "invalid" for row in details),
            "novel_case_counts": dict(Counter(row["novel_name"] for row in details)),
            "selected_cases_without_metric_annotations": sum(
                row["status"] == "adapted" and not row["metric_ids"] for row in details
            ),
        },
        "limitations": [
            "Public data provenance is independent of this project's synthetic tests; pretrained-model contamination is not ruled out.",
            "Roles/novels differ from the project's target character. This cannot replace target-canon evaluation.",
            "No copyrighted source dialogue/profile text is copied into this audit report. Consult the external cache or upstream corpus by ID.",
            "No gold reply, human rating, training approval, CharacterRM inference or model-quality score is manufactured.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--roles", type=int, default=8)
    parser.add_argument("--per-role", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = audit(*load_pinned_corpus(args.cache_dir), roles=args.roles, per_role=args.per_role)
    report["source_sha256"] = SOURCE_SHA256
    report["adapter_sha256"] = hashlib.sha256(
        (ROOT / "backend/evaluation/charactereval_adapter.py").read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["summary"]["invalid_selected_cases"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
