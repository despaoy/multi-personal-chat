#!/usr/bin/env python3
"""Re-audit the approved Gold v3 against the current frozen V4 splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.experiment_contracts import (  # noqa: E402
    canonical_json_hash,
    sha256_text_file,
    validate_frozen_gold,
)
from build_kisaki_gold_v3 import contamination_audit  # noqa: E402


DEFAULT_MANIFEST = (
    BACKEND_ROOT
    / "data/character_dialogues/experiments/v4/canonical_dataset_manifest.json"
)
DEFAULT_GOLD = BACKEND_ROOT / "evaluation/kisaki_gold_set_v3.json"
DEFAULT_AUDIT = BACKEND_ROOT / "evaluation/kisaki_gold_set_v3_contamination_audit.json"
DEFAULT_APPROVAL = (
    PROJECT_ROOT
    / "docs/research/review_packets/kisaki_v4/07_GOLD_V3/gold_v3_final_approval.json"
)
DEFAULT_DEVELOPMENT_GOLD = BACKEND_ROOT / "evaluation/kisaki_gold_set_v21_candidates.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def reaudit(
    manifest_path: Path = DEFAULT_MANIFEST,
    gold_path: Path = DEFAULT_GOLD,
    approval_path: Path = DEFAULT_APPROVAL,
    audit_path: Path = DEFAULT_AUDIT,
    development_gold_path: Path = DEFAULT_DEVELOPMENT_GOLD,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if manifest.get("status") not in {"frozen_data_pending_gold", "frozen"}:
        raise ValueError("V4 train and validation must be frozen before Gold re-audit")

    split_paths: dict[str, Path] = {}
    for split in ("train", "validation"):
        contract = manifest.get(split, {})
        data_path = _resolve(project_root, str(contract.get("path", "")))
        if contract.get("status") != "frozen" or not data_path.exists():
            raise ValueError(f"V4 {split} data are not frozen")
        if contract.get("sha256") != sha256_text_file(data_path):
            raise ValueError(f"V4 {split} data changed after freezing")
        split_paths[split] = data_path

    gold = _load(gold_path)
    gold_errors = validate_frozen_gold(gold, require_final_held_out=True)
    if gold_errors:
        raise ValueError("; ".join(gold_errors))
    if gold.get("total_prompts") != 150:
        raise ValueError("Gold v3 must contain exactly 150 prompts")

    prompts = gold["prompts"]
    content_sha256 = canonical_json_hash(prompts)
    approval = _load(approval_path)
    if approval.get("status") != "approved" or approval.get("approved_count") != 150:
        raise ValueError("Gold v3 direct human approval is incomplete")
    if approval.get("content_sha256") != content_sha256:
        raise ValueError("Gold v3 approval does not match the frozen Gold content")

    audit = contamination_audit(
        prompts,
        train_path=split_paths["train"],
        validation_path=split_paths["validation"],
        manifest_path=manifest_path,
        development_gold_path=development_gold_path,
    )
    if audit["status"] != "clean":
        raise ValueError("Gold v3 contamination re-audit is blocked")
    if audit["candidate_content_sha256"] != content_sha256:
        raise ValueError("Gold v3 contamination audit content hash mismatch")
    if audit["frozen_reference_count"] != (
        manifest["train"]["count"] + manifest["validation"]["count"]
    ):
        raise ValueError("Gold v3 contamination audit reference count mismatch")

    _write_json_atomic(audit_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--development-gold", type=Path, default=DEFAULT_DEVELOPMENT_GOLD)
    args = parser.parse_args()
    try:
        result = reaudit(
            manifest_path=args.manifest.resolve(),
            gold_path=args.gold.resolve(),
            approval_path=args.approval.resolve(),
            audit_path=args.audit.resolve(),
            development_gold_path=args.development_gold.resolve(),
        )
    except ValueError as exc:
        print(f"gold_v3_reaudit_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
