#!/usr/bin/env python3
"""Attach approved Gold v3 and mark the already-frozen V4 data as final."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.experiment_contracts import validate_frozen_gold  # noqa: E402


DEFAULT_MANIFEST = (
    BACKEND_ROOT
    / "data/character_dialogues/experiments/v4/canonical_dataset_manifest.json"
)
DEFAULT_GOLD = BACKEND_ROOT / "evaluation/kisaki_gold_set_v3.json"
DEFAULT_REVIEW = (
    PROJECT_ROOT / "docs/research/review_packets/kisaki_v4/review_manifest.json"
)
DEFAULT_GOLD_APPROVAL = (
    PROJECT_ROOT
    / "docs/research/review_packets/kisaki_v4/07_GOLD_V3/gold_v3_final_approval.json"
)
DEFAULT_AUDIT = BACKEND_ROOT / "evaluation/kisaki_gold_set_v3_contamination_audit.json"
DEFAULT_REGISTRY = (
    BACKEND_ROOT
    / "data/character_dialogues/experiments/research/research_program_registry_v4.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _text_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_current_audit(
    audit: dict[str, Any],
    manifest: dict[str, Any],
    gold_content_sha256: str,
) -> None:
    if audit.get("status") != "clean":
        raise ValueError("Gold v3 contamination audit is not clean")
    if audit.get("candidate_count") != 150:
        raise ValueError("Gold v3 contamination audit candidate count mismatch")
    if audit.get("candidate_content_sha256") != gold_content_sha256:
        raise ValueError("Gold v3 contamination audit does not match frozen Gold content")

    for field in (
        "duplicate_ids",
        "duplicate_normalized_prompts",
        "text_overlap_matches",
        "rag_evidence_event_overlaps",
    ):
        if audit.get(field) != []:
            raise ValueError(f"Gold v3 contamination audit has unresolved {field}")

    train = manifest["train"]
    validation = manifest["validation"]
    expected_reference_count = train["count"] + validation["count"]
    if audit.get("frozen_train_sha256") != train["sha256"]:
        raise ValueError("Gold v3 contamination audit uses a stale train split")
    if audit.get("frozen_validation_sha256") != validation["sha256"]:
        raise ValueError("Gold v3 contamination audit uses a stale validation split")
    if audit.get("frozen_train_count") != train["count"]:
        raise ValueError("Gold v3 contamination audit train count mismatch")
    if audit.get("frozen_validation_count") != validation["count"]:
        raise ValueError("Gold v3 contamination audit validation count mismatch")
    if audit.get("frozen_reference_count") != expected_reference_count:
        raise ValueError("Gold v3 contamination audit reference count mismatch")


def _updated_registry(
    registry: dict[str, Any],
    manifest: dict[str, Any],
    gold: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    updated = copy.deepcopy(registry)
    updated["status"] = "r0v4_complete_r1v4_config_regeneration_pending"
    canonical = updated["active_assets"]["canonical_dataset"]
    canonical["status"] = "frozen"
    canonical["train_count"] = manifest["train"]["count"]
    canonical["validation_count"] = manifest["validation"]["count"]

    gold_asset = updated["active_assets"]["gold_v3"]
    gold_asset.update(
        status="frozen",
        formal_use_allowed=True,
        candidate_count=gold["total_prompts"],
        content_sha256=gold["content_sha256"],
        contamination_audit_path=_manifest_path(audit_path),
    )
    for research in updated["research"]:
        if research["id"] == "R0V4":
            research["status"] = "complete"
        elif research["id"] == "R1V4":
            research["status"] = "config_regeneration_pending"
    return updated


def finalize(
    manifest_path: Path = DEFAULT_MANIFEST,
    gold_path: Path = DEFAULT_GOLD,
    review_path: Path = DEFAULT_REVIEW,
    audit_path: Path = DEFAULT_AUDIT,
    gold_approval_path: Path = DEFAULT_GOLD_APPROVAL,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if manifest.get("status") != "frozen_data_pending_gold":
        raise ValueError("V4 train and validation data are not frozen pending Gold")

    for split in ("train", "validation"):
        contract = manifest.get(split, {})
        data_path = _resolve(str(contract.get("path", "")))
        if contract.get("status") != "frozen" or not data_path.exists():
            raise ValueError(f"V4 {split} data are not frozen")
        if contract.get("sha256") != _text_hash(data_path):
            raise ValueError(f"V4 {split} data changed after freezing")

    review_items = _load(review_path).get("approval", {}).get("items", {})
    if review_items.get("gold_v21", {}).get("status") != "approved":
        raise ValueError("Gold v2.1 human review is not approved")
    review_gold = review_items.get("gold_v3", {})
    if review_gold.get("status") != "approved":
        raise ValueError("Gold v3 human review is not approved")

    gold = _load(gold_path)
    gold_errors = validate_frozen_gold(gold, require_final_held_out=True)
    if gold_errors:
        raise ValueError("; ".join(gold_errors))
    if gold.get("total_prompts") != 150:
        raise ValueError("Gold v3 must contain exactly 150 prompts")
    gold_content_sha256 = gold["content_sha256"]
    if review_gold.get("content_sha256") not in {None, gold_content_sha256}:
        raise ValueError("Gold v3 review manifest does not match frozen Gold content")

    direct_approval = _load(gold_approval_path)
    if direct_approval.get("status") != "approved":
        raise ValueError("Gold v3 direct human approval is not approved")
    if direct_approval.get("approved_count") != 150:
        raise ValueError("Gold v3 direct human approval count mismatch")
    if direct_approval.get("content_sha256") != gold_content_sha256:
        raise ValueError("Gold v3 direct human approval does not match frozen Gold content")

    audit = _load(audit_path)
    _validate_current_audit(audit, manifest, gold_content_sha256)

    frozen = copy.deepcopy(manifest)
    frozen["status"] = "frozen"
    frozen["gold_v3"] = {
        "id": gold.get("dataset_id", gold.get("gold_id", "KISAKI-GOLD-V3")),
        "status": "frozen",
        "evaluation_role": gold["evaluation_role"],
        "count": gold["total_prompts"],
        "path": _manifest_path(gold_path),
        "sha256": _text_hash(gold_path),
        "formal_use_allowed": True,
        "contamination_reaudit": {
            "status": "clean",
            "path": _manifest_path(audit_path),
            "sha256": _text_hash(audit_path),
            "train_sha256": audit["frozen_train_sha256"],
            "validation_sha256": audit["frozen_validation_sha256"],
            "frozen_reference_count": audit["frozen_reference_count"],
            "gold_content_sha256": audit["candidate_content_sha256"],
        },
    }
    frozen["freeze_blockers"] = []

    updated_registry = None
    if registry_path is not None:
        updated_registry = _updated_registry(_load(registry_path), frozen, gold, audit_path)

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    registry_temporary = None
    if updated_registry is not None and registry_path is not None:
        registry_temporary = registry_path.with_suffix(".json.tmp")
        registry_temporary.write_text(
            json.dumps(updated_registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    os.replace(temporary, manifest_path)
    if registry_temporary is not None and registry_path is not None:
        os.replace(registry_temporary, registry_path)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--gold-approval", type=Path, default=DEFAULT_GOLD_APPROVAL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    try:
        result = finalize(
            manifest_path=args.manifest.resolve(),
            gold_path=args.gold.resolve(),
            review_path=args.review.resolve(),
            audit_path=args.audit.resolve(),
            gold_approval_path=args.gold_approval.resolve(),
            registry_path=args.registry.resolve(),
        )
    except ValueError as exc:
        print(f"finalization_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
