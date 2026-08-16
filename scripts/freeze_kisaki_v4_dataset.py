#!/usr/bin/env python3
"""Apply approved reviews to V4 candidates and freeze train/validation data."""

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
V4_DIR = PROJECT_ROOT / "backend/data/character_dialogues/experiments/v4"
REVIEW_ROOT = PROJECT_ROOT / "docs/research/review_packets/kisaki_v4"
GAME_APPROVAL = REVIEW_ROOT / "03_GAME_TRAIN/game_train_final_approval.json"
CONSTRUCTED_APPROVAL = REVIEW_ROOT / "04_CONSTRUCTED_TRAIN/constructed_final_approval.json"
VALIDATION_APPROVAL = REVIEW_ROOT / "05_VALIDATION/validation_current_approval.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _text_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _manifest_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def apply_validation_approval(
    records: list[dict[str, Any]],
    approval: dict[str, Any],
    required_exclusions: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if (
        approval.get("status") != "approved"
        or approval.get("default_decision") != "approve"
        or approval.get("candidate_count") != len(records)
    ):
        raise ValueError("current validation review is not approved")
    record_ids = {record["id"] for record in records}
    excluded_ids = set(approval.get("excluded_ids", []))
    if not excluded_ids <= record_ids:
        raise ValueError("validation review references unknown candidate IDs")
    if not required_exclusions <= excluded_ids:
        raise ValueError("validation review does not resolve all leakage blockers")

    approved: list[dict[str, Any]] = []
    for source in records:
        if source["id"] in excluded_ids:
            continue
        record = copy.deepcopy(source)
        record.setdefault("metadata", {})["human_review"] = {
            "status": "approved",
            "review_id": approval["review_id"],
            "reviewed_by": approval["reviewed_by"],
            "reviewed_at": approval["reviewed_at"],
        }
        approved.append(record)
    return approved, sorted(excluded_ids)


def freeze(v4_dir: Path = V4_DIR) -> dict[str, Any]:
    manifest_path = v4_dir / "canonical_dataset_manifest.json"
    manifest = _load(manifest_path)
    if manifest.get("status") != "draft_rebuilt_pending_review":
        raise ValueError("canonical candidate manifest is not awaiting review")
    train_candidates = _load_jsonl(v4_dir / "train_candidate.jsonl")
    validation_candidates = _load_jsonl(v4_dir / "validation_candidate.jsonl")
    game_count = sum(
        record.get("metadata", {}).get("data_source") == "game_extraction"
        for record in train_candidates
    )
    constructed_count = len(train_candidates) - game_count

    game_approval = _load(GAME_APPROVAL)
    if (
        game_approval.get("status") != "approved"
        or game_approval.get("default_decision") != "approve"
        or game_approval.get("candidate_count") != game_count
    ):
        raise ValueError("current game train review is not approved")
    excluded_game = set(game_approval.get("excluded_ids", []))
    game_ids = {
        record["id"]
        for record in train_candidates
        if record.get("metadata", {}).get("data_source") == "game_extraction"
    }
    if not excluded_game <= game_ids:
        raise ValueError("game train review references unknown candidate IDs")

    constructed_approval = _load(CONSTRUCTED_APPROVAL)
    if (
        constructed_approval.get("status") != "approved"
        or not constructed_approval.get("approved_by")
        or constructed_approval.get("approved_count") != constructed_count
    ):
        raise ValueError("constructed train review is not approved")

    final_train = [record for record in train_candidates if record["id"] not in excluded_game]
    leakage_suggestions = _load(v4_dir / "validation_exclusions.json")
    required_validation_exclusions = {
        item["validation_id"]
        for item in leakage_suggestions.get("exclusions", [])
        if item.get("paired_train_id") not in excluded_game
    }
    final_validation, validation_exclusions = apply_validation_approval(
        validation_candidates,
        _load(VALIDATION_APPROVAL),
        required_validation_exclusions,
    )

    train_path = v4_dir / "train.jsonl"
    validation_path = v4_dir / "validation.jsonl"
    _write_jsonl(train_path, final_train)
    _write_jsonl(validation_path, final_validation)

    frozen = copy.deepcopy(manifest)
    frozen["status"] = "frozen_data_pending_gold"
    frozen["train"].update(
        status="frozen",
        count=len(final_train),
        path=_manifest_path(train_path),
        sha256=_text_hash(train_path),
        source_distribution={
            "game_extraction_current_sft": game_count - len(excluded_game),
            "llm_v4_reviewed_constructed": constructed_count,
        },
        game_human_review_status="approved",
        constructed_human_review_status="approved",
    )
    frozen["validation"].update(
        status="frozen",
        count=len(final_validation),
        path=_manifest_path(validation_path),
        sha256=_text_hash(validation_path),
        source_distribution={"game_extraction_current_sft": len(final_validation)},
        human_review_status="approved",
        applied_exclusions=validation_exclusions,
    )
    frozen["checks"]["train_validation_blocker_pairs"] = 0
    frozen["freeze_blockers"] = ["gold_v3_missing"]
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary_manifest, manifest_path)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-dir", type=Path, default=V4_DIR)
    args = parser.parse_args()
    try:
        result = freeze(args.v4_dir.resolve())
    except ValueError as exc:
        print(f"freeze_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
