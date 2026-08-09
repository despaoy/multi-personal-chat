#!/usr/bin/env python3
"""Refuse formal Kisaki V4 training until human-review contracts are complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = PROJECT_ROOT / "docs" / "research" / "review_packets" / "kisaki_v4" / "review_manifest.json"
DEFAULT_DATASET = PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "experiments" / "v4" / "canonical_dataset_manifest.json"
DEFAULT_GOLD = PROJECT_ROOT / "backend" / "evaluation" / "kisaki_gold_set_v3.json"
DEFAULT_PROMPT = PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "kisaki_system_prompt_v3.txt"
REQUIRED_CATEGORIES = {
    "profile_prompt",
    "source_coverage",
    "game_train",
    "constructed_train",
    "validation",
    "gold_v2",
    "exclusions",
    "experiment_configs",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_gate(
    *,
    review_path: Path,
    dataset_path: Path,
    gold_path: Path,
    prompt_path: Path = DEFAULT_PROMPT,
    disk_path: Path | None = None,
    minimum_free_gb: float = 15.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    approved: set[str] = set()
    if not review_path.exists():
        blockers.append("human review manifest is missing")
    else:
        review = _load(review_path)
        approved = set(review.get("approval", {}).get("approved_categories", []))
        missing = sorted(REQUIRED_CATEGORIES - approved)
        if missing:
            blockers.append("human review categories are pending: " + ", ".join(missing))
        if review.get("approval", {}).get("all_required_approved") is not True:
            blockers.append("human review has not been explicitly finalized")
        prompt_review = review.get("approval", {}).get("items", {}).get("system_prompt_v3", {})
        if prompt_review.get("status") != "approved":
            blockers.append("system prompt v3 has not been explicitly approved")
        if not prompt_path.exists():
            blockers.append("system prompt v3 is missing")
        elif review.get("source_hashes", {}).get("prompt_v3") != _sha256(prompt_path):
            blockers.append("system prompt v3 hash does not match the reviewed prompt")

    if not dataset_path.exists():
        blockers.append("canonical V4 dataset manifest is missing")
    else:
        dataset = _load(dataset_path)
        if dataset.get("status") != "frozen":
            blockers.append("canonical V4 dataset is not frozen")
        for key in ("train", "validation"):
            if not dataset.get(key, {}).get("sha256"):
                blockers.append(f"canonical V4 {key} hash is missing")

    if not gold_path.exists():
        blockers.append("Gold v3 is missing")
    else:
        gold = _load(gold_path)
        if gold.get("status") != "frozen":
            blockers.append("Gold v3 is not frozen")
        if gold.get("total_prompts") != 150:
            blockers.append("Gold v3 must contain exactly 150 prompts")
        if not gold.get("content_sha256"):
            blockers.append("Gold v3 content hash is missing")

    free_gb = None
    if disk_path is not None:
        free_gb = shutil.disk_usage(disk_path).free / (1024**3)
        if free_gb < minimum_free_gb:
            blockers.append(
                f"free disk space {free_gb:.2f}GB is below {minimum_free_gb:.2f}GB"
            )
    return {
        "schema_version": 1,
        "gate": "KISAKI-V4-FORMAL-TRAINING",
        "passed": not blockers,
        "blockers": blockers,
        "approved_categories": sorted(approved),
        "free_disk_gb": None if free_gb is None else round(free_gb, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--disk-path", type=Path)
    parser.add_argument("--minimum-free-gb", type=float, default=15.0)
    args = parser.parse_args()
    result = validate_gate(
        review_path=args.review.resolve(),
        dataset_path=args.dataset.resolve(),
        gold_path=args.gold.resolve(),
        prompt_path=args.prompt.resolve(),
        disk_path=args.disk_path.resolve() if args.disk_path else None,
        minimum_free_gb=args.minimum_free_gb,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
