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


def finalize(
    manifest_path: Path = DEFAULT_MANIFEST,
    gold_path: Path = DEFAULT_GOLD,
    review_path: Path = DEFAULT_REVIEW,
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
    if review_items.get("gold_v3", {}).get("status") != "approved":
        raise ValueError("Gold v3 human review is not approved")

    gold = _load(gold_path)
    gold_errors = validate_frozen_gold(gold, require_final_held_out=True)
    if gold_errors:
        raise ValueError("; ".join(gold_errors))
    if gold.get("total_prompts") != 150:
        raise ValueError("Gold v3 must contain exactly 150 prompts")

    frozen = copy.deepcopy(manifest)
    frozen["status"] = "frozen"
    frozen["gold_v3"] = {
        "id": gold.get("dataset_id", gold.get("gold_id", "KISAKI-GOLD-V3")),
        "status": "frozen",
        "evaluation_role": gold["evaluation_role"],
        "count": gold["total_prompts"],
        "path": _manifest_path(gold_path),
        "sha256": _text_hash(gold_path),
    }
    frozen["freeze_blockers"] = []

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, manifest_path)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    try:
        result = finalize(
            manifest_path=args.manifest.resolve(),
            gold_path=args.gold.resolve(),
            review_path=args.review.resolve(),
        )
    except ValueError as exc:
        print(f"finalization_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
