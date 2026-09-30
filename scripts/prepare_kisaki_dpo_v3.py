#!/usr/bin/env python3
"""Freeze reviewed Kisaki preference pairs into deterministic R4 train/held-out sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from training.preference_validation import group_split


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"id", "prompt", "chosen", "rejected", "review_status"} - set(row)
        if missing:
            raise ValueError(f"line {line_no} missing fields: {sorted(missing)}")
        rows.append(row)
    return rows


def _is_human_approved(row: dict[str, Any]) -> bool:
    if row.get("review_status") != "approved":
        return False
    metadata = row.get("metadata") or {}
    return metadata.get("human_final_approved") is True or row.get("annotator") in {
        "project_owner",
        "manual",
        "human_blind_review",
    }


def _is_kisaki(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    persona = str(metadata.get("persona") or metadata.get("character") or "").lower()
    return persona in {"kisaki", "tsukiyashiro_kisaki", "月社妃"}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"preference source not found: {args.input}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty frozen output: {args.output_dir}"
        )

    source = _read_jsonl(args.input)
    approved = [row for row in source if _is_human_approved(row) and _is_kisaki(row)]
    ids = [str(row["id"]) for row in approved]
    prompts = ["".join(str(row["prompt"]).split()).lower() for row in approved]
    errors = []
    if len(approved) < args.minimum:
        errors.append(
            f"human-approved Kisaki pairs {len(approved)} < required {args.minimum}"
        )
    if len(ids) != len(set(ids)):
        errors.append("duplicate preference pair ids")
    if len(prompts) != len(set(prompts)):
        errors.append("duplicate normalized prompts")
    if any(
        not str(row["chosen"]).strip() or not str(row["rejected"]).strip()
        for row in approved
    ):
        errors.append("empty chosen/rejected response")
    if errors:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "approved_kisaki": len(approved),
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    train, heldout = group_split(approved, seed=args.seed)
    if {row["id"] for row in train} & {row["id"] for row in heldout}:
        raise RuntimeError("train/held-out id overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "kisaki_dpo_train.jsonl"
    heldout_path = args.output_dir / "kisaki_dpo_heldout.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(heldout_path, heldout)
    manifest = {
        "schema_version": 2,
        "experiment_id": "R4-DPO-PILOT",
        "status": "frozen",
        "claim": "DPO pilot; not RLHF",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source": str(args.input),
        "source_sha256": _sha256(args.input),
        "human_approved_count": len(approved),
        "train": {
            "path": str(train_path),
            "count": len(train),
            "sha256": _sha256(train_path),
        },
        "heldout": {
            "path": str(heldout_path),
            "count": len(heldout),
            "sha256": _sha256(heldout_path),
        },
        "split": "approximately 80/20 by connected source/scene/evidence groups; validation, not final test",
        "validation_fraction_actual": len(heldout) / len(approved),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
