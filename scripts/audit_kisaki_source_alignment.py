#!/usr/bin/env python3
"""Audit Kisaki source attribution and extraction reproducibility."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = PROJECT_ROOT / "backend" / "data" / "character_dialogues"
DEFAULT_GAMETEXT = PROJECT_ROOT / "gametext" / "纸上魔法使"
EXTRACTOR = PROJECT_ROOT / "scripts" / "extract_character_dialogues.py"
NORMALIZER = PROJECT_ROOT / "scripts" / "apply_kisaki_v4_text_normalizations.py"
NORMALIZATION_LEDGER = (
    CHARACTER_DIR / "experiments/v4/text_normalizations.json"
)


def _extractor():
    spec = importlib.util.spec_from_file_location("extract_character_dialogues", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _normalizer():
    spec = importlib.util.spec_from_file_location(
        "apply_kisaki_v4_text_normalizations", NORMALIZER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(gametext: Path = DEFAULT_GAMETEXT) -> dict[str, Any]:
    extractor = _extractor()
    target = "tsukiyashiro_kisaki"
    groups = [
        {"source_id": source_id, "source_role": role, "events": extractor.read_script_events(path, source_id)}
        for path, source_id, role in extractor.source_scripts(gametext)
    ]
    expected_raw = extractor.make_raw(groups, target)
    expected_sft, expected_full, expected_excluded = extractor.make_sft(groups, target)
    normalizer = _normalizer()
    normalization_items = _load_json(NORMALIZATION_LEDGER)["items"]
    expected_sft = normalizer.normalize_sft_rows(expected_sft, normalization_items)
    expected_full = normalizer.normalize_sft_rows(expected_full, normalization_items)
    stored_raw = _load_jsonl(CHARACTER_DIR / f"{target}_raw.jsonl")
    stored_sft = _load_json(CHARACTER_DIR / f"{target}_sft.json")
    stored_full = _load_json(CHARACTER_DIR / f"{target}_sft_full.json")
    stored_excluded = _load_jsonl(CHARACTER_DIR / f"{target}_excluded.jsonl")

    errors: list[str] = []
    persisted = {
        "raw": (stored_raw, expected_raw),
        "recommended_sft": (stored_sft, expected_sft),
        "full_sft": (stored_full, expected_full),
        "excluded": (stored_excluded, expected_excluded),
    }
    for name, (stored, expected) in persisted.items():
        if stored != expected:
            errors.append(f"{name} does not match deterministic re-extraction")

    raw_ids = [row["id"] for row in stored_raw]
    if len(raw_ids) != len(set(raw_ids)):
        errors.append("raw IDs are not unique")
    sft_ids = [row["id"] for row in stored_sft]
    if len(sft_ids) != len(set(sft_ids)):
        errors.append("recommended SFT IDs are not unique")

    event_by_source = {
        event["source"]: event
        for group in groups
        for event in group["events"]
        if event["speaker"] in extractor.TARGETS[target]["train_aliases"]
    }
    for row in stored_raw:
        event = event_by_source.get(row["source"])
        if event is None:
            errors.append(f"missing source event: {row['id']}")
        elif event["speaker"] != row["speaker_label"] or event["text"] != row["text"]:
            errors.append(f"source attribution mismatch: {row['id']}")

    referenced_ids = {
        event_id
        for collection in (stored_full, stored_excluded)
        for row in collection
        for event_id in row.get("metadata", {}).get("target_event_ids", [])
    }
    missing_disposition = sorted(set(raw_ids) - referenced_ids)
    if missing_disposition:
        errors.append(f"raw events without full/excluded disposition: {len(missing_disposition)}")

    source_files = list(gametext.glob("*.txt"))
    result = {
        "schema_version": 1,
        "audit": "KISAKI-SOURCE-ALIGNMENT",
        "passed": not errors,
        "counts": {
            "source_files": len(source_files),
            "raw_dialogues": len(stored_raw),
            "recommended_sft": len(stored_sft),
            "full_sft": len(stored_full),
            "excluded": len(stored_excluded),
            "raw_with_disposition": len(set(raw_ids) & referenced_ids),
        },
        "checks": {
            "deterministic_re_extraction": all(stored == expected for stored, expected in persisted.values()),
            "source_attribution": not any("source attribution" in error or "missing source" in error for error in errors),
            "unique_ids": len(raw_ids) == len(set(raw_ids)) and len(sft_ids) == len(set(sft_ids)),
            "complete_disposition": not missing_disposition,
        },
        "errors": errors,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gametext", type=Path, default=DEFAULT_GAMETEXT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.gametext.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
