#!/usr/bin/env python3
"""Apply the versioned speaker identity contract to canonical Kisaki V4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from training.speaker_contract import (  # noqa: E402
    CANONICAL_CHARACTER,
    GENERIC_USER,
    parse_speaker_contract,
    resolve_message_speaker,
)


V4_DIR = BACKEND_ROOT / "data/character_dialogues/experiments/v4"
DEFAULT_TRAIN = V4_DIR / "train.jsonl"
DEFAULT_MANIFEST = V4_DIR / "canonical_dataset_manifest.json"
CONTRACT_ID = "KISAKI-SPEAKER-CONTRACT-V1"
CONTRACT_VERSION = "1.0"
CONTRACT_DOCUMENT = (
    "backend/data/character_dialogues/experiments/v4/SPEAKER_CONTRACT.md"
)
TARGET_SPEAKERS = {"妃", "月社妃"}
CONSTRUCTED_SOURCES = {
    "codex_user_simulation_v41_reviewed",
    "deepseek_user_simulation_v41_reviewed",
    "llm_v4_blindfix",
    "llm_v4_lifestyle",
    "llm_v4_manual",
    "llm_v4_riou",
    "llm_v4_yoruko",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _jsonl_text(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _normalized_sha256(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _text_sha256(path: Path) -> str:
    return _normalized_sha256(path.read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _validate_game_attribution(record: dict[str, Any]) -> bool:
    metadata = record["metadata"]
    context_label = metadata.get("context_speaker_label")
    if not isinstance(context_label, str) or not context_label.strip():
        raise ValueError(f"game record {record.get('id')} has no context speaker")
    if metadata.get("source_speaker_label") not in TARGET_SPEAKERS:
        raise ValueError(f"game record {record.get('id')} has a non-Kisaki assistant")

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"game record {record.get('id')} has no messages")
    roles = [message.get("role") for message in messages]
    if roles not in (["user", "assistant"], ["user", "assistant", "user", "assistant"]):
        raise ValueError(f"game record {record.get('id')} has an unsupported turn layout")
    if len(messages) == 2:
        return False

    review = metadata.get("final_review")
    added = review.get("added_source_turns") if isinstance(review, dict) else None
    if not isinstance(added, list) or not added:
        raise ValueError(f"multi-turn game record {record.get('id')} lacks source turns")
    unexpected = {
        turn.get("speaker")
        for turn in added
        if turn.get("speaker") not in TARGET_SPEAKERS | {context_label}
    }
    if unexpected:
        raise ValueError(
            f"multi-turn game record {record.get('id')} contains other speakers: "
            f"{sorted(unexpected)}"
        )

    historical_user = messages[0]["content"]
    source_user_texts = {
        turn.get("text") for turn in added if turn.get("speaker") == context_label
    }
    source_assistant = "\n".join(
        str(turn.get("text", ""))
        for turn in added
        if turn.get("speaker") in TARGET_SPEAKERS
    )
    if historical_user not in source_user_texts:
        raise ValueError(f"multi-turn game record {record.get('id')} user source mismatch")
    if messages[1]["content"] != source_assistant:
        raise ValueError(
            f"multi-turn game record {record.get('id')} assistant source mismatch"
        )
    return True


def _migrate_record(record: dict[str, Any]) -> tuple[str, bool]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"record {record.get('id')} metadata must be an object")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record {record.get('id')} messages must be a non-empty list")
    source = metadata.get("data_source")
    if source == "game_extraction":
        multi_turn_verified = _validate_game_attribution(record)
        metadata["interlocutor_kind"] = CANONICAL_CHARACTER
        metadata["interlocutor_label"] = metadata["context_speaker_label"].strip()
        kind = CANONICAL_CHARACTER
    elif source in CONSTRUCTED_SOURCES:
        if metadata.get("context_speaker_label") or metadata.get("interlocutor_label"):
            raise ValueError(
                f"constructed record {record.get('id')} contains an unverified speaker"
            )
        metadata["interlocutor_kind"] = GENERIC_USER
        metadata.pop("interlocutor_label", None)
        multi_turn_verified = False
        kind = GENERIC_USER
    else:
        raise ValueError(f"record {record.get('id')} has unsupported data_source {source!r}")

    contract = parse_speaker_contract(record)
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"record {record.get('id')} contains a non-object message")
        role = message.get("role")
        resolve_message_speaker(message, role=str(role), contract=contract)
    return kind, multi_turn_verified


def apply_contract(*, train_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    records = _load_jsonl(train_path)
    current_hash = _text_sha256(train_path)
    revision = manifest.get("speaker_contract_revision", {})
    if (
        revision.get("contract_id") == CONTRACT_ID
        and revision.get("status") == "applied"
        and manifest.get("train", {}).get("sha256") == current_hash
    ):
        for record in records:
            _migrate_record(record)
        return {
            "status": "already_applied",
            "train_count": len(records),
            "train_sha256": current_hash,
        }

    expected_hash = manifest.get("train", {}).get("sha256")
    if current_hash != expected_hash:
        raise ValueError("canonical train file does not match its manifest hash")
    if len(records) != manifest.get("train", {}).get("count"):
        raise ValueError("canonical train count does not match its manifest")

    migrated = copy.deepcopy(records)
    kind_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    verified_multiturn = 0
    for record in migrated:
        kind, was_verified_multiturn = _migrate_record(record)
        kind_counts[kind] += 1
        if kind == CANONICAL_CHARACTER:
            label_counts[record["metadata"]["interlocutor_label"]] += 1
        verified_multiturn += int(was_verified_multiturn)

    migrated_text = _jsonl_text(migrated)
    migrated_hash = _normalized_sha256(migrated_text)
    updated = copy.deepcopy(manifest)
    updated["schema_version"] = max(int(updated.get("schema_version", 0)) + 1, 10)
    updated["train"]["sha256"] = migrated_hash
    updated["train"]["speaker_contract"] = {
        "version": CONTRACT_VERSION,
        "canonical_character_records": kind_counts[CANONICAL_CHARACTER],
        "generic_user_records": kind_counts[GENERIC_USER],
        "source_verified_multiturn_records": verified_multiturn,
    }
    updated["speaker_contract_revision"] = {
        "contract_id": CONTRACT_ID,
        "version": CONTRACT_VERSION,
        "status": "applied",
        "documentation_path": CONTRACT_DOCUMENT,
        "previous_train_sha256": current_hash,
        "result_train_sha256": migrated_hash,
        "record_count": len(migrated),
        "canonical_character_records": kind_counts[CANONICAL_CHARACTER],
        "generic_user_records": kind_counts[GENERIC_USER],
        "source_verified_multiturn_records": verified_multiturn,
        "canonical_speaker_distribution": dict(label_counts.most_common()),
        "message_level_labels_added": 0,
        "gold_or_validation_modified": False,
    }
    contamination = updated.get("gold_v3", {}).get("contamination_reaudit")
    if isinstance(contamination, dict):
        contamination["expected_train_sha256"] = migrated_hash

    _write_text_atomic(train_path, migrated_text)
    _write_text_atomic(
        manifest_path,
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "status": "applied_pending_existing_freeze_blockers",
        "train_count": len(migrated),
        "train_sha256": migrated_hash,
        "canonical_character_records": kind_counts[CANONICAL_CHARACTER],
        "generic_user_records": kind_counts[GENERIC_USER],
        "source_verified_multiturn_records": verified_multiturn,
        "freeze_blockers": updated.get("freeze_blockers", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        result = apply_contract(
            train_path=args.train.resolve(), manifest_path=args.manifest.resolve()
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"speaker_contract_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
