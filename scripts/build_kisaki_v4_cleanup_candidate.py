#!/usr/bin/env python3
"""Build the non-destructive Kisaki V4 length-cleanup candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from training.chat_dataset import normalize_chat_record, tokenize_assistant_turns  # noqa: E402


V4_DIR = BACKEND_ROOT / "data/character_dialogues/experiments/v4"
DEFAULT_TRAIN = V4_DIR / "train.jsonl"
DEFAULT_VALIDATION = V4_DIR / "validation.jsonl"
DEFAULT_CANONICAL_MANIFEST = V4_DIR / "canonical_dataset_manifest.json"
DEFAULT_POLICY = V4_DIR / "cleanup/length_cleanup_policy.json"
DEFAULT_OUTPUT = V4_DIR / "cleanup/candidate"
DEFAULT_CONFIG = V4_DIR / "configs/kisaki_r1v4_e1.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def apply_cleanup(
    records: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    exclusions = set(policy.get("formal_training_exclusion_ids", []))
    truncation_damaged = set(policy.get("truncation_damaged_ids", []))
    repairs = policy.get("scene_metadata_repairs", {})
    if not exclusions or not isinstance(repairs, dict):
        raise ValueError("cleanup policy must define exclusions and scene repairs")

    record_ids = [record.get("id") for record in records]
    duplicate_ids = sorted(key for key, count in Counter(record_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate training record IDs: {duplicate_ids}")
    missing_exclusions = sorted(exclusions - set(record_ids))
    missing_repairs = sorted(set(repairs) - set(record_ids))
    unknown_damaged = sorted(truncation_damaged - exclusions)
    if missing_exclusions or missing_repairs or unknown_damaged:
        raise ValueError(
            "cleanup policy references missing records: "
            f"exclusions={missing_exclusions}, repairs={missing_repairs}, "
            f"damaged_not_excluded={unknown_damaged}"
        )

    kept: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    repaired_ids: list[str] = []
    for source in records:
        record = json.loads(json.dumps(source, ensure_ascii=False))
        record_id = record["id"]
        if record_id in repairs:
            record.setdefault("metadata", {})["scene"] = repairs[record_id]
            repaired_ids.append(record_id)
        if record_id in exclusions:
            record.setdefault("metadata", {})["formal_training_status"] = "auxiliary_only"
            reason = (
                "truncation_damaged_and_code_dominant"
                if record_id in truncation_damaged
                else "code_dominant_auxiliary_capability_sample"
            )
            record["metadata"]["formal_training_exclusion_reason"] = reason
            auxiliary.append(record)
        else:
            kept.append(record)

    if len(auxiliary) != len(exclusions):
        raise ValueError("not every formal-training exclusion was written to auxiliary data")
    if any("\ufffd" in str(record.get("metadata", {}).get("scene", "")) for record in kept + auxiliary):
        raise ValueError("replacement characters remain in scene metadata after cleanup")
    return kept, auxiliary, sorted(repaired_ids)


def audit_token_lengths(
    records: list[dict[str, Any]],
    *,
    tokenizer_path: Path,
    config: dict[str, Any],
    max_sequence_length: int,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise ValueError("transformers is required for the exact token audit") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    lengths: list[int] = []
    over_limit: list[dict[str, Any]] = []
    for record in records:
        messages = normalize_chat_record(
            record,
            default_system_prompt=str(config["system_prompt"]),
            system_prompt_policy=str(config["system_prompt_policy"]),
        )
        tokenized = tokenize_assistant_turns(
            tokenizer,
            messages,
            max_length=1_000_000,
            truncation_direction=str(config.get("truncation_direction", "left")),
            use_chat_template=bool(config.get("chat_template", True)),
            assistant_supervision=str(
                record.get("metadata", {}).get("assistant_supervision", "all")
            ),
        )
        length = len(tokenized["input_ids"])
        lengths.append(length)
        if length > max_sequence_length:
            over_limit.append({"id": record["id"], "tokens": length})

    if over_limit:
        raise ValueError(f"cleaned records still exceed {max_sequence_length}: {over_limit}")
    ordered = sorted(lengths)
    return {
        "tokenizer_path": project_path(tokenizer_path),
        "chat_template": bool(config.get("chat_template", True)),
        "max_sequence_length": max_sequence_length,
        "record_count": len(records),
        "minimum_tokens": ordered[0],
        "median_tokens": (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2,
        "maximum_tokens": ordered[-1],
        "records_over_limit": 0,
    }


def supervised_assistant_turns(record: dict[str, Any]) -> int:
    assistant_turns = sum(
        message.get("role") == "assistant" for message in record.get("messages", [])
    )
    if record.get("metadata", {}).get("assistant_supervision") == "last":
        return min(assistant_turns, 1)
    return assistant_turns


def build_candidate(
    *,
    train_path: Path = DEFAULT_TRAIN,
    validation_path: Path = DEFAULT_VALIDATION,
    canonical_manifest_path: Path = DEFAULT_CANONICAL_MANIFEST,
    policy_path: Path = DEFAULT_POLICY,
    config_path: Path = DEFAULT_CONFIG,
    tokenizer_path: Path,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    policy = load_json(policy_path)
    source_manifest = load_json(canonical_manifest_path)
    source_train_hash = text_sha256(train_path)
    existing_manifest_path = output_dir / "manifest.json"
    cleanup_revision = source_manifest.get("cleanup_revision", {})
    if cleanup_revision.get("status") == "promoted" and existing_manifest_path.is_file():
        existing = load_json(existing_manifest_path)
        existing_train = Path(existing["train"]["path"])
        if not existing_train.is_absolute():
            existing_train = PROJECT_ROOT / existing_train
        auxiliary = Path(existing["formal_training_exclusions"]["path"])
        if not auxiliary.is_absolute():
            auxiliary = PROJECT_ROOT / auxiliary
        if source_train_hash != existing["train"]["sha256"]:
            raise ValueError("promoted train no longer matches the cleanup candidate")
        if text_sha256(existing_train) != existing["train"]["sha256"]:
            raise ValueError("stored cleanup candidate train hash mismatch")
        if text_sha256(auxiliary) != existing["formal_training_exclusions"]["sha256"]:
            raise ValueError("stored cleanup auxiliary hash mismatch")
        return existing
    if source_manifest.get("train", {}).get("sha256") != source_train_hash:
        raise ValueError("source train file does not match the canonical manifest")

    train, auxiliary, repaired_ids = apply_cleanup(load_jsonl(train_path), policy)
    target_max_length = int(policy["target_max_sequence_length"])
    token_audit = audit_token_lengths(
        train,
        tokenizer_path=tokenizer_path,
        config=load_json(config_path),
        max_sequence_length=target_max_length,
    )

    output_train = output_dir / "train.jsonl"
    output_auxiliary = output_dir / "technical_auxiliary.jsonl"
    output_manifest = output_dir / "manifest.json"
    write_jsonl_atomic(output_train, train)
    write_jsonl_atomic(output_auxiliary, auxiliary)

    source_distribution = Counter(
        str(record.get("metadata", {}).get("data_source", "unknown")) for record in train
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": "KISAKI-CANONICAL-V4-CLEANUP-CANDIDATE",
        "status": "ready_for_review",
        "source": {
            "canonical_manifest_path": project_path(canonical_manifest_path),
            "canonical_train_path": project_path(train_path),
            "canonical_train_sha256": source_train_hash,
            "canonical_train_count": source_manifest["train"]["count"],
        },
        "train": {
            "path": project_path(output_train),
            "count": len(train),
            "sha256": text_sha256(output_train),
            "assistant_supervision_targets": sum(
                supervised_assistant_turns(record) for record in train
            ),
            "source_distribution": dict(sorted(source_distribution.items())),
        },
        "validation": {
            "path": project_path(validation_path),
            "count": source_manifest["validation"]["count"],
            "sha256": text_sha256(validation_path),
            "modified": False,
        },
        "formal_training_exclusions": {
            "count": len(auxiliary),
            "path": project_path(output_auxiliary),
            "sha256": text_sha256(output_auxiliary),
            "ids": [record["id"] for record in auxiliary],
            "reason_counts": dict(
                sorted(
                    Counter(
                        record["metadata"]["formal_training_exclusion_reason"]
                        for record in auxiliary
                    ).items()
                )
            ),
        },
        "scene_metadata_repairs": {
            "count": len(repaired_ids),
            "ids": repaired_ids,
        },
        "training_contract": {
            "truncation_direction": "left",
            "token_audit": token_audit,
        },
        "promotion_blockers": [
            "human_review_of_cleanup_candidate_pending",
            "gold_set_chronology_and_hash_refreeze_pending",
            "prompt_policy_alignment_pending",
        ],
    }
    write_json_atomic(output_manifest, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = build_candidate(
            train_path=args.train.resolve(),
            validation_path=args.validation.resolve(),
            canonical_manifest_path=args.canonical_manifest.resolve(),
            policy_path=args.policy.resolve(),
            config_path=args.config.resolve(),
            tokenizer_path=args.tokenizer.resolve(),
            output_dir=args.output.resolve(),
        )
    except (KeyError, OSError, ValueError) as exc:
        print(f"cleanup_candidate_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
