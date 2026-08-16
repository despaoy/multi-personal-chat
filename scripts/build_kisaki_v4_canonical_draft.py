#!/usr/bin/env python3
"""Build the non-frozen Kisaki V4 train and independent validation draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from inference.prompt_policy import PROMPT_POLICY_VERSION  # noqa: E402

CHARACTER_DIR = PROJECT_ROOT / "backend" / "data" / "character_dialogues"
EXPERIMENT_DIR = CHARACTER_DIR / "experiments"
V3_DIR = EXPERIMENT_DIR / "v3"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "v4"
PROMPT_PATH = CHARACTER_DIR / "kisaki_system_prompt_v3.txt"
RAW_DATA_PATH = CHARACTER_DIR / "tsukiyashiro_kisaki_raw.jsonl"
CANONICAL_SFT_PATH = CHARACTER_DIR / "tsukiyashiro_kisaki_sft.json"
RAG_DOCUMENTS_PATH = (
    EXPERIMENT_DIR / "research" / "character_rag_seed_documents.json"
)
CONSTRUCTED_DATA_PATH = EXPERIMENT_DIR / "train_v5_clean.jsonl"
SPLIT_SEED_PATH = DEFAULT_OUTPUT / "split_seed.json"
USER_SIMILARITY_THRESHOLD = 0.90
LINE_PROXIMITY_THRESHOLD = 5
BLOCK_KEYS = (
    "split_group",
    "conversation_block_id",
    "dialogue_group_id",
    "conversation_group_id",
    "dialogue_block_id",
)
CONTEXT_WINDOW_KEYS = (
    ("context_line_start", "context_line_end"),
    ("source_context_start", "source_context_end"),
    ("context_start_line", "context_end_line"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return "".join(value.casefold().split())


def runtime_policy_semantic_hits(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Find the three reviewed runtime-policy families in role-training data."""

    hits: list[dict[str, str]] = []
    for record in records:
        user_text = normalize_text("\n".join(user_messages(record)))
        scene = normalize_text(str(record.get("metadata", {}).get("scene", "")))
        assistant_text = normalize_text(
            "\n".join(
                message["content"]
                for message in record["messages"]
                if message["role"] == "assistant"
            )
        )
        family = None
        if "系统提示" in user_text and any(word in user_text for word in ("发给我", "告诉我", "泄露", "输出")):
            family = "system_prompt_extraction"
        elif any(word in user_text for word in ("密钥", "apikey", "api密钥")) and any(
            word in user_text for word in ("泄露", "发给我", "告诉我", "输出", "提供")
        ):
            family = "secret_exfiltration"
        elif (
            (
                "伤害自己" in assistant_text
                or "确认她现在是否安全" in assistant_text
                or "确认他现在是否安全" in assistant_text
                or "你现在安全吗" in assistant_text
            )
            and (
                scene == "安全边界"
                or any(phrase in user_text for phrase in ("撑不住", "想放弃", "不想活", "自杀"))
            )
        ):
            family = "self_harm_risk_triage"
        if family:
            hits.append({"id": record["id"], "family": family})
    return hits


def user_messages(record: dict[str, Any]) -> list[str]:
    return [
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    ]


def source_location(record: dict[str, Any]) -> tuple[str | None, int | None, int | None]:
    metadata = record.get("metadata", {})
    source_file = metadata.get("source_file")
    start = metadata.get("source_line_start")
    end = metadata.get("source_line_end")
    source = metadata.get("source")
    if (source_file is None or start is None) and isinstance(source, str):
        match = re.match(r"^(.*):line:(\d+)$", source)
        if match:
            source_file = source_file or match.group(1)
            start = start if start is not None else int(match.group(2))
    if isinstance(start, int) and not isinstance(end, int):
        end = start
    return source_file, start if isinstance(start, int) else None, end if isinstance(end, int) else None


def string_set(value: Any) -> set[str]:
    if isinstance(value, (str, int)):
        return {str(value)}
    if isinstance(value, list):
        return {str(item) for item in value if isinstance(item, (str, int))}
    return set()


def explicit_context_windows(record: dict[str, Any]) -> list[tuple[str, int, int]]:
    metadata = record.get("metadata", {})
    source_file = metadata.get("source_file")
    windows: list[tuple[str, int, int]] = []
    if isinstance(source_file, str):
        for start_key, end_key in CONTEXT_WINDOW_KEYS:
            start, end = metadata.get(start_key), metadata.get(end_key)
            if isinstance(start, int) and isinstance(end, int):
                windows.append((source_file, min(start, end), max(start, end)))
    raw_window = metadata.get("context_window")
    if isinstance(raw_window, dict):
        window_file = raw_window.get("source_file", source_file)
        start, end = raw_window.get("start"), raw_window.get("end")
        if isinstance(window_file, str) and isinstance(start, int) and isinstance(end, int):
            windows.append((window_file, min(start, end), max(start, end)))
    return windows


def intervals_overlap(left: tuple[str, int, int], right: tuple[str, int, int]) -> bool:
    return left[0] == right[0] and max(left[1], right[1]) <= min(left[2], right[2])


def record_pair_audit(train: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    similarities = [
        (SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio(), left, right)
        for left in user_messages(train)
        for right in user_messages(validation)
    ]
    best_similarity, train_user, validation_user = max(similarities, default=(0.0, "", ""))
    reasons: list[str] = []
    if normalize_text(train_user) == normalize_text(validation_user):
        reasons.append("exact_user_question")
    elif best_similarity >= USER_SIMILARITY_THRESHOLD:
        reasons.append("near_duplicate_user_question")

    train_metadata = train.get("metadata", {})
    validation_metadata = validation.get("metadata", {})
    event_overlap = sorted(
        string_set(train_metadata.get("target_event_ids"))
        & string_set(validation_metadata.get("target_event_ids"))
    )
    if event_overlap:
        reasons.append("target_event_overlap")

    block_overlaps = {
        key: sorted(string_set(train_metadata.get(key)) & string_set(validation_metadata.get(key)))
        for key in BLOCK_KEYS
    }
    block_overlaps = {key: values for key, values in block_overlaps.items() if values}
    if block_overlaps:
        reasons.append("explicit_dialogue_block_overlap")

    context_overlaps = [
        {"train": left, "validation": right}
        for left in explicit_context_windows(train)
        for right in explicit_context_windows(validation)
        if intervals_overlap(left, right)
    ]
    if context_overlaps:
        reasons.append("explicit_context_window_overlap")

    train_file, train_start, train_end = source_location(train)
    validation_file, validation_start, validation_end = source_location(validation)
    line_distance = None
    if train_file == validation_file and train_start is not None and validation_start is not None:
        line_distance = min(
            abs(train_start - validation_start),
            abs((train_end or train_start) - validation_start),
            abs(train_start - (validation_end or validation_start)),
        )

    return {
        "train_id": train["id"],
        "validation_id": validation["id"],
        "train_user": train_user,
        "validation_user": validation_user,
        "user_similarity": round(best_similarity, 10),
        "train_source_file": train_file,
        "train_source_lines": [train_start, train_end],
        "validation_source_file": validation_file,
        "validation_source_lines": [validation_start, validation_end],
        "line_distance": line_distance,
        "target_event_id_overlap": event_overlap,
        "explicit_block_overlap": block_overlaps,
        "explicit_context_window_overlap": context_overlaps,
        "reasons": reasons,
    }


def audit_validation(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []
    for train_record in train:
        for validation_record in validation:
            result = record_pair_audit(train_record, validation_record)
            if result["reasons"]:
                blockers.append(result)
            elif result["line_distance"] is not None and result["line_distance"] <= LINE_PROXIMITY_THRESHOLD:
                result["advisory"] = "line_proximity_only"
                advisories.append(result)
    return blockers, advisories


def path_for_manifest(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def canonical_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("messages") or record.get("conversations")
    if not isinstance(raw, list):
        raise ValueError(f"record has no message list: {record.get('id')}")
    messages: list[dict[str, str]] = []
    for message in raw:
        role = message.get("role") or message.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = message.get("content", message.get("value"))
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported role in {record.get('id')}: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"empty message in {record.get('id')}")
        messages.append({"role": role, "content": content})
    roles = [message["role"] for message in messages]
    if not roles or roles[0] != "user" or roles[-1] != "assistant":
        raise ValueError(f"record must start with user and end with assistant: {record.get('id')}")
    if any(left == right for left, right in zip(roles, roles[1:])):
        raise ValueError(f"adjacent message roles must alternate: {record.get('id')}")
    return messages


def canonical_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "messages": canonical_messages(record),
        "metadata": record.get("metadata", {}),
    }


def validate_current_source_lineage(
    records: list[dict[str, Any]], split_name: str
) -> dict[str, int]:
    raw_ids = {record["id"] for record in load_jsonl(RAW_DATA_PATH)}
    canonical_sft_ids = {record["id"] for record in load_json(CANONICAL_SFT_PATH)}
    missing_events: dict[str, list[str]] = {}
    missing_sft_ids: list[str] = []
    checked = 0
    for record in records:
        if record.get("metadata", {}).get("data_source") != "game_extraction":
            continue
        checked += 1
        if record["id"] not in canonical_sft_ids:
            missing_sft_ids.append(record["id"])
        event_ids = string_set(record.get("metadata", {}).get("target_event_ids"))
        missing = sorted(event_ids - raw_ids)
        if missing:
            missing_events[record["id"]] = missing
    if missing_sft_ids or missing_events:
        raise ValueError(
            f"{split_name} source lineage is stale: "
            f"missing_sft_ids={sorted(missing_sft_ids)}, missing_events={missing_events}"
        )
    return {
        "game_extraction_records_checked": checked,
        "missing_canonical_sft_ids": 0,
        "missing_raw_event_ids": 0,
    }


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def quote_balance_errors(records: Iterable[dict[str, Any]]) -> list[str]:
    pairs = (("“", "”"), ("「", "」"), ("『", "』"))
    errors = []
    for record in records:
        assistant_text = "\n".join(
            message["content"]
            for message in record["messages"]
            if message["role"] == "assistant"
        )
        if any(assistant_text.count(left) != assistant_text.count(right) for left, right in pairs):
            errors.append(record["id"])
    return errors


def game_record(record: dict[str, Any]) -> dict[str, Any]:
    output = canonical_record(record)
    metadata = output.setdefault("metadata", {})
    metadata["data_source"] = "game_extraction"
    metadata["split_origin"] = "current_sft_migrated_from_v3_event_membership"
    return output


def event_ids(record: dict[str, Any]) -> set[str]:
    return string_set(record.get("metadata", {}).get("target_event_ids"))


def split_membership(split_seed: dict[str, Any]) -> tuple[set[str], set[str]]:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    legacy_train = load_json(resolve(split_seed["train_seed_path"]))
    legacy_validation = load_json(resolve(split_seed["validation_seed_path"]))
    train_events = set().union(*(event_ids(record) for record in legacy_train))
    validation_events = set().union(*(event_ids(record) for record in legacy_validation))
    for migration in split_seed.get("event_id_migrations", []):
        target = train_events if migration.get("split") == "train" else validation_events
        target.discard(str(migration["previous_event_id"]))
        target.add(str(migration["current_event_id"]))
    overlap = train_events & validation_events
    if overlap:
        raise ValueError(f"train/validation split seed overlaps: {sorted(overlap)}")
    return train_events, validation_events


def rag_evidence_event_ids() -> tuple[set[str], int]:
    payload = load_json(RAG_DOCUMENTS_PATH)
    documents = payload.get("documents", [])
    evidence = {
        str(event_id)
        for document in documents
        for event_id in document.get("source_event_ids", [])
    }
    if not documents or not evidence:
        raise ValueError("RAG evidence lineage is empty")
    return evidence, len(documents)


def partition_current_game_sft(
    split_seed: dict[str, Any], evidence_events: set[str]
) -> dict[str, Any]:
    current = [game_record(record) for record in load_json(CANONICAL_SFT_PATH)]
    train_events, validation_events = split_membership(split_seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    rag_withheld: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    for record in current:
        targets = event_ids(record)
        if not targets:
            raise ValueError(f"current SFT record has no target events: {record['id']}")
        evidence_overlap = sorted(targets & evidence_events)
        if evidence_overlap:
            rag_withheld.append({"id": record["id"], "event_ids": evidence_overlap})
        elif targets <= train_events:
            train.append(record)
        elif targets <= validation_events:
            validation.append(record)
        else:
            unassigned.append({"id": record["id"], "event_ids": sorted(targets)})

    return {
        "current_count": len(current),
        "train": train,
        "validation": validation,
        "rag_withheld": rag_withheld,
        "unassigned": unassigned,
    }


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    split_seed = load_json(SPLIT_SEED_PATH)
    evidence_events, rag_document_count = rag_evidence_event_ids()
    game_partition = partition_current_game_sft(split_seed, evidence_events)
    game_train = game_partition["train"]
    constructed = [
        canonical_record(record)
        for record in load_jsonl(CONSTRUCTED_DATA_PATH)
    ]
    train = game_train + constructed
    validation = game_partition["validation"]
    train_lineage = validate_current_source_lineage(game_train, "train")
    validation_lineage = validate_current_source_lineage(validation, "validation")
    runtime_policy_hits = runtime_policy_semantic_hits(constructed)
    quote_errors = quote_balance_errors(train + validation)
    if quote_errors:
        raise ValueError(f"unbalanced Chinese quotes in canonical records: {quote_errors}")

    train_events = set().union(*(event_ids(record) for record in game_train))
    validation_events = set().union(*(event_ids(record) for record in validation))
    if train_events & evidence_events or validation_events & evidence_events:
        raise ValueError("RAG evidence event leaked into train or validation")
    if train_events & validation_events:
        raise ValueError("train and validation target events overlap")

    ids = [record["id"] for record in train + validation]
    if len(ids) != len(set(ids)):
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate IDs across V4 train/validation: {duplicates}")
    candidate_blockers, line_proximity_advisories = audit_validation(train, validation)

    train_path = output / "train_candidate.jsonl"
    validation_path = output / "validation_candidate.jsonl"
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(validation_path, validation)

    audit_path = output / "validation_leakage_audit.json"
    advisory_validation_ids = sorted({item["validation_id"] for item in line_proximity_advisories})
    audit_report = {
        "schema_version": 1,
        "status": "pending_review" if candidate_blockers else "passed_with_advisories",
        "thresholds": {
            "user_similarity_block": USER_SIMILARITY_THRESHOLD,
            "line_proximity_advisory": LINE_PROXIMITY_THRESHOLD,
        },
        "policy": {
            "blockers": [
                "exact_user_question",
                "near_duplicate_user_question",
                "target_event_overlap",
                "explicit_dialogue_block_overlap",
                "explicit_context_window_overlap",
            ],
            "advisory_only": ["line_proximity_only"],
        },
        "candidate_blockers": candidate_blockers,
        "line_proximity_advisories": line_proximity_advisories,
        "summary": {
            "candidate_blocker_pairs": len(candidate_blockers),
            "line_proximity_advisory_pairs": len(line_proximity_advisories),
            "line_proximity_advisory_validation_records": len(advisory_validation_ids),
        },
    }
    audit_path.write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    suggested_exclusions_path = output / "validation_exclusions.json"
    suggested_exclusions = {
        "schema_version": 1,
        "dataset_id": "KISAKI-CANONICAL-V4",
        "status": "candidate_suggestions",
        "exclusions": [
            {
                "validation_id": item["validation_id"],
                "paired_train_id": item["train_id"],
                "reason": item["reasons"][0],
            }
            for item in candidate_blockers
        ],
    }
    suggested_exclusions_path.write_text(
        json.dumps(suggested_exclusions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    freeze_blockers = [
        "constructed_final_unified_review_pending",
        "game_train_rebuilt_review_pending",
        "validation_review_pending",
        "gold_v21_human_review_pending",
        "gold_v3_missing",
    ]
    if runtime_policy_hits:
        freeze_blockers.insert(0, "constructed_round_02_review_pending")

    manifest = {
        "schema_version": 6,
        "dataset_id": "KISAKI-CANONICAL-V4",
        "status": "draft_rebuilt_pending_review",
        "hash_mode": "sha256_utf8_lf_v1",
        "seed": 42,
        "train": {
            "status": "candidate",
            "count": len(train),
            "path": path_for_manifest(train_path),
            "sha256": sha256_text(train_path),
            "source_distribution": {
                "game_extraction_current_sft": len(game_train),
                "llm_v4_pending_final_review": len(constructed),
            },
            "game_human_review_status": "pending_rebuilt_game_review",
            "constructed_human_review_status": "pending_final_unified_review",
        },
        "validation": {
            "status": "candidate_pending_review",
            "count": len(validation),
            "path": path_for_manifest(validation_path),
            "sha256": sha256_text(validation_path),
            "source_distribution": {"game_extraction_current_sft": len(validation)},
            "suggested_exclusion_count": len(candidate_blockers),
            "suggested_exclusions_path": path_for_manifest(suggested_exclusions_path),
        },
        "rag_holdout": {
            "documents_path": path_for_manifest(RAG_DOCUMENTS_PATH),
            "document_count": rag_document_count,
            "source_event_count": len(evidence_events),
            "withheld_sft_record_count": len(game_partition["rag_withheld"]),
            "withheld_sft_records": game_partition["rag_withheld"],
        },
        "prompt_policy": {
            "mode": "training_config_injected",
            "version": PROMPT_POLICY_VERSION,
            "record_system_messages": 0,
            "path": PROMPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "required_training_policy": "replace",
        },
        "checks": {
            "train_validation_blocker_pairs": len(candidate_blockers),
            "line_proximity_advisory_validation_records": len(advisory_validation_ids),
            "validation_leakage_audit_path": path_for_manifest(audit_path),
            "unique_ids_across_train_validation": True,
            "record_system_messages": 0,
            "train_source_lineage": train_lineage,
            "validation_source_lineage": validation_lineage,
            "train_validation_target_event_overlap": 0,
            "rag_evidence_train_overlap": 0,
            "rag_evidence_validation_overlap": 0,
            "unassigned_current_sft_records": game_partition["unassigned"],
            "unbalanced_quote_records": [],
            "runtime_policy_semantic_hits": runtime_policy_hits,
        },
        "provenance": {
            "current_sft": path_for_manifest(CANONICAL_SFT_PATH),
            "split_seed": path_for_manifest(SPLIT_SEED_PATH),
            "constructed_train": "backend/data/character_dialogues/experiments/train_v5_clean.jsonl",
            "rag_evidence": path_for_manifest(RAG_DOCUMENTS_PATH),
        },
        "freeze_blockers": freeze_blockers,
    }
    manifest_path = output / "canonical_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
