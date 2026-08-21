#!/usr/bin/env python3
"""Promote the reviewed Round 06 sessions into the canonical V4 train split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
V4_DIR = ROOT / "backend/data/character_dialogues/experiments/v4"
ROUND06_DIR = V4_DIR / "augmentation_candidates/deepseek_user_simulation_round06"
SOURCE_SESSIONS = ROUND06_DIR / "sessions.json"
APPROVED_SESSIONS = ROUND06_DIR / "approved_sessions.json"
PROMOTION_RESULT = ROUND06_DIR / "promotion_result.json"
GOLD_V3 = ROOT / "backend/evaluation/kisaki_gold_set_v3.json"
GOLD_V21 = ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json"
GOLD_V3_AUDIT = ROOT / "backend/evaluation/kisaki_gold_set_v3_contamination_audit.json"
RESEARCH_REGISTRY = (
    ROOT
    / "backend/data/character_dialogues/experiments/research/research_program_registry_v4.json"
)
REVIEW_MANIFEST = ROOT / "docs/research/review_packets/kisaki_v4/review_manifest.json"
SIMILARITY_THRESHOLD = 0.90
REVIEWED_AUGMENTATION_SOURCES = {
    "deepseek_user_simulation_v41_reviewed",
    "codex_user_simulation_v41_reviewed",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected JSON object list: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in JSONL: {path}")
    return rows


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _normalized_sha256_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _text_sha256(path: Path) -> str:
    return _normalized_sha256_text(path.read_text(encoding="utf-8"))


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _matches_authorized_downstream_review(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    """Accept an exact chain of provenance-backed revisions after promotion."""

    def matches(current: dict[str, Any], target: dict[str, Any], depth: int) -> bool:
        if depth > 4:
            return False
        metadata = current.get("metadata", {})
        for review_key in ("full_dialogue_review", "persona_review"):
            review = metadata.get(review_key, {})
            decision_source = review.get("decision_source")
            if (
                review.get("status") not in {"approved_unchanged", "approved_after_revision"}
                or not isinstance(decision_source, str)
                or not decision_source.endswith("/record_reviews.jsonl")
            ):
                continue

            review_dir = ROOT / Path(decision_source).parent
            original_path = review_dir / "original_llm_records.jsonl"
            reviewed_path = review_dir / "reviewed_llm_records.jsonl"
            summary_path = review_dir / "summary.json"
            if not all(path.is_file() for path in (original_path, reviewed_path, summary_path)):
                continue
            summary = _load_json(summary_path)
            if (
                summary.get("status") != "approved_and_promoted"
                or summary.get("review_id") != review.get("review_id")
            ):
                continue
            original = {record["id"]: record for record in _load_jsonl(original_path)}
            reviewed = {record["id"]: record for record in _load_jsonl(reviewed_path)}
            record_id = current.get("id")
            before = original.get(record_id)
            if reviewed.get(record_id) != current or before is None:
                continue
            if before == target or matches(before, target, depth + 1):
                return True
        return False

    return matches(actual, expected, 0)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)
    os.replace(temporary, path)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _user_texts(record: dict[str, Any]) -> list[str]:
    if isinstance(record.get("prompt"), str):
        return [record["prompt"]]
    messages = record.get("messages") or record.get("conversation") or []
    return [
        message["content"]
        for message in messages
        if isinstance(message, dict)
        and message.get("role", "user") == "user"
        and isinstance(message.get("content"), str)
    ]


def _validate_role_sequence(messages: list[dict[str, Any]], *, session_id: str) -> None:
    if len(messages) != 10:
        raise ValueError(f"{session_id} must contain exactly five user/assistant turns")
    expected = ["user", "assistant"] * 5
    roles = [message.get("role") for message in messages]
    if roles != expected:
        raise ValueError(f"{session_id} has invalid role sequence: {roles}")
    if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
        raise ValueError(f"{session_id} contains an empty message")


def build_promoted_records(
    source_path: Path = SOURCE_SESSIONS,
    approved_path: Path = APPROVED_SESSIONS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_sessions = _load_json_list(source_path)
    approval = _load_json(approved_path)
    if approval.get("status") not in {"approved_after_review", "approved_after_revision"}:
        raise ValueError("sessions have not passed the delegated review")
    contract = approval.get("promotion_contract", {})
    approved_sessions = approval.get("sessions")
    if not isinstance(approved_sessions, list):
        raise ValueError("approved sessions are missing")
    record_count = contract.get("record_count")
    if not isinstance(record_count, int) or not 1 <= record_count <= 4:
        raise ValueError("one batch must promote between one and four complete sessions")
    if len(approved_sessions) != record_count:
        raise ValueError("approved session count does not match the promotion contract")

    source_by_id = {session["session_id"]: session for session in source_sessions}
    if len(source_by_id) != len(source_sessions):
        raise ValueError("source sessions contain duplicate session IDs")

    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    assistant_turn_count = 0
    for approved in approved_sessions:
        session_id = approved.get("session_id")
        if session_id not in source_by_id:
            raise ValueError(f"approved session is not present in source: {session_id}")
        source = source_by_id[session_id]
        messages = approved.get("messages")
        if messages is None and approved.get("messages_mode") == "source_unchanged":
            messages = copy.deepcopy(source.get("messages"))
        if not isinstance(messages, list):
            raise ValueError(f"approved messages are missing: {session_id}")
        _validate_role_sequence(messages, session_id=session_id)
        _validate_role_sequence(source["messages"], session_id=f"source:{session_id}")

        approved_users = [message["content"] for message in messages if message["role"] == "user"]
        source_users = [
            message["content"] for message in source["messages"] if message["role"] == "user"
        ]
        if approved_users != source_users:
            raise ValueError(f"promotion must preserve original user turns: {session_id}")

        approved_assistants = [
            message["content"] for message in messages if message["role"] == "assistant"
        ]
        source_assistants = [
            message["content"] for message in source["messages"] if message["role"] == "assistant"
        ]
        changed_turns = [
            index
            for index, (original, revised) in enumerate(
                zip(source_assistants, approved_assistants), start=1
            )
            if original != revised
        ]
        if changed_turns != approved.get("revised_assistant_turns"):
            raise ValueError(
                f"revision provenance does not match changed turns for {session_id}: {changed_turns}"
            )

        record_id = approved.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError(f"invalid or duplicate promoted record ID: {record_id!r}")
        record_ids.add(record_id)
        assistant_turn_count += len(approved_assistants)
        records.append(
            {
                "id": record_id,
                "messages": copy.deepcopy(messages),
                "metadata": {
                    "character": "月社妃",
                    "data_source": contract["data_source"],
                    "source": contract.get("source", source_path.parent.name),
                    "scene": approved["title"],
                    "task_type": approved["task_type"],
                    "turns": 5,
                    "version": contract.get("version", "v4.1_reviewed"),
                    "generation_provenance": {
                        "source_session_id": session_id,
                        "source_session_sha256": _canonical_json_sha256(source),
                        "source_candidate_path": _relative(source_path),
                        "model": source.get("metadata", {}).get("model"),
                        "temperature": source.get("metadata", {}).get("temperature"),
                        "source_corpus_count": source.get("metadata", {}).get(
                            "source_corpus_count"
                        ),
                    },
                    "human_review": {
                        "status": approval["status"],
                        "review_id": approval["approval_id"],
                        "reviewed_by": approval["approved_by"],
                        "reviewed_at": approval["approved_at"],
                        "decision_source": _relative(approved_path),
                        "revised_assistant_turns": changed_turns,
                    },
                },
            }
        )

    expected_assistant_turns = record_count * 5
    if (
        contract.get("assistant_turn_count") != expected_assistant_turns
        or assistant_turn_count != expected_assistant_turns
    ):
        raise ValueError("assistant turn count must equal five per complete session")
    return approval, records


def _similarity_matches(
    candidates: Iterable[tuple[str, str]],
    references: Iterable[tuple[str, str, str]],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    reference_rows = list(references)
    matches: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates:
        left = _normalized_text(candidate)
        for reference_id, reference, source in reference_rows:
            right = _normalized_text(reference)
            similarity = SequenceMatcher(None, left, right).ratio()
            if left == right or similarity >= threshold:
                matches.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_text": candidate,
                        "reference_id": reference_id,
                        "reference_text": reference,
                        "reference_source": source,
                        "similarity": round(similarity, 10),
                    }
                )
    return matches


def _gold_contamination_audit(
    *,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    gold_v3: dict[str, Any],
    gold_v21: dict[str, Any],
    train_sha256: str,
    validation_sha256: str,
) -> dict[str, Any]:
    frozen = train + validation
    gold_rows = gold_v3["prompts"]
    development_rows = gold_v21["prompts"]
    references = [
        (record["id"], text, "frozen_data")
        for record in frozen
        for text in _user_texts(record)
    ] + [
        (record["id"], text, "gold_v21")
        for record in development_rows
        for text in _user_texts(record)
    ]
    candidates = [
        (record["id"], text) for record in gold_rows for text in _user_texts(record)
    ]
    matches = _similarity_matches(candidates, references)

    frozen_event_ids = {
        event_id
        for record in frozen
        for event_id in record.get("metadata", {}).get("target_event_ids", [])
    }
    rag_event_ids = {
        event_id
        for record in gold_rows
        if record.get("category") == "rag_grounded"
        for evidence in record.get("evidence_refs", [])
        for event_id in evidence.get("source_event_ids", [])
    }
    ids = [record["id"] for record in gold_rows]
    prompts = [_normalized_text(text) for _, text in candidates]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    duplicate_prompts = sorted(key for key, count in Counter(prompts).items() if count > 1)
    rag_overlap = sorted(frozen_event_ids & rag_event_ids)
    status = (
        "clean"
        if not matches and not duplicate_ids and not duplicate_prompts and not rag_overlap
        else "blocked"
    )
    return {
        "schema_version": 1,
        "status": status,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "candidate_count": len(gold_rows),
        "frozen_reference_count": len(frozen),
        "development_reference_count": len(development_rows),
        "duplicate_ids": duplicate_ids,
        "duplicate_normalized_prompts": duplicate_prompts,
        "text_overlap_matches": matches,
        "rag_evidence_event_count": len(rag_event_ids),
        "rag_evidence_event_overlaps": rag_overlap,
        "frozen_train_sha256": train_sha256,
        "frozen_validation_sha256": validation_sha256,
    }


def _updated_authority_payloads(
    canonical_manifest: dict[str, Any],
    approval: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = _load_json(RESEARCH_REGISTRY)
    review_manifest = _load_json(REVIEW_MANIFEST)
    distribution = canonical_manifest["train"]["source_distribution"]
    augmentation_count = sum(
        int(distribution.get(source, 0)) for source in REVIEWED_AUGMENTATION_SOURCES
    )

    canonical_asset = registry["active_assets"]["canonical_dataset"]
    canonical_asset["status"] = canonical_manifest["status"]
    canonical_asset["train_count"] = canonical_manifest["train"]["count"]
    canonical_asset["reviewed_multiturn_augmentation_count"] = augmentation_count

    review_manifest["counts"]["frozen_train"] = canonical_manifest["train"]["count"]
    review_manifest["counts"]["reviewed_multiturn_augmentation"] = augmentation_count
    automatic_batches = [
        item
        for item in canonical_manifest.get("augmentations", [])
        if str(item.get("id", "")).startswith("KISAKI-V41-AUTO-BATCH-")
    ]
    review_manifest["approval"]["items"]["automated_multiturn_augmentation"] = {
        "status": "active_delegated_review",
        "approved_batch_count": len(automatic_batches),
        "approved_record_count": sum(int(item["record_count"]) for item in automatic_batches),
        "assistant_turn_count": sum(
            int(item["assistant_turn_count"]) for item in automatic_batches
        ),
        "last_approval_id": approval["approval_id"],
        "target_records": 1000,
        "target_assistant_turns": 1500,
        "target_assistant_characters": 80000,
    }
    return registry, review_manifest


def promote(
    *,
    v4_dir: Path = V4_DIR,
    source_path: Path = SOURCE_SESSIONS,
    approved_path: Path = APPROVED_SESSIONS,
    gold_v3_path: Path = GOLD_V3,
    gold_v21_path: Path = GOLD_V21,
    audit_path: Path = GOLD_V3_AUDIT,
    result_path: Path = PROMOTION_RESULT,
) -> dict[str, Any]:
    approval, promoted = build_promoted_records(source_path, approved_path)
    train_path = v4_dir / "train.jsonl"
    validation_path = v4_dir / "validation.jsonl"
    manifest_path = v4_dir / "canonical_dataset_manifest.json"
    train = _load_jsonl(train_path)
    validation = _load_jsonl(validation_path)
    manifest = _load_json(manifest_path)
    gold_v3 = _load_json(gold_v3_path)
    gold_v21 = _load_json(gold_v21_path)
    base_contract = approval["base_contract"]

    if len(validation) != base_contract["validation_count"]:
        raise ValueError("validation count changed since Round 06 approval")
    if _text_sha256(validation_path) != base_contract["validation_sha256"]:
        raise ValueError("validation content changed since Round 06 approval")

    promoted_ids = {record["id"] for record in promoted}
    existing_ids = {record["id"] for record in train}
    auxiliary_path = v4_dir / "cleanup/candidate/technical_auxiliary.jsonl"
    auxiliary = _load_jsonl(auxiliary_path) if auxiliary_path.is_file() else []
    auxiliary_ids = {record["id"] for record in auxiliary}
    if existing_ids & auxiliary_ids:
        raise ValueError("cleanup auxiliary records also exist in canonical train")
    present = promoted_ids & (existing_ids | auxiliary_ids)
    if present and present != promoted_ids:
        raise ValueError("canonical train contains a partial Round 06 promotion")
    if present == promoted_ids:
        actual = {
            record["id"]: record
            for record in train + auxiliary
            if record["id"] in promoted_ids
        }
        expected = {record["id"]: record for record in promoted}
        comparable = copy.deepcopy(actual)
        for record in comparable.values():
            metadata = record.get("metadata", {})
            for downstream_field in (
                "formal_training_status",
                "formal_training_exclusion_reason",
                "interlocutor_kind",
                "interlocutor_label",
            ):
                metadata.pop(downstream_field, None)
        mismatched = {
            record_id
            for record_id in promoted_ids
            if actual[record_id] != expected[record_id]
            and not _matches_authorized_downstream_review(
                comparable[record_id], expected[record_id]
            )
        }
        if mismatched:
            raise ValueError("existing Round 06 records differ from the approved sessions")
        train_sha256 = _text_sha256(train_path)
        if manifest.get("train", {}).get("count") != len(train):
            raise ValueError("manifest count does not match the already-promoted train split")
        if manifest.get("train", {}).get("sha256") != train_sha256:
            raise ValueError("manifest hash does not match the already-promoted train split")
        return {
            "status": "already_promoted",
            "train_count": len(train),
            "train_sha256": train_sha256,
            "record_ids": sorted(promoted_ids),
            "formal_record_count": len(promoted_ids & existing_ids),
            "auxiliary_record_count": len(promoted_ids & auxiliary_ids),
        }

    if len(train) != base_contract["train_count"]:
        raise ValueError("canonical train count does not match the approved promotion base")
    if _text_sha256(train_path) != base_contract["train_sha256"]:
        raise ValueError("canonical train hash does not match the approved promotion base")
    if existing_ids & promoted_ids:
        raise ValueError("promoted record IDs already exist in canonical train")

    new_candidates = [
        (record["id"], text) for record in promoted for text in _user_texts(record)
    ]
    split_references = [
        (record["id"], text, "existing_train")
        for record in train
        for text in _user_texts(record)
    ] + [
        (record["id"], text, "validation")
        for record in validation
        for text in _user_texts(record)
    ] + [
        (record["id"], text, "gold_v3")
        for record in gold_v3["prompts"]
        for text in _user_texts(record)
    ]
    promotion_overlap = _similarity_matches(new_candidates, split_references)
    if promotion_overlap:
        raise ValueError(f"Round 06 promotion overlaps protected data: {promotion_overlap}")

    final_train = train + promoted
    train_text = _jsonl_text(final_train)
    train_sha256 = _normalized_sha256_text(train_text)
    validation_sha256 = _text_sha256(validation_path)
    gold_audit = _gold_contamination_audit(
        train=final_train,
        validation=validation,
        gold_v3=gold_v3,
        gold_v21=gold_v21,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
    )
    if gold_audit["status"] != "clean":
        raise ValueError(f"Gold v3 contamination audit blocked promotion: {gold_audit}")

    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["schema_version"] = max(int(updated_manifest.get("schema_version", 0)), 7)
    updated_manifest["train"].update(
        count=len(final_train),
        sha256=train_sha256,
        augmentation_human_review_status=approval["status"],
    )
    distribution = updated_manifest["train"].setdefault("source_distribution", {})
    data_source = approval["promotion_contract"]["data_source"]
    distribution[data_source] = int(distribution.get(data_source, 0)) + len(promoted)
    check_key = approval["promotion_contract"].get("manifest_check_key")
    if not check_key:
        slug = re.sub(r"[^a-z0-9]+", "_", approval["approval_id"].lower()).strip("_")
        check_key = f"{slug}_promotion"
    updated_manifest.setdefault("checks", {})[check_key] = {
        "record_count": len(promoted),
        "assistant_turn_count": sum(
            message["role"] == "assistant"
            for record in promoted
            for message in record["messages"]
        ),
        "cumulative_prefix_record_count": 0,
        "protected_similarity_overlap_count": len(promotion_overlap),
        "gold_v3_contamination_reaudit": gold_audit["status"],
    }
    provenance = updated_manifest.setdefault("provenance", {})
    provenance.setdefault("augmentation_batch_approvals", {})[
        approval["approval_id"]
    ] = _relative(approved_path)
    updated_manifest.setdefault("augmentations", [])
    if any(item.get("id") == approval["approval_id"] for item in updated_manifest["augmentations"]):
        raise ValueError("manifest already contains this augmentation batch")
    updated_manifest["augmentations"].append(
        {
            "id": approval["approval_id"],
            "status": "approved_and_promoted",
            "record_count": len(promoted),
            "assistant_turn_count": approval["promotion_contract"]["assistant_turn_count"],
            "source_path": _relative(source_path),
            "approved_path": _relative(approved_path),
            "result_path": _relative(result_path),
            "reviewed_by": approval["approved_by"],
            "reviewed_at": approval["approved_at"],
        }
    )
    updated_manifest.setdefault("gold_v3", {})["contamination_reaudit"] = {
        "status": gold_audit["status"],
        "path": _relative(audit_path),
        "train_sha256": train_sha256,
        "gold_content_modified": False,
    }
    registry, review_manifest = _updated_authority_payloads(updated_manifest, approval)

    result = {
        "schema_version": 1,
        "status": "promoted",
        "approval_id": approval["approval_id"],
        "source_sessions_sha256": _text_sha256(source_path),
        "approved_sessions_sha256": _text_sha256(approved_path),
        "previous_train": {
            "count": base_contract["train_count"],
            "sha256": base_contract["train_sha256"],
        },
        "promoted": {
            "record_count": len(promoted),
            "assistant_turn_count": approval["promotion_contract"]["assistant_turn_count"],
            "record_ids": [record["id"] for record in promoted],
        },
        "result_train": {"count": len(final_train), "sha256": train_sha256},
        "validation": {"count": len(validation), "sha256": validation_sha256, "modified": False},
        "gold_v3": {
            "content_modified": False,
            "contamination_reaudit": gold_audit["status"],
            "contamination_audit_path": _relative(audit_path),
        },
    }

    _write_atomic(train_path, train_text)
    _write_atomic(audit_path, _json_text(gold_audit))
    _write_atomic(manifest_path, _json_text(updated_manifest))
    _write_atomic(RESEARCH_REGISTRY, _json_text(registry))
    _write_atomic(REVIEW_MANIFEST, _json_text(review_manifest))
    _write_atomic(result_path, _json_text(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-dir", type=Path, default=V4_DIR)
    parser.add_argument("--source", type=Path, default=SOURCE_SESSIONS)
    parser.add_argument("--approved", type=Path, default=APPROVED_SESSIONS)
    parser.add_argument("--gold-v3", type=Path, default=GOLD_V3)
    parser.add_argument("--gold-v21", type=Path, default=GOLD_V21)
    parser.add_argument("--audit", type=Path, default=GOLD_V3_AUDIT)
    parser.add_argument("--result", type=Path, default=PROMOTION_RESULT)
    args = parser.parse_args()
    try:
        result = promote(
            v4_dir=args.v4_dir.resolve(),
            source_path=args.source.resolve(),
            approved_path=args.approved.resolve(),
            gold_v3_path=args.gold_v3.resolve(),
            gold_v21_path=args.gold_v21.resolve(),
            audit_path=args.audit.resolve(),
            result_path=args.result.resolve(),
        )
    except ValueError as exc:
        print(f"promotion_blocked={exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
