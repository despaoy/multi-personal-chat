#!/usr/bin/env python3
"""Apply approved surface-only text normalizations to active Kisaki data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHARACTER_DIR = ROOT / "backend/data/character_dialogues"
EXPERIMENT_DIR = CHARACTER_DIR / "experiments"
V4_DIR = EXPERIMENT_DIR / "v4"

DEFAULT_LEDGER = V4_DIR / "text_normalizations.json"
DEFAULT_SFT = CHARACTER_DIR / "tsukiyashiro_kisaki_sft.json"
DEFAULT_SFT_FULL = CHARACTER_DIR / "tsukiyashiro_kisaki_sft_full.json"
DEFAULT_TRAIN = V4_DIR / "train.jsonl"
DEFAULT_MANIFEST = V4_DIR / "canonical_dataset_manifest.json"
DEFAULT_RAG_DOCUMENTS = EXPERIMENT_DIR / "research/character_rag_seed_documents.json"
DEFAULT_RAG_RETRIEVAL = EXPERIMENT_DIR / "research/character_rag_retrieval_eval.json"
DEFAULT_RAG_V2 = EXPERIMENT_DIR / "research/kisaki_rag_eval_v2_candidates.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _sha256_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalization_note(item: dict[str, Any]) -> dict[str, str]:
    return {
        "id": item["id"],
        "policy": "surface_only_no_semantic_edit",
        "source_event_id": item["source_event_id"],
        "original": item["original"],
        "reason": item["reason"],
    }


def _attach_note(metadata: dict[str, Any], item: dict[str, Any]) -> None:
    notes = metadata.setdefault("surface_normalizations", [])
    note = _normalization_note(item)
    existing = {entry.get("id") for entry in notes if isinstance(entry, dict)}
    if note["id"] not in existing:
        notes.append(note)


def _replace_record_assistant(
    record: dict[str, Any],
    item: dict[str, Any],
    *,
    messages_key: str,
    role_key: str,
    content_key: str,
    assistant_role: str,
) -> int:
    replacements = 0
    for message in record[messages_key]:
        if message.get(role_key) != assistant_role:
            continue
        content = message.get(content_key)
        if item.get("match_mode", "exact") == "substring":
            original_count = content.count(item["original"])
            normalized_count = content.count(item["normalized"])
            if original_count == 1 and normalized_count == 0:
                message[content_key] = content.replace(
                    item["original"], item["normalized"]
                )
                replacements += 1
            elif original_count == 0 and normalized_count == 1:
                replacements += 1
        elif content == item["original"]:
            message[content_key] = item["normalized"]
            replacements += 1
        elif content == item["normalized"]:
            replacements += 1
    if replacements != 1:
        raise ValueError(
            f"{item['id']} expected exactly one assistant target in {record['id']}, "
            f"found {replacements}"
        )
    _attach_note(record.setdefault("metadata", {}), item)
    return replacements


def normalize_sft_rows(
    rows: list[dict[str, Any]], items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = copy.deepcopy(rows)
    by_id = {row["id"]: row for row in output}
    for item in items:
        record = by_id.get(item["record_id"])
        if record is None:
            raise ValueError(f"{item['record_id']} missing from SFT records")
        _replace_record_assistant(
            record,
            item,
            messages_key="conversations",
            role_key="from",
            content_key="value",
            assistant_role="assistant",
        )
    return output


def _update_sft(path: Path, items: list[dict[str, Any]]) -> int:
    rows = normalize_sft_rows(_load_json(path), items)
    _write_atomic(path, _json_text(rows))
    return len(items)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _update_train(path: Path, items: list[dict[str, Any]]) -> tuple[int, str, str]:
    original_text = path.read_text(encoding="utf-8")
    rows = _load_jsonl(path)
    by_id = {row["id"]: row for row in rows}
    for item in items:
        record = by_id.get(item["record_id"])
        if record is None:
            raise ValueError(f"{item['record_id']} missing from {path}")
        _replace_record_assistant(
            record,
            item,
            messages_key="messages",
            role_key="role",
            content_key="content",
            assistant_role="assistant",
        )
    updated_text = _jsonl_text(rows)
    _write_atomic(path, updated_text)
    return len(rows), _sha256_text(original_text), _sha256_text(updated_text)


def _update_rag_documents(path: Path, item: dict[str, Any]) -> int:
    payload = _load_json(path)
    documents = [doc for doc in payload["documents"] if doc["id"] == item["document_id"]]
    if len(documents) != 1:
        raise ValueError(f"{item['document_id']} missing or duplicated in {path}")
    document = documents[0]
    content = document["content"]
    if item["original"] in content:
        document["content"] = content.replace(item["original"], item["normalized"])
    elif item["normalized"] not in content:
        raise ValueError(f"{item['id']} expected text missing from {path}")
    _attach_note(document.setdefault("metadata", {}), item)
    _write_atomic(path, _json_text(payload))
    return 1


def _replace_recursive(value: Any, original: str, normalized: str) -> tuple[Any, int]:
    if isinstance(value, str):
        count = value.count(original)
        return value.replace(original, normalized), count
    if isinstance(value, list):
        output = []
        total = 0
        for item in value:
            updated, count = _replace_recursive(item, original, normalized)
            output.append(updated)
            total += count
        return output, total
    if isinstance(value, dict):
        output = {}
        total = 0
        for key, item in value.items():
            updated, count = _replace_recursive(item, original, normalized)
            output[key] = updated
            total += count
        return output, total
    return value, 0


def _update_derived_json(
    path: Path,
    item: dict[str, Any],
    *,
    expected_replacements: int,
) -> int:
    payload = _load_json(path)
    updated, count = _replace_recursive(payload, item["original"], item["normalized"])
    if count == 0:
        _, normalized_count = _replace_recursive(payload, item["normalized"], item["normalized"])
        if normalized_count != expected_replacements:
            raise ValueError(f"{item['id']} expected normalized text missing from {path}")
        count = normalized_count
    if count != expected_replacements:
        raise ValueError(
            f"{item['id']} expected {expected_replacements} replacements in {path}, found {count}"
        )
    _write_atomic(path, _json_text(updated))
    return count


def _update_manifest(
    path: Path,
    *,
    ledger_path: Path,
    train_count: int,
    previous_train_sha256: str,
    train_sha256: str,
    item_count: int,
) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest["train"]["count"] != train_count:
        raise ValueError("canonical manifest train count does not match train.jsonl")
    declared_hash = manifest["train"]["sha256"]
    if declared_hash not in {previous_train_sha256, train_sha256}:
        raise ValueError("canonical manifest does not match pre-normalization train data")
    existing_revision = manifest.get("surface_normalization_revision", {})
    if (
        previous_train_sha256 == train_sha256
        and declared_hash == train_sha256
        and manifest.get("status") == "frozen"
        and existing_revision.get("policy_id")
        == "KISAKI-SURFACE-NORMALIZATION-20260821"
        and existing_revision.get("result_train_sha256") == train_sha256
        and existing_revision.get("status") == "applied_and_refrozen"
    ):
        return manifest

    updated = copy.deepcopy(manifest)
    updated["schema_version"] = max(int(updated.get("schema_version", 0)) + 1, 12)
    updated["status"] = "frozen_data_pending_gold"
    updated["train"]["sha256"] = train_sha256
    updated["surface_normalization_revision"] = {
        "policy_id": "KISAKI-SURFACE-NORMALIZATION-20260821",
        "status": "applied_pending_gold_reaudit",
        "semantic_content_modified": False,
        "raw_sources_modified": False,
        "item_count": item_count,
        "ledger_path": ledger_path.resolve().relative_to(ROOT).as_posix(),
        "ledger_sha256": _sha256_text(ledger_path.read_text(encoding="utf-8")),
        "previous_train_sha256": previous_train_sha256,
        "result_train_sha256": train_sha256,
    }
    blockers = [
        blocker
        for blocker in updated.get("freeze_blockers", [])
        if blocker != "gold_v3_reaudit_after_surface_normalization"
    ]
    blockers.append("gold_v3_reaudit_after_surface_normalization")
    updated["freeze_blockers"] = blockers
    gold = updated.setdefault("gold_v3", {})
    gold["status"] = "stale_after_surface_normalization"
    gold["formal_use_allowed"] = False
    gold.setdefault("contamination_reaudit", {}).update(
        status="stale_after_surface_normalization",
        expected_train_sha256=train_sha256,
    )
    _write_atomic(path, _json_text(updated))
    return updated


def apply_normalizations(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    sft_path: Path = DEFAULT_SFT,
    sft_full_path: Path = DEFAULT_SFT_FULL,
    train_path: Path = DEFAULT_TRAIN,
    manifest_path: Path = DEFAULT_MANIFEST,
    rag_documents_path: Path = DEFAULT_RAG_DOCUMENTS,
    rag_retrieval_path: Path = DEFAULT_RAG_RETRIEVAL,
    rag_v2_path: Path = DEFAULT_RAG_V2,
) -> dict[str, Any]:
    ledger = _load_json(ledger_path)
    if ledger.get("policy") != "surface_only_no_semantic_edit":
        raise ValueError("unsupported normalization policy")
    items = ledger["items"]
    train_items = [item for item in items if "canonical_train" in item["targets"]]
    rag_items = [item for item in items if "rag_seed_documents" in item["targets"]]
    if len(rag_items) != 1:
        raise ValueError("expected exactly one active RAG normalization")

    sft_count = _update_sft(sft_path, items)
    sft_full_count = _update_sft(sft_full_path, items)
    train_count, previous_hash, train_hash = _update_train(train_path, train_items)
    rag_item = rag_items[0]
    rag_document_count = _update_rag_documents(rag_documents_path, rag_item)
    rag_retrieval_count = _update_derived_json(
        rag_retrieval_path, rag_item, expected_replacements=1
    )
    rag_v2_count = _update_derived_json(rag_v2_path, rag_item, expected_replacements=2)
    manifest = _update_manifest(
        manifest_path,
        ledger_path=ledger_path,
        train_count=train_count,
        previous_train_sha256=previous_hash,
        train_sha256=train_hash,
        item_count=len(items),
    )
    already_applied = (
        manifest["status"] == "frozen"
        and manifest.get("surface_normalization_revision", {}).get("status")
        == "applied_and_refrozen"
    )
    return {
        "status": "already_applied" if already_applied else "applied_pending_gold_reaudit",
        "policy": ledger["policy"],
        "normalization_count": len(items),
        "sft_records_updated": sft_count,
        "sft_full_records_updated": sft_full_count,
        "train_records_updated": len(train_items),
        "train_count": train_count,
        "previous_train_sha256": previous_hash,
        "train_sha256": train_hash,
        "rag_documents_updated": rag_document_count,
        "rag_retrieval_replacements": rag_retrieval_count,
        "rag_v2_replacements": rag_v2_count,
        "manifest_status": manifest["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    try:
        result = apply_normalizations(ledger_path=args.ledger.resolve())
    except (KeyError, TypeError, ValueError) as exc:
        print(f"surface_normalization_blocked={exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
