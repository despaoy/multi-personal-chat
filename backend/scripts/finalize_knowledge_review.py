"""Apply approved P5 decisions to candidate knowledge cards.

The decision JSON is authoritative. Reviewers edit that document and rerun
this script to rebuild approved JSONL files and summaries.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from knowledge.game_rag.knowledge_candidate import (  # noqa: E402
    KnowledgeReviewDocument,
    load_documents_jsonl,
    save_documents_jsonl,
    save_knowledge_review,
)
from knowledge.game_rag.models import EventDocument, FactDocument, RelationDocument  # noqa: E402

BASE = BACKEND_DIR / "data" / "knowledge" / "tsukiyashiro_kisaki" / "knowledge_candidate_review"
CANDIDATE_PATHS = {
    "fact": BASE / "facts_candidate.jsonl",
    "relation": BASE / "relations_candidate.jsonl",
    "event": BASE / "events_candidate.jsonl",
}
APPROVED_PATHS = {
    "fact": BASE / "facts_approved.jsonl",
    "relation": BASE / "relations_approved.jsonl",
    "event": BASE / "events_approved.jsonl",
}
REVIEW_PATH = BASE / "knowledge_review.json"
DECISIONS_PATH = BASE / "knowledge_manual_decisions.json"
REPORT_PATH = BASE / "knowledge_approval_report.json"
QUALITY_REPORT_PATH = BASE / "knowledge_quality_report.json"

DOCUMENT_ADAPTERS = {
    "fact": TypeAdapter(FactDocument),
    "relation": TypeAdapter(RelationDocument),
    "event": TypeAdapter(EventDocument),
}
IMMUTABLE_PATCH_FIELDS = {
    "id",
    "document_type",
    "story",
    "source",
    "evidence_text",
    "review_status",
}


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for document_type, path in CANDIDATE_PATHS.items():
        for raw in load_documents_jsonl(path):
            doc = DOCUMENT_ADAPTERS[document_type].validate_python(raw).model_dump(mode="json")
            if doc["document_type"] != document_type:
                raise ValueError(f"candidate type mismatch: {doc['id']}")
            candidates.append(doc)
    ids = [doc["id"] for doc in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate files contain duplicate ids")
    return candidates


def _load_decisions(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    reviewer = str(document.get("reviewer") or "").strip()
    if document.get("review_status") != "approved" or not reviewer:
        raise ValueError("manual decisions must be approved and have a reviewer")

    decisions = document.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("manual decisions must contain a decisions list")
    by_id: dict[str, dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("manual decision entries must be objects")
        card_id = str(item.get("card_id") or "")
        if not card_id or card_id in by_id:
            raise ValueError(f"invalid or duplicate decision card_id: {card_id!r}")
        if item.get("decision") not in {"approved", "rejected"}:
            raise ValueError(f"invalid decision for {card_id}")
        if not isinstance(item.get("patch"), dict):
            raise ValueError(f"patch must be an object for {card_id}")
        by_id[card_id] = item

    candidate_ids = {doc["id"] for doc in candidates}
    if set(by_id) != candidate_ids or document.get("total_candidates") != len(candidates):
        missing = sorted(candidate_ids - set(by_id))
        unknown = sorted(set(by_id) - candidate_ids)
        raise ValueError(f"decision coverage mismatch: missing={missing[:5]} unknown={unknown[:5]}")
    return document, by_id


def _apply_patch(
    doc: dict[str, Any],
    patch: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    patch = dict(patch)
    evidence_from = patch.pop("evidence_from_card_id", None)
    forbidden = IMMUTABLE_PATCH_FIELDS.intersection(patch)
    if forbidden:
        raise ValueError(f"card {doc['id']} patch changes immutable fields: {sorted(forbidden)}")
    if evidence_from is not None:
        source_doc = candidates_by_id.get(str(evidence_from))
        if source_doc is None:
            raise ValueError(f"card {doc['id']} references unknown evidence card {evidence_from}")
        patch.update(
            evidence_text=source_doc["evidence_text"],
            story=source_doc["story"],
            source=source_doc["source"],
        )
    updated = {**doc, **patch, "review_status": "approved"}
    return DOCUMENT_ADAPTERS[doc["document_type"]].validate_python(updated).model_dump(mode="json")


def _build_approved(
    candidates: list[dict[str, Any]],
    decision_document: dict[str, Any],
    decisions_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidates_by_id = {doc["id"]: doc for doc in candidates}
    approved: dict[str, list[dict[str, Any]]] = {"fact": [], "relation": [], "event": []}
    for doc in candidates:
        decision = decisions_by_id[doc["id"]]
        if decision.get("document_type") != doc["document_type"]:
            raise ValueError(f"decision type mismatch for {doc['id']}")
        if decision["decision"] == "approved":
            approved[doc["document_type"]].append(_apply_patch(doc, decision["patch"], candidates_by_id))

    manual_ids: set[str] = set()
    for addition in decision_document.get("manual_additions") or []:
        if not isinstance(addition, dict) or not isinstance(addition.get("card"), dict):
            raise ValueError("manual additions must contain full card objects")
        raw = addition["card"]
        document_type = raw.get("document_type")
        if document_type not in DOCUMENT_ADAPTERS:
            raise ValueError(f"invalid manual addition type: {document_type!r}")
        card = DOCUMENT_ADAPTERS[document_type].validate_python(raw).model_dump(mode="json")
        if card["review_status"] != "approved":
            raise ValueError(f"manual addition must be approved: {card['id']}")
        if card["id"] in candidates_by_id or card["id"] in manual_ids:
            raise ValueError(f"duplicate manual addition id: {card['id']}")
        manual_ids.add(card["id"])
        approved[document_type].append(card)
    return approved


def _validate_sources(approved: dict[str, list[dict[str, Any]]]) -> None:
    line_counts: dict[str, int] = {}
    ids: set[str] = set()
    for documents in approved.values():
        for doc in documents:
            if doc["id"] in ids:
                raise ValueError(f"approved output contains duplicate id: {doc['id']}")
            ids.add(doc["id"])
            source = doc["source"]
            path = source["source_path"]
            full_path = REPO_ROOT / path
            if path not in line_counts:
                if not full_path.is_file():
                    raise ValueError(f"source file does not exist: {path}")
                line_counts[path] = len(full_path.read_text(encoding="utf-8").splitlines())
            if not 1 <= source["line_start"] <= source["line_end"] <= line_counts[path]:
                raise ValueError(f"source range is invalid for {doc['id']}")
            if not doc["evidence_text"].strip():
                raise ValueError(f"evidence is empty for {doc['id']}")

    wrong_sibling_directions = {("琉璃", "哥哥", "妃"), ("妃", "妹妹", "琉璃")}
    if any(
        (doc["subject"], doc["relation"], doc["target"]) in wrong_sibling_directions for doc in approved["relation"]
    ):
        raise ValueError("approved relations contain a reversed sibling relation")


def _update_review(decisions_by_id: dict[str, dict[str, Any]], reviewer: str) -> KnowledgeReviewDocument:
    raw = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    if {item["card_id"] for item in raw["card_reviews"]} != set(decisions_by_id):
        raise ValueError("knowledge review and manual decisions cover different cards")
    for item in raw["card_reviews"]:
        decision = decisions_by_id[item["card_id"]]
        item["review_status"] = decision["decision"]
        item["reviewer"] = reviewer
        item["notes"] = str(decision.get("reason") or "")
    raw.update(
        reviewer=reviewer,
        review_status="approved",
        notes="P5 candidates were reviewed against source text; approved cards are the production knowledge output.",
    )
    return KnowledgeReviewDocument.model_validate(raw)


def main() -> None:
    candidates = _load_candidates()
    decision_document, decisions_by_id = _load_decisions(candidates)
    approved = _build_approved(candidates, decision_document, decisions_by_id)
    _validate_sources(approved)
    review = _update_review(decisions_by_id, decision_document["reviewer"])

    for document_type, path in APPROVED_PATHS.items():
        documents = [DOCUMENT_ADAPTERS[document_type].validate_python(doc) for doc in approved[document_type]]
        save_documents_jsonl(path, documents)
    save_knowledge_review(REVIEW_PATH, review)

    decision_counts = Counter(item["decision"] for item in decisions_by_id.values())
    patch_counts = Counter(
        item["document_type"] for item in decisions_by_id.values() if item["decision"] == "approved" and item["patch"]
    )
    report = {
        "schema_version": 1,
        "reviewer": decision_document["reviewer"],
        "source_candidate_count": len(candidates),
        "manual_added_count": len(decision_document.get("manual_additions") or []),
        "decision_counts": dict(sorted(decision_counts.items())),
        "approved_counts": {key: len(value) for key, value in approved.items()},
        "approved_total": sum(len(value) for value in approved.values()),
        "patched_counts": dict(sorted(patch_counts.items())),
        "manual_question_count": 0,
        "review_status": "approved",
        "notes": "P5 approval summary; embedding and indexing are separate P6 steps.",
    }
    _write_json(REPORT_PATH, report)

    if QUALITY_REPORT_PATH.is_file():
        quality = json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
        quality["manual_review_final"] = {
            "review_status": "approved",
            "reviewer": decision_document["reviewer"],
            "source_candidate_count": len(candidates),
            "approved_counts": report["approved_counts"],
            "approved_total": report["approved_total"],
            "rejected_total": decision_counts["rejected"],
            "patched_counts": report["patched_counts"],
            "manual_question_count": 0,
        }
        _write_json(QUALITY_REPORT_PATH, quality)

    print(
        "P5 finalized: "
        f"approved={report['approved_total']} rejected={decision_counts['rejected']} "
        f"facts={len(approved['fact'])} relations={len(approved['relation'])} "
        f"events={len(approved['event'])}"
    )


if __name__ == "__main__":
    main()
