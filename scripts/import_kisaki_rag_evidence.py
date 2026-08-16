#!/usr/bin/env python3
"""Import the frozen Kisaki R2 evidence into an isolated vector index."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("status") != "frozen" or not dataset.get("formal_use_allowed"):
        raise SystemExit("R2 evidence import requires a reviewed and frozen evaluation dataset")
    source = json.loads(args.documents.read_text(encoding="utf-8"))
    documents = source["documents"]
    expected_ids = {str(value) for row in dataset["questions"] for value in row.get("expected_doc_ids", [])}
    document_ids = {str(row["id"]) for row in documents}
    if expected_ids - document_ids:
        raise SystemExit(f"missing evidence ids: {sorted(expected_ids - document_ids)}")

    from knowledge.vector_db import get_vector_db
    database = get_vector_db()
    existing = {str(row.get("id")): row for row in database.metadata}
    present = document_ids & set(existing)
    if present:
        mismatched = [doc_id for doc_id in present if existing[doc_id].get("content_hash") != content_hash(next(row["content"] for row in documents if str(row["id"]) == doc_id))]
        if mismatched or present != document_ids:
            raise SystemExit("isolated R2 index contains a partial or mismatched evidence set")
        print(json.dumps({"status": "already_imported", "count": len(present)}))
        return 0
    database.add_documents(documents, kb_revision=f"kisaki-r2-{dataset['questions_sha256'][:12]}")
    database.flush()
    print(json.dumps({"status": "imported", "count": len(documents)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
