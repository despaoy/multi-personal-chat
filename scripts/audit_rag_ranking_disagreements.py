"""Build a review queue from measured disagreements; never modify gold labels.

No model call, no automatic approval, no test-set relabeling. Source texts and
both rankings are shown together so equivalent support and genuine regression
can be distinguished by a reviewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_queue(report, cases, documents):
    baseline = {row["id"]: row for row in report["variants"]["deterministic"]["per_query"]}
    semantic = {row["id"]: row for row in report["variants"]["cross_encoder"]["per_query"]}
    queries = {row["id"]: row for row in cases}
    if set(baseline) != set(semantic) or set(baseline) != set(queries):
        raise ValueError("reports and query corpus do not have identical case IDs")
    queue = []
    for key, old in baseline.items():
        new = semantic[key]
        if old["gold_ids"] != new["gold_ids"]:
            raise ValueError("paired variants have different gold annotations")
        if old["retrieved_ids"][:1] == new["retrieved_ids"][:1]:
            continue
        ids = list(dict.fromkeys([*old["retrieved_ids"], *new["retrieved_ids"], *old["gold_ids"]]))
        missing = set(ids) - set(documents)
        if missing:
            raise ValueError(f"report references missing index documents: {sorted(missing)}")
        queue.append(
            {
                "id": key,
                "query": queries[key]["query"],
                "category": queries[key].get("category"),
                "review_status": "pending",
                "review_judgment": None,
                "review_choices": [
                    "equivalent_support_not_in_gold",
                    "genuine_regression",
                    "genuine_improvement",
                    "ambiguous",
                ],
                "frozen_expected": queries[key]["expected"],
                "frozen_gold_ids": old["gold_ids"],
                "baseline_ids": old["retrieved_ids"],
                "semantic_ids": new["retrieved_ids"],
                "hit_at_1_delta": new["hit_at_1"] - old["hit_at_1"],
                "documents": [documents[doc_id] for doc_id in ids],
            }
        )
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "backend/data/eval/rag_retrieval/tsukiyashiro_kisaki.jsonl"
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=ROOT / "backend/data/knowledge/tsukiyashiro_kisaki/character_knowledge_index_v3",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    source = args.dataset.read_text(encoding="utf-8-sig")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != report["dataset_sha256"]:
        parser.error("query corpus no longer matches the measured report")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    documents = {}
    for path in sorted(args.index_root.glob("*/documents.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                doc = json.loads(line)
                if doc["id"] in documents:
                    raise ValueError("duplicate document ID in index")
                documents[doc["id"]] = doc
    queue = build_queue(report, cases, documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for row in queue:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"review_cases": len(queue), "status": "pending_not_relabelled"}))


if __name__ == "__main__":
    main()
