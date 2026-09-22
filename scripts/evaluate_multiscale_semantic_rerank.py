"""Real-model, paired RAG ranking ablation on the existing 68-query dev set.

Both variants use the same routed indexes and embedding provider. Gold matching
is independent of ranking. Unresolved annotations are reported, never silently
counted as misses or dropped without a count. This is a development evaluation,
not an unseen test benchmark and not a measurement of generated answer quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime  # noqa: E402
from knowledge.reranker import CrossEncoderReranker, RerankConfig  # noqa: E402
from knowledge.retrieval_core.rerank import PipelineReranker  # noqa: E402


def matches(doc, criteria):
    if not criteria:
        return False
    for key, expected in criteria.items():
        if key == "either_of":
            if not any(matches(doc, alternative) for alternative in expected):
                return False
        elif key == "title_contains":
            if expected not in doc.title:
                return False
        elif key == "value_contains":
            if expected not in str(doc.metadata.get("value", "")):
                return False
        elif key == "entities_contain":
            if expected not in doc.entities:
                return False
        else:
            field = "relation" if key == "relation_type" else key
            actual = getattr(doc, field, doc.metadata.get(field))
            if actual != expected:
                return False
    return True


def resolve_gold(case, documents):
    if case.get("category") == "no_answer":
        return set(), "negative"
    expected = case["expected"]
    if "expected_document_ids" in expected:
        ids = set(expected["expected_document_ids"])
        if not ids or not ids <= set(documents):
            return ids, "unresolved"
    else:
        ids = {key for key, doc in documents.items() if matches(doc, expected.get("criteria", {}))}
    return ids, "resolved" if ids else "unresolved"


def summarize(rows):
    scored = [row for row in rows if row["gold_status"] == "resolved"]
    negative = [row for row in rows if row["gold_status"] == "negative"]
    negative_decisions = [row for row in negative if "abstained" in row]
    return {
        "queries": len(rows),
        "answerable_scored": len(scored),
        "unresolved_annotations": sum(row["gold_status"] == "unresolved" for row in rows),
        "hit_at_1": statistics.mean(row["hit_at_1"] for row in scored) if scored else None,
        "hit_at_5": statistics.mean(row["hit_at_5"] for row in scored) if scored else None,
        "mrr_at_5": statistics.mean(row["rr"] for row in scored) if scored else None,
        "recall_at_5": statistics.mean(row["recall"] for row in scored) if scored else None,
        "ndcg_at_5": statistics.mean(row["ndcg"] for row in scored) if scored else None,
        "negative_queries": len(negative),
        "negative_domain_hit_rate": statistics.mean(bool(row["retrieved_ids"]) for row in negative)
        if negative
        else None,
        "negative_nonabstention_rate": statistics.mean(not row["abstained"] for row in negative_decisions)
        if negative_decisions
        else None,
        "negative_abstention_decisions_scored": len(negative_decisions),
        "median_latency_ms": statistics.median(row["latency_ms"] for row in rows) if rows else None,
        "semantic_queries": sum(row["actual_rerank_method"] == "cross_encoder" for row in rows),
        "notes": "Ranking metrics use results before abstention; logits need separate validation calibration.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "backend/data/eval/rag_retrieval/tsukiyashiro_kisaki.jsonl"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--text-view", choices=("content", "summary", "embedding_text"), default="content")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    source = args.dataset.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        parser.error("duplicate query IDs")
    runtime = MultiScaleRagRuntime()
    if not runtime.is_available():
        raise RuntimeError("index unavailable")
    runtime._provider.embed_query("检索预热")  # Fail loudly: no fake or fallback embeddings.
    encoder = CrossEncoderReranker(
        RerankConfig(
            model_name=str(args.model_path.resolve()),
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            allow_download=False,
            fallback_to_original=False,
        )
    )
    if not encoder._load_model():
        raise RuntimeError("cross encoder failed to load; no semantic result will be reported")
    variants = {
        "deterministic": PipelineReranker(cross_encoder_enabled=False),
        "cross_encoder": PipelineReranker(cross_encoder=encoder, text_view=args.text_view),
    }
    all_results = {}
    for name, reranker in variants.items():
        runtime._service.reranker = reranker
        rows = []
        for index, case in enumerate(cases):
            gold, status = resolve_gold(case, runtime._service.by_id)
            started = time.perf_counter()
            bundle = runtime.retrieve_with_citations(case["query"], top_k=5, domain_id=case.get("domain_id"))
            results = (bundle or {}).get("results", [])
            ids = [item["id"] for item in results]
            actual = (bundle or {}).get("rerank_method", "none")
            if name == "cross_encoder" and results and actual != "cross_encoder":
                raise RuntimeError(f"semantic reranking silently fell back on {case['id']}")
            relevance = [key in gold for key in ids]
            first = next((rank for rank, hit in enumerate(relevance, 1) if hit), None)
            dcg = sum(int(hit) / math.log2(rank + 1) for rank, hit in enumerate(relevance, 1))
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(gold)) + 1))
            rows.append(
                {
                    "id": case["id"],
                    "category": case.get("category"),
                    "gold_status": status,
                    "gold_ids": sorted(gold),
                    "retrieved_ids": ids,
                    "hit_at_1": int(bool(relevance) and relevance[0]),
                    "hit_at_5": int(any(relevance)),
                    "rr": 1 / first if first else 0.0,
                    "recall": len(gold & set(ids)) / len(gold) if gold else 0.0,
                    "ndcg": dcg / ideal if ideal else 0.0,
                    "actual_rerank_method": actual,
                    "abstained": bool((bundle or {}).get("abstained", True)),
                    "scores": [item.get("rerank_score") for item in results],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            print(f"{name}: {index + 1}/{len(cases)} {case['id']}", flush=True)
        all_results[name] = {"summary": summarize(rows), "per_query": rows}
    report = {
        "evaluation_status": "real_model_development_ablation_not_held_out_test",
        "dataset_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "model_path": str(args.model_path.resolve()),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "text_view": args.text_view,
        "annotation_statuses": sorted(
            {case.get("annotation_status", "existing_project_development_labels") for case in cases}
        ),
        "index": runtime.stats(),
        "variants": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps({key: value["summary"] for key, value in all_results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
