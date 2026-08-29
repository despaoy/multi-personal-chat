"""Offline sparse/entity A/B for the isolated multi-scale RAG experiment.

This script deliberately does not load or write the production FAISS index.
Both arms use the same query analyzer, BM25/entity recall depth, RRF fusion,
and deterministic reranker.  The only changed variable is the corpus/layout:

* baseline: approved fact/relation/event cards
* experiment: story/scene/card/evidence hierarchy + parent-aware scoring

The report contains a strict answer-bearing metric (card or its evidence) and
a looser context-coverage metric (a returned scene/story contains a gold card).
The latter is diagnostic and must not be presented as answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from knowledge.multiscale_rag import MultiScaleDocumentBuilder, MultiScaleRagService, OriginalTextExtractor  # noqa: E402
from knowledge.rag_pipeline.index import DomainIndex  # noqa: E402
from knowledge.rag_pipeline.query import QueryAnalyzer  # noqa: E402
from knowledge.rag_pipeline.registry import KnowledgeDomainConfig, get_default_registry  # noqa: E402
from knowledge.rag_pipeline.rerank import PipelineReranker  # noqa: E402
from knowledge.rag_pipeline.retrieval import HybridRetriever  # noqa: E402
from evaluate_rag_retrieval import resolve_expected_ids  # noqa: E402

EVAL_DIR = BACKEND_DIR / "data" / "eval" / "rag_retrieval"
REPORT_DIR = EVAL_DIR / "reports"
ENRICHED_SCENES = (
    BACKEND_DIR
    / "data"
    / "knowledge"
    / "tsukiyashiro_kisaki"
    / "scene_metadata_enriched"
    / "enriched_scenes.jsonl"
)


class SparseOnlyEmbeddingProvider:
    """A no-op provider: the in-memory indexes intentionally have no FAISS."""

    dimension = 1
    model_id = "disabled-for-sparse-entity-ab"

    def embed_query(self, _query: str) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


def _in_memory_index(domain_id: str, documents: Iterable[Any]) -> DomainIndex:
    index = DomainIndex(domain_id, Path("__in_memory_only__"), dimension=1)
    index.documents = list(documents)
    index._rebuild_entity_index()  # deterministic in-memory setup; no files are written
    index.bm25.build(index.documents)
    index.manifest = {
        "document_count": len(index.documents),
        "vector_dimension": None,
        "index_version": "offline-ab",
    }
    return index


def _experimental_config(base: KnowledgeDomainConfig, domain_id: str) -> KnowledgeDomainConfig:
    return KnowledgeDomainConfig(
        domain_id=domain_id,
        source_root=base.source_root,
        loader=lambda _root: [],
        document_types=["story", "scene", "fact", "relation", "event", "evidence"],
        aliases=dict(base.aliases),
        story_titles=list(base.story_titles),
        narrative_policy=base.narrative_policy,
        retrieval_defaults=base.retrieval_defaults,
        prompt_supplement=base.prompt_supplement,
        index_version="offline-multiscale-ab",
        enabled=True,
        display_name="multiscale offline A/B",
    )


class BaselineService:
    def __init__(self, config: KnowledgeDomainConfig, index: DomainIndex, provider: Any) -> None:
        self.analyzer = QueryAnalyzer([config])
        self.retriever = HybridRetriever(config, index, provider)
        self.reranker = PipelineReranker(cross_encoder_enabled=False)

    def retrieve(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        analysis = self.analyzer.analyze(query, domain_id=self.retriever.config.domain_id)
        recall_k = max(top_k * 8, 40)
        recalled = self.retriever.search(
            analysis,
            top_k=recall_k,
            recall_k=recall_k,
            mode="hybrid",
        )
        ranked = self.reranker.rerank(analysis, recalled, top_k=top_k)
        return {"results": [candidate.to_dict() for candidate in ranked]}


def _load_eval(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rank(result_sets: list[set[str]], expected: set[str], limit: int) -> int | None:
    for rank, represented_ids in enumerate(result_sets[:limit], 1):
        if represented_ids & expected:
            return rank
    return None


def _represented_ids(item: dict[str, Any], *, context: bool, cards: list[Any]) -> set[str]:
    doc_id = str(item.get("id") or "")
    doc_type = str(item.get("document_type") or "")
    metadata = item.get("metadata") or {}
    if doc_type in {"fact", "relation", "event"}:
        return {doc_id}
    if doc_type == "evidence":
        parent = str(metadata.get("parent_id") or item.get("parent_id") or "")
        return {parent} if parent else set()
    if not context:
        return set()
    if doc_type == "scene":
        return {card.id for card in cards if card.metadata.get("scene_id") == doc_id}
    if doc_type == "story":
        story_unit_id = str(metadata.get("story_unit_id") or "")
        return {card.id for card in cards if str(card.metadata.get("story_unit_id") or "") == story_unit_id}
    return set()


def _metric_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    n = len(rows)
    result: dict[str, float] = {}
    for k in (1, 3, 5):
        result[f"hit@{k}"] = round(
            sum(1 for row in rows if row[f"{prefix}_rank"] is not None and row[f"{prefix}_rank"] <= k) / n,
            4,
        ) if n else 0.0
    result["mrr"] = round(
        sum(1.0 / row[f"{prefix}_rank"] for row in rows if row[f"{prefix}_rank"] is not None) / n,
        4,
    ) if n else 0.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="多粒度 RAG 稀疏/实体离线 A/B")
    parser.add_argument("--domain", default="tsukiyashiro_kisaki")
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_file) if args.eval_file else EVAL_DIR / f"{args.domain}.jsonl"
    entries = _load_eval(eval_path)
    base_config = get_default_registry().require(args.domain)
    baseline_docs = base_config.loader(base_config.source_root)
    expected_map: dict[str, set[str]] = {}
    for entry in entries:
        spec = entry.get("expected") or {}
        explicit = spec.get("expected_document_ids")
        expected_map[entry["id"]] = set(map(str, explicit)) if explicit else resolve_expected_ids(
            baseline_docs, spec.get("criteria")
        )

    experiment_domain = f"{args.domain}_multiscale_sparse_ab"
    extractor = OriginalTextExtractor(REPO_ROOT)
    build = MultiScaleDocumentBuilder(
        domain_id=experiment_domain,
        index_version="offline-multiscale-ab",
        aliases=base_config.aliases,
        source_extractor=extractor,
    ).build(base_config.source_root, ENRICHED_SCENES)
    experiment_config = _experimental_config(base_config, experiment_domain)
    provider = SparseOnlyEmbeddingProvider()
    baseline = BaselineService(base_config, _in_memory_index(base_config.domain_id, baseline_docs), provider)
    experiment_index = _in_memory_index(experiment_domain, build.documents)
    experiment = MultiScaleRagService(
        experiment_config,
        experiment_index,
        provider,
        source_extractor=extractor,
        reranker=PipelineReranker(cross_encoder_enabled=False),
    )
    experiment_cards = [
        doc for doc in build.documents if doc.document_type in {"fact", "relation", "event"}
    ]
    gate = QueryAnalyzer([base_config])

    rows: list[dict[str, Any]] = []
    latency: dict[str, list[float]] = {"baseline": [], "experiment": []}
    raw_success = 0
    answer_cases = 0
    for entry in entries:
        expected = expected_map[entry["id"]]
        is_no_answer = not expected
        routed = bool(gate.analyze(entry["query"]).matched_domains)

        started = time.perf_counter()
        baseline_bundle = None if is_no_answer and not routed else baseline.retrieve(entry["query"], top_k=args.top_k)
        latency["baseline"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        experiment_bundle = None if is_no_answer and not routed else experiment.retrieve(
            entry["query"], top_k=args.top_k, raw_text=not is_no_answer
        )
        latency["experiment"].append((time.perf_counter() - started) * 1000)

        baseline_items = (baseline_bundle or {}).get("results") or []
        experiment_items = (experiment_bundle or {}).get("results") or []
        if is_no_answer:
            baseline_rank = 1 if not baseline_items else None
            experiment_rank = 1 if not experiment_items else None
            experiment_context_rank = experiment_rank
        else:
            answer_cases += 1
            raw_success += int(bool((experiment_bundle or {}).get("raw_excerpt")))
            baseline_sets = [{str(item.get("id"))} for item in baseline_items]
            strict_sets = [
                _represented_ids(item, context=False, cards=experiment_cards) for item in experiment_items
            ]
            context_sets = [
                _represented_ids(item, context=True, cards=experiment_cards) for item in experiment_items
            ]
            baseline_rank = _rank(baseline_sets, expected, args.top_k)
            experiment_rank = _rank(strict_sets, expected, args.top_k)
            experiment_context_rank = _rank(context_sets, expected, args.top_k)

        rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "query": entry["query"],
                "expected_ids": sorted(expected),
                "baseline_rank": baseline_rank,
                "experiment_rank": experiment_rank,
                "experiment_context_rank": experiment_context_rank,
                "baseline_ids": [str(item.get("id")) for item in baseline_items],
                "experiment_ids": [str(item.get("id")) for item in experiment_items],
            }
        )

    baseline_metrics = _metric_summary(rows, "baseline")
    experiment_metrics = _metric_summary(rows, "experiment")
    context_metrics = _metric_summary(rows, "experiment_context")
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "n": len(subset),
            "baseline": _metric_summary(subset, "baseline"),
            "experiment": _metric_summary(subset, "experiment"),
            "experiment_context_coverage": _metric_summary(subset, "experiment_context"),
        }

    improved = [row for row in rows if (row["experiment_rank"] or 999) < (row["baseline_rank"] or 999)]
    regressed = [row for row in rows if (row["experiment_rank"] or 999) > (row["baseline_rank"] or 999)]
    report = {
        "scope": "sparse_entity_deterministic_rerank_pre_experiment",
        "warning": "FAISS/semantic embeddings are disabled; this is not the full production RAG accuracy.",
        "eval_file": str(eval_path),
        "entries": len(entries),
        "top_k": args.top_k,
        "corpora": {
            "baseline": {"total": len(baseline_docs), "counts": dict(_counts(baseline_docs))},
            "experiment": {
                "total": len(build.documents),
                "counts": build.counts,
                "exact_evidence_matches": build.exact_evidence_matches,
            },
        },
        "baseline": {**baseline_metrics, "mean_latency_ms": round(mean(latency["baseline"]), 2)},
        "experiment_strict": {
            **experiment_metrics,
            "mean_latency_ms": round(mean(latency["experiment"]), 2),
            "raw_available_rate": round(raw_success / answer_cases, 4) if answer_cases else 0.0,
        },
        "experiment_context_coverage": context_metrics,
        "delta_strict": {
            key: round(experiment_metrics[key] - baseline_metrics[key], 4)
            for key in ("hit@1", "hit@3", "hit@5", "mrr")
        },
        "categories": categories,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "improved": improved,
        "regressed": regressed,
        "rows": rows,
        "promotion_decision": "manual_review_required_after_full_vector_ab",
    }

    print(f"评测题数: {len(entries)}  baseline={len(baseline_docs)} docs  experiment={len(build.documents)} docs")
    print("当前模式: BM25 + entity + RRF + deterministic rerank（无向量）")
    print(
        "baseline          "
        f"Hit@1={baseline_metrics['hit@1']:.3f} Hit@3={baseline_metrics['hit@3']:.3f} "
        f"Hit@5={baseline_metrics['hit@5']:.3f} MRR={baseline_metrics['mrr']:.3f}"
    )
    print(
        "multiscale strict "
        f"Hit@1={experiment_metrics['hit@1']:.3f} Hit@3={experiment_metrics['hit@3']:.3f} "
        f"Hit@5={experiment_metrics['hit@5']:.3f} MRR={experiment_metrics['mrr']:.3f}"
    )
    print(
        "context coverage  "
        f"Hit@1={context_metrics['hit@1']:.3f} Hit@3={context_metrics['hit@3']:.3f} "
        f"Hit@5={context_metrics['hit@5']:.3f} MRR={context_metrics['mrr']:.3f}"
    )
    print(f"严格指标改善={len(improved)} 退化={len(regressed)} 原文可用率={report['experiment_strict']['raw_available_rate']:.3f}")

    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"{args.domain}_multiscale_sparse_ab_{timestamp}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {report_path}")
    return 0


def _counts(documents: Iterable[Any]) -> defaultdict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for doc in documents:
        counts[str(doc.document_type)] += 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
