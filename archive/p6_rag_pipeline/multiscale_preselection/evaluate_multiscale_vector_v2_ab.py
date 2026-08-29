"""Build scale-specific texts/vectors and run a fair full-vector RAG A/B."""

from __future__ import annotations

import argparse
import hashlib
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

from evaluate_rag_retrieval import resolve_expected_ids  # noqa: E402
from knowledge.multiscale_rag import (  # noqa: E402
    LocalMeanPoolingEmbeddingProvider,
    OriginalTextExtractor,
    ReprocessedMultiScaleBuilder,
    ReprocessedMultiScaleService,
)
from knowledge.multiscale_rag.vector_runtime import attach_vectors, write_vector_artifacts  # noqa: E402
from knowledge.rag_pipeline.index import DomainIndex  # noqa: E402
from knowledge.rag_pipeline.query import QueryAnalyzer  # noqa: E402
from knowledge.rag_pipeline.registry import KnowledgeDomainConfig, get_default_registry  # noqa: E402
from knowledge.rag_pipeline.rerank import PipelineReranker  # noqa: E402
from knowledge.rag_pipeline.retrieval import HybridRetriever  # noqa: E402

EVAL_DIR = BACKEND_DIR / "data" / "eval" / "rag_retrieval"
REPORT_DIR = EVAL_DIR / "reports"
KNOWLEDGE_ROOT = BACKEND_DIR / "data" / "knowledge" / "tsukiyashiro_kisaki"
ENRICHED_SCENES = KNOWLEDGE_ROOT / "scene_metadata_enriched" / "enriched_scenes.jsonl"
DEFAULT_ARTIFACT_ROOT = KNOWLEDGE_ROOT / "rag_vector_ab_multiscale_v2"


def _config_for_experiment(base: KnowledgeDomainConfig, domain_id: str) -> KnowledgeDomainConfig:
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
        index_version="multiscale-semantic-v2",
        enabled=True,
        display_name="multi-scale semantic v2",
    )


def _memory_index(domain_id: str, documents: list, vectors: np.ndarray) -> DomainIndex:
    index = DomainIndex(domain_id, Path("__offline_vector_ab__"), dimension=vectors.shape[1])
    index.documents = list(documents)
    index._rebuild_entity_index()
    index.bm25.build(index.documents)
    index.manifest = {"document_count": len(documents), "index_version": "offline-vector-ab"}
    attach_vectors(index, vectors)
    return index


class BaselineVectorService:
    def __init__(self, config, index, provider) -> None:
        self.analyzer = QueryAnalyzer([config])
        self.retriever = HybridRetriever(config, index, provider)
        self.reranker = PipelineReranker(cross_encoder_enabled=False)

    def retrieve(self, query: str, *, top_k: int) -> dict[str, Any]:
        analysis = self.analyzer.analyze(query, domain_id=self.retriever.config.domain_id)
        recall_k = max(top_k * 12, 60)
        recalled = self.retriever.search(
            analysis,
            top_k=recall_k,
            recall_k=recall_k,
            mode="hybrid",
        )
        ranked = self.reranker.rerank(analysis, recalled, top_k=top_k)
        return {"results": [candidate.to_dict() for candidate in ranked]}


def _load_eval(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank(ids: list[str], expected: set[str]) -> int | None:
    for rank, document_id in enumerate(ids, 1):
        if document_id in expected:
            return rank
    return None


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    n = len(rows)
    metrics: dict[str, float] = {}
    for k in (1, 3, 5):
        metrics[f"hit@{k}"] = round(
            sum(row[f"{prefix}_rank"] is not None and row[f"{prefix}_rank"] <= k for row in rows) / n,
            4,
        ) if n else 0.0
    metrics["mrr"] = round(
        sum(1 / row[f"{prefix}_rank"] for row in rows if row[f"{prefix}_rank"] is not None) / n,
        4,
    ) if n else 0.0
    return metrics


def _counts(documents: Iterable) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for document in documents:
        counts[document.document_type] += 1
    return dict(counts)


def _fingerprint(documents: Iterable) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.id.encode("utf-8"))
        digest.update(document.embedding_text_fingerprint.encode("ascii"))
    return digest.hexdigest()[:16]


def _write_bundle(root: Path, documents: list, vectors: np.ndarray, provider, *, text_pipeline: str) -> None:
    lengths = [len(document.embedding_text) for document in documents]
    write_vector_artifacts(
        root,
        documents,
        vectors,
        {
            "text_pipeline": text_pipeline,
            "document_count": len(documents),
            "document_type_counts": _counts(documents),
            "embedding_model": provider.model_id,
            "embedding_model_path": provider.model_path,
            "vector_dimension": int(vectors.shape[1]),
            "normalized": True,
            "max_sequence_length": provider.max_length,
            "source_fingerprint": _fingerprint(documents),
            "embedding_text_chars": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(mean(lengths), 2),
            },
        },
    )


def _cached_vectors(root: Path, documents: list, provider) -> np.ndarray | None:
    manifest_path = root / "manifest.json"
    vectors_path = root / "vectors.npy"
    if not manifest_path.exists() or not vectors_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_fingerprint") != _fingerprint(documents):
            return None
        if manifest.get("embedding_model") != provider.model_id:
            return None
        vectors = np.load(vectors_path, allow_pickle=False)
        if vectors.shape != (len(documents), provider.dimension):
            return None
        return np.asarray(vectors, dtype=np.float32)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="多粒度语义文本 V2 完整向量 A/B")
    parser.add_argument("--domain", default="tsukiyashiro_kisaki")
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_file) if args.eval_file else EVAL_DIR / f"{args.domain}.jsonl"
    entries = _load_eval(eval_path)
    base_config = get_default_registry().require(args.domain)
    baseline_docs = base_config.loader(base_config.source_root)
    experiment_domain = f"{args.domain}_multiscale_semantic_v2"
    experiment_config = _config_for_experiment(base_config, experiment_domain)
    build = ReprocessedMultiScaleBuilder(
        domain_id=experiment_domain,
        index_version="multiscale-semantic-v2",
        aliases=base_config.aliases,
        corpus_root=REPO_ROOT,
    ).build(base_config.source_root, ENRICHED_SCENES)
    experiment_docs = list(build.documents)

    expected_map: dict[str, set[str]] = {}
    for entry in entries:
        spec = entry.get("expected") or {}
        explicit = spec.get("expected_document_ids")
        expected_map[entry["id"]] = set(map(str, explicit)) if explicit else resolve_expected_ids(
            baseline_docs, spec.get("criteria")
        )
    invalid = [entry["id"] for entry in entries if entry["category"] != "no_answer" and not expected_map[entry["id"]]]
    if invalid:
        raise ValueError(f"非 no_answer 题目的 gold 解析为空: {invalid}")

    provider = LocalMeanPoolingEmbeddingProvider(batch_size=args.batch_size)
    print(f"模型: {provider.model_id}")
    print(f"模型路径: {provider.model_path}")
    print(f"baseline 文档: {len(baseline_docs)}；V2 文档: {len(experiment_docs)}")

    artifact_root = Path(args.artifact_root)
    started = time.perf_counter()
    baseline_vectors = _cached_vectors(artifact_root / "baseline_original_text", baseline_docs, provider)
    if baseline_vectors is None:
        baseline_vectors = provider.embed_texts([document.embedding_text for document in baseline_docs])
    baseline_embedding_seconds = time.perf_counter() - started
    print(f"baseline 向量完成: {baseline_vectors.shape}，{baseline_embedding_seconds:.1f}s")

    started = time.perf_counter()
    experiment_vectors = _cached_vectors(artifact_root / "multiscale_semantic_v2", experiment_docs, provider)
    if experiment_vectors is None:
        experiment_vectors = provider.embed_texts([document.embedding_text for document in experiment_docs])
    experiment_embedding_seconds = time.perf_counter() - started
    print(f"V2 向量完成: {experiment_vectors.shape}，{experiment_embedding_seconds:.1f}s")

    _write_bundle(artifact_root / "baseline_original_text", baseline_docs, baseline_vectors, provider, text_pipeline="p6_original")
    _write_bundle(
        artifact_root / "multiscale_semantic_v2",
        experiment_docs,
        experiment_vectors,
        provider,
        text_pipeline="story_scene_card_evidence_scale_specific",
    )

    baseline = BaselineVectorService(
        base_config,
        _memory_index(base_config.domain_id, baseline_docs, baseline_vectors),
        provider,
    )
    experiment = ReprocessedMultiScaleService(
        experiment_config,
        _memory_index(experiment_domain, experiment_docs, experiment_vectors),
        provider,
        source_extractor=OriginalTextExtractor(REPO_ROOT),
    )
    gate = QueryAnalyzer([base_config])
    # Exclude one-time model loading from both retrieval latency measurements.
    provider.embed_query("离线评测预热")

    rows: list[dict[str, Any]] = []
    baseline_latencies: list[float] = []
    experiment_latencies: list[float] = []
    raw_requested = 0
    raw_available = 0
    raw_gold_aligned = 0
    for entry in entries:
        query = entry["query"]
        expected = expected_map[entry["id"]]
        is_no_answer = not expected
        routed = bool(gate.analyze(query).matched_domains)

        started = time.perf_counter()
        baseline_bundle = None if is_no_answer and not routed else baseline.retrieve(query, top_k=args.top_k)
        baseline_latencies.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        experiment_bundle = None if is_no_answer and not routed else experiment.retrieve(
            query,
            top_k=args.top_k,
            raw_text=not is_no_answer,
        )
        experiment_latencies.append((time.perf_counter() - started) * 1000)

        baseline_ids = [str(item.get("id")) for item in (baseline_bundle or {}).get("results", [])]
        experiment_ids = [str(item.get("id")) for item in (experiment_bundle or {}).get("results", [])]
        if is_no_answer:
            baseline_rank = 1 if not baseline_ids else None
            experiment_rank = 1 if not experiment_ids else None
        else:
            baseline_rank = _rank(baseline_ids, expected)
            experiment_rank = _rank(experiment_ids, expected)
            raw_requested += 1
            raw = (experiment_bundle or {}).get("raw_excerpt")
            raw_available += int(bool(raw))
            raw_gold_aligned += int(bool(raw) and str(raw.get("parent_id")) in expected)

        rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "query": query,
                "expected_ids": sorted(expected),
                "baseline_rank": baseline_rank,
                "experiment_rank": experiment_rank,
                "baseline_ids": baseline_ids,
                "experiment_ids": experiment_ids,
            }
        )

    baseline_metrics = _metrics(rows, "baseline")
    experiment_metrics = _metrics(rows, "experiment")
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "n": len(subset),
            "baseline": _metrics(subset, "baseline"),
            "experiment": _metrics(subset, "experiment"),
        }
    improved = [row for row in rows if (row["experiment_rank"] or 999) < (row["baseline_rank"] or 999)]
    regressed = [row for row in rows if (row["experiment_rank"] or 999) > (row["baseline_rank"] or 999)]

    report = {
        "scope": "full_vector_fair_ab_multiscale_semantic_v2",
        "eval_file": str(eval_path),
        "entries": len(entries),
        "top_k": args.top_k,
        "fairness_controls": {
            "same_embedding_model": provider.model_id,
            "same_vector_dimension": provider.dimension,
            "same_query_analyzer": True,
            "same_bm25_entity_rrf": True,
            "same_deterministic_reranker": True,
            "same_recall_k": max(args.top_k * 12, 60),
            "only_intervention": "scale-specific text processing, hierarchy, and intent routing",
        },
        "corpora": {
            "baseline": {"documents": len(baseline_docs), "counts": _counts(baseline_docs)},
            "experiment": {
                "documents": len(experiment_docs),
                "counts": build.counts,
                "parented_cards": build.parented_cards,
                "exact_evidence_matches": build.exact_evidence_matches,
            },
        },
        "embedding_seconds": {
            "baseline": round(baseline_embedding_seconds, 2),
            "experiment": round(experiment_embedding_seconds, 2),
        },
        "baseline": {**baseline_metrics, "mean_latency_ms": round(mean(baseline_latencies), 2)},
        "experiment": {
            **experiment_metrics,
            "mean_latency_ms": round(mean(experiment_latencies), 2),
            "raw_available_rate": round(raw_available / raw_requested, 4),
            "raw_gold_aligned_rate": round(raw_gold_aligned / raw_requested, 4),
        },
        "delta": {
            key: round(experiment_metrics[key] - baseline_metrics[key], 4)
            for key in ("hit@1", "hit@3", "hit@5", "mrr")
        },
        "categories": categories,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "improved": improved,
        "regressed": regressed,
        "rows": rows,
        "artifact_root": str(artifact_root),
        "promotion_decision": "manual_review_required",
    }

    print(
        f"baseline: Hit@1={baseline_metrics['hit@1']:.4f} Hit@3={baseline_metrics['hit@3']:.4f} "
        f"Hit@5={baseline_metrics['hit@5']:.4f} MRR={baseline_metrics['mrr']:.4f}"
    )
    print(
        f"V2:       Hit@1={experiment_metrics['hit@1']:.4f} Hit@3={experiment_metrics['hit@3']:.4f} "
        f"Hit@5={experiment_metrics['hit@5']:.4f} MRR={experiment_metrics['mrr']:.4f}"
    )
    print(f"改善={len(improved)}，退化={len(regressed)}，gold 对齐原文率={report['experiment']['raw_gold_aligned_rate']:.4f}")

    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"{args.domain}_multiscale_vector_v2_ab_{timestamp}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
