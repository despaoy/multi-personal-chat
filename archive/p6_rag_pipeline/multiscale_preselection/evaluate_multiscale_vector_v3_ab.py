"""Evaluate physically routed multi-scale V3 against an equally routed P6 baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluate_rag_retrieval import resolve_expected_ids  # noqa: E402
from knowledge.multiscale_rag import LocalMeanPoolingEmbeddingProvider, OriginalTextExtractor, RoutedMultiScaleService  # noqa: E402
from knowledge.multiscale_rag.service_v3 import CARD_TYPES  # noqa: E402
from knowledge.multiscale_rag.vector_runtime import attach_vectors, write_vector_artifacts  # noqa: E402
from knowledge.rag_pipeline.documents import KnowledgeIndexDocument  # noqa: E402
from knowledge.rag_pipeline.index import DomainIndex  # noqa: E402
from knowledge.rag_pipeline.pipeline import DomainRuntime, RagPipeline  # noqa: E402
from knowledge.rag_pipeline.query import QueryAnalyzer  # noqa: E402
from knowledge.rag_pipeline.registry import KnowledgeDomainConfig, get_default_registry  # noqa: E402

EVAL_DIR = BACKEND_DIR / "data" / "eval" / "rag_retrieval"
REPORT_DIR = EVAL_DIR / "reports"
KNOWLEDGE_ROOT = BACKEND_DIR / "data" / "knowledge" / "tsukiyashiro_kisaki"
V2_ARTIFACT_ROOT = KNOWLEDGE_ROOT / "rag_vector_ab_multiscale_v2"
DEFAULT_ARTIFACT_ROOT = KNOWLEDGE_ROOT / "rag_vector_ab_multiscale_v3_routed"


def _load_bundle(root: Path) -> tuple[list[KnowledgeIndexDocument], np.ndarray, dict[str, Any]]:
    documents = [
        KnowledgeIndexDocument.from_dict(json.loads(line))
        for line in (root / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(root / "vectors.npy", allow_pickle=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if vectors.shape != (len(documents), 384):
        raise ValueError(f"bundle 数量或维度不一致: {root} {vectors.shape} {len(documents)}")
    return documents, np.asarray(vectors, dtype=np.float32), manifest


def _config(base: KnowledgeDomainConfig, domain_id: str, document_types: list[str]) -> KnowledgeDomainConfig:
    return KnowledgeDomainConfig(
        domain_id=domain_id,
        source_root=base.source_root,
        loader=lambda _root: [],
        document_types=document_types,
        aliases=dict(base.aliases),
        story_titles=list(base.story_titles),
        narrative_policy=base.narrative_policy,
        retrieval_defaults=base.retrieval_defaults,
        prompt_supplement=base.prompt_supplement,
        index_version="multiscale-routed-v3",
        enabled=True,
    )


def _index(domain_id: str, documents: list[KnowledgeIndexDocument], vectors: np.ndarray) -> DomainIndex:
    index = DomainIndex(domain_id, Path("__multiscale_v3_memory__"), vectors.shape[1])
    index.documents = documents
    index._rebuild_entity_index()
    index.bm25.build(documents)
    index.manifest = {"document_count": len(documents), "index_version": "multiscale-routed-v3"}
    attach_vectors(index, vectors)
    return index


def _subset(
    documents: list[KnowledgeIndexDocument],
    vectors: np.ndarray,
    allowed: frozenset[str],
) -> tuple[list[KnowledgeIndexDocument], np.ndarray]:
    rows = [row for row, document in enumerate(documents) if document.document_type in allowed]
    return [documents[row] for row in rows], vectors[rows]


def _route_indexes(domain_id: str, documents: list[KnowledgeIndexDocument], vectors: np.ndarray) -> dict[frozenset[str], DomainIndex]:
    routes = {
        CARD_TYPES,
        frozenset({"fact"}),
        frozenset({"relation"}),
        frozenset({"event"}),
        frozenset({"fact", "relation"}),
        frozenset({"fact", "event"}),
        frozenset({"relation", "event"}),
        frozenset({"story", "scene"}),
        frozenset({"evidence"}),
    }
    indexes: dict[frozenset[str], DomainIndex] = {}
    for route in routes:
        subset_docs, subset_vectors = _subset(documents, vectors, route)
        if subset_docs:
            indexes[route] = _index(domain_id, subset_docs, subset_vectors)
    return indexes


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    n = len(rows)
    result = {
        f"hit@{k}": round(
            sum(row[f"{prefix}_rank"] is not None and row[f"{prefix}_rank"] <= k for row in rows) / n,
            4,
        ) if n else 0.0
        for k in (1, 3, 5)
    }
    result["mrr"] = round(
        sum(1 / row[f"{prefix}_rank"] for row in rows if row[f"{prefix}_rank"] is not None) / n,
        4,
    ) if n else 0.0
    return result


def _rank(ids: list[str], expected: set[str]) -> int | None:
    return next((rank for rank, document_id in enumerate(ids, 1) if document_id in expected), None)


def _write_split_artifacts(
    root: Path,
    documents: list[KnowledgeIndexDocument],
    vectors: np.ndarray,
    model_id: str,
) -> None:
    for name, types in (
        ("card_index", CARD_TYPES),
        ("scene_story_index", frozenset({"scene", "story"})),
        ("evidence_index", frozenset({"evidence"})),
    ):
        docs, matrix = _subset(documents, vectors, types)
        write_vector_artifacts(
            root / name,
            docs,
            matrix,
            {
                "version": "multiscale-routed-v3",
                "route_types": sorted(types),
                "document_count": len(docs),
                "vector_dimension": int(matrix.shape[1]),
                "embedding_model": model_id,
                "normalized": True,
                "physical_scale_partition": True,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="多粒度 V3 物理分区向量 A/B")
    parser.add_argument("--domain", default="tsukiyashiro_kisaki")
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--comparison",
        choices=("controlled", "best"),
        default="controlled",
        help="controlled=相同V3路由器；best=P6原生完整链路对多粒度V3最佳链路",
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_file) if args.eval_file else EVAL_DIR / f"{args.domain}.jsonl"
    entries = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline_docs, baseline_vectors, baseline_manifest = _load_bundle(V2_ARTIFACT_ROOT / "baseline_original_text")
    experiment_docs, experiment_vectors, experiment_manifest = _load_bundle(V2_ARTIFACT_ROOT / "multiscale_semantic_v2")
    if baseline_manifest["embedding_model"] != experiment_manifest["embedding_model"]:
        raise ValueError("baseline 与 experiment embedding 模型不一致")

    base = get_default_registry().require(args.domain)
    experiment_domain = experiment_docs[0].domain_id
    baseline_config = _config(base, base.domain_id, ["fact", "relation", "event"])
    experiment_config = _config(
        base,
        experiment_domain,
        ["story", "scene", "fact", "relation", "event", "evidence"],
    )
    provider = LocalMeanPoolingEmbeddingProvider()
    baseline_index = _index(base.domain_id, baseline_docs, baseline_vectors)
    if args.comparison == "best":
        baseline_service = RagPipeline(registry=get_default_registry(), embedding_provider=provider)
        baseline_service._runtimes = {
            base.domain_id: DomainRuntime(base, baseline_index, provider),
        }
        baseline_service._load_attempted = True
        baseline_service._available = True
    else:
        baseline_service = RoutedMultiScaleService(
            baseline_config,
            _route_indexes(base.domain_id, baseline_docs, baseline_vectors),
            provider,
            all_documents=baseline_docs,
        )
    experiment_service = RoutedMultiScaleService(
        experiment_config,
        _route_indexes(experiment_domain, experiment_docs, experiment_vectors),
        provider,
        all_documents=experiment_docs,
        source_extractor=OriginalTextExtractor(REPO_ROOT),
    )
    _write_split_artifacts(Path(args.artifact_root) / "multiscale", experiment_docs, experiment_vectors, provider.model_id)
    _write_split_artifacts(Path(args.artifact_root) / "baseline", baseline_docs, baseline_vectors, provider.model_id)

    expected_map: dict[str, set[str]] = {}
    for entry in entries:
        spec = entry.get("expected") or {}
        explicit = spec.get("expected_document_ids")
        expected_map[entry["id"]] = set(map(str, explicit)) if explicit else resolve_expected_ids(
            baseline_docs, spec.get("criteria")
        )
    gate = QueryAnalyzer([base])
    provider.embed_query("离线评测预热")

    rows: list[dict[str, Any]] = []
    baseline_ms: list[float] = []
    experiment_ms: list[float] = []
    route_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        query = entry["query"]
        expected = expected_map[entry["id"]]
        no_answer = not expected
        routed = bool(gate.analyze(query).matched_domains)

        started = time.perf_counter()
        if no_answer and not routed:
            baseline_bundle = None
        elif args.comparison == "best":
            baseline_bundle = baseline_service.retrieve(
                query,
                domain_id=None if no_answer else base.domain_id,
                top_k=args.top_k,
                mode="hybrid",
                use_rerank=True,
            )
        else:
            baseline_bundle = baseline_service.retrieve(query, top_k=args.top_k)
        baseline_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        experiment_bundle = None if no_answer and not routed else experiment_service.retrieve(
            query, top_k=args.top_k, raw_text=not no_answer
        )
        experiment_ms.append((time.perf_counter() - started) * 1000)

        baseline_ids = [str(item["id"]) for item in (baseline_bundle or {}).get("results", [])]
        experiment_ids = [str(item["id"]) for item in (experiment_bundle or {}).get("results", [])]
        route = "+".join((experiment_bundle or {}).get("route_types", [])) or "no_answer"
        route_counts[route] += 1
        if no_answer:
            baseline_rank = 1 if not baseline_ids else None
            experiment_rank = 1 if not experiment_ids else None
        else:
            baseline_rank = _rank(baseline_ids, expected)
            experiment_rank = _rank(experiment_ids, expected)
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
                "route": route,
                "timeline_count": len((experiment_bundle or {}).get("relation_timeline", [])),
            }
        )

    baseline_metrics = _metrics(rows, "baseline")
    experiment_metrics = _metrics(rows, "experiment")
    categories = {}
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
        "scope": (
            "best_native_p6_vs_multiscale_routed_v3"
            if args.comparison == "best"
            else "full_vector_fair_ab_multiscale_routed_v3"
        ),
        "entries": len(entries),
        "fairness_controls": {
            "same_embedding_model": provider.model_id,
            "same_query_analysis_fix": "explicit-domain single-character entity recovery",
            "same_quoted_title_rerank": True,
            "same_bm25_entity_rrf": True,
            "same_recall_k": max(args.top_k * 12, 60),
        },
        "v3_interventions": [
            "physical scale indexes before recall",
            "schema partition before recall",
            "relationship-focused routing",
            "relation timeline context",
        ],
        "comparison_policy": (
            "same_questions_and_labels_each_system_uses_native_best_pipeline"
            if args.comparison == "best"
            else "controlled_shared_routing"
        ),
        "baseline": {**baseline_metrics, "mean_latency_ms": round(mean(baseline_ms), 2)},
        "experiment": {**experiment_metrics, "mean_latency_ms": round(mean(experiment_ms), 2)},
        "delta": {key: round(experiment_metrics[key] - baseline_metrics[key], 4) for key in baseline_metrics},
        "route_counts": dict(route_counts),
        "categories": categories,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "improved": improved,
        "regressed": regressed,
        "rows": rows,
        "artifact_root": str(Path(args.artifact_root)),
        "promotion_decision": "manual_review_required",
    }

    print(
        f"baseline: Hit@1={baseline_metrics['hit@1']:.4f} Hit@3={baseline_metrics['hit@3']:.4f} "
        f"Hit@5={baseline_metrics['hit@5']:.4f} MRR={baseline_metrics['mrr']:.4f}"
    )
    print(
        f"V3:       Hit@1={experiment_metrics['hit@1']:.4f} Hit@3={experiment_metrics['hit@3']:.4f} "
        f"Hit@5={experiment_metrics['hit@5']:.4f} MRR={experiment_metrics['mrr']:.4f}"
    )
    print(f"改善={len(improved)} 退化={len(regressed)} 路由={dict(route_counts)}")
    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"{args.domain}_multiscale_vector_v3_ab_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
