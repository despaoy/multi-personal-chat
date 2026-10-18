"""R2 retrieval ablation with per-query provenance and no formal mock results."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(float(value) for value in values)
    rank = (len(values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    interpolated = values[lower] + (values[upper] - values[lower]) * (rank - lower)
    return round(interpolated, 3)


@dataclass
class RAGAblationResult:
    variant_name: str
    mock: bool = False
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    answerable_count: int = 0
    unanswerable_count: int = 0
    failed_queries: int = 0
    per_question: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""


class RAGAblation:
    DEFAULT_VARIANTS = ["vector_only", "bm25_only", "hybrid", "hybrid_reranker"]

    def __init__(self, dataset_path: str = "", k: int = 5, *, formal: bool = False):
        default = _BACKEND_DIR / "data" / "character_dialogues" / "experiments" / "research" / "kisaki_rag_eval_v2_candidates.json"
        self.dataset_path = Path(dataset_path) if dataset_path else default
        self.k = k
        self.formal = formal
        self._vector_db = None
        self._rag_helper = None

    def _dataset(self) -> Dict[str, Any]:
        dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if self.formal and (dataset.get("status") != "frozen" or not dataset.get("formal_use_allowed")):
            raise ValueError("formal R2 evaluation requires a reviewed and frozen dataset")
        return dataset

    def _get_vector_db(self):
        if self._vector_db is None:
            from knowledge.vector_db import get_vector_db
            self._vector_db = get_vector_db()
        return self._vector_db

    def _get_rag_helper(self):
        if self._rag_helper is None:
            from knowledge.rag_helper import get_rag_helper
            self._rag_helper = get_rag_helper()
        return self._rag_helper

    def variant_vector_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self._get_vector_db().search(query, top_k=top_k, threshold=0.0)

    def variant_bm25_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        database = self._get_vector_db()
        return [
            {**database.metadata[index], "score": score}
            for index, score in database.bm25.search(query, top_k=top_k, threshold=0.0)
            if 0 <= index < len(database.metadata)
        ]

    def variant_hybrid(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self._get_vector_db().hybrid_search(query, top_k=top_k, threshold=0.0, keyword_weight=0.3)

    def variant_hybrid_reranker(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        helper = self._get_rag_helper()
        if self.formal and (not helper.enable_reranking or helper.reranker is None):
            raise RuntimeError("formal hybrid+reranker requires an available Cross-Encoder")
        rows = helper.retrieve_context(
            query, top_k=top_k, enable_rerank=True, use_cache=False
        )
        if self.formal and len(rows) > 1 and not any("rerank_score" in row for row in rows):
            raise RuntimeError("Cross-Encoder did not produce rerank scores")
        return rows

    def _variant(self, name: str) -> Callable[[str, int], List[Dict[str, Any]]]:
        variants = {
            "vector_only": self.variant_vector_only,
            "bm25_only": self.variant_bm25_only,
            "hybrid": self.variant_hybrid,
            "hybrid_reranker": self.variant_hybrid_reranker,
        }
        if name not in variants:
            raise ValueError(f"unknown R2 variant: {name}")
        return variants[name]

    def run_variant(self, variant_name: str) -> RAGAblationResult:
        from evaluation.retrieval_metrics import RetrievalMetrics

        dataset = self._dataset()
        retrieve = self._variant(variant_name)
        metrics = RetrievalMetrics()
        result = RAGAblationResult(variant_name=variant_name)
        recalls_1: List[float] = []
        recalls_5: List[float] = []
        mrr_values: List[float] = []
        ndcg_values: List[float] = []
        latencies: List[float] = []
        for question in dataset["questions"]:
            expected = list(question.get("expected_doc_ids", []))
            started = time.perf_counter()
            retrieved = retrieve(str(question.get("question", "")), self.k)
            latency = (time.perf_counter() - started) * 1000
            latencies.append(latency)
            retrieved_ids = [str(row.get("id", row.get("chunk_id", ""))) for row in retrieved]
            if expected:
                result.answerable_count += 1
                r1 = metrics.recall_at_k(retrieved_ids, expected, 1)
                r5 = metrics.recall_at_k(retrieved_ids, expected, self.k)
                mrr = metrics.mrr(retrieved_ids, expected)
                ndcg = metrics.ndcg(retrieved_ids, expected, self.k)
                recalls_1.append(r1); recalls_5.append(r5); mrr_values.append(mrr); ndcg_values.append(ndcg)
            else:
                result.unanswerable_count += 1
                r1 = r5 = mrr = ndcg = None
            result.per_question.append({
                "id": question.get("id"),
                "question_type": question.get("question_type"),
                "expected_doc_ids": expected,
                "retrieved_doc_ids": retrieved_ids,
                "recall_at_1": r1,
                "recall_at_5": r5,
                "mrr": mrr,
                "ndcg_at_5": ndcg,
                "latency_ms": round(latency, 3),
                "error": "",
            })
        result.recall_at_1 = round(statistics.mean(recalls_1), 4) if recalls_1 else 0.0
        result.recall_at_5 = round(statistics.mean(recalls_5), 4) if recalls_5 else 0.0
        result.mrr = round(statistics.mean(mrr_values), 4) if mrr_values else 0.0
        result.ndcg_at_5 = round(statistics.mean(ndcg_values), 4) if ndcg_values else 0.0
        result.avg_latency_ms = round(statistics.mean(latencies), 3) if latencies else 0.0
        result.p50_latency_ms = _percentile(latencies, 0.50)
        result.p95_latency_ms = _percentile(latencies, 0.95)
        return result

    def run_all(self, variants: Optional[List[str]] = None) -> List[RAGAblationResult]:
        return [self.run_variant(name) for name in (variants or self.DEFAULT_VARIANTS)]

    def run_all_mock(self) -> List[RAGAblationResult]:
        return [RAGAblationResult(variant_name=name, mock=True) for name in self.DEFAULT_VARIANTS]

    def build_comparison_table(self, results: List[RAGAblationResult]) -> List[Dict[str, Any]]:
        return [{
            "variant": row.variant_name,
            "mock": row.mock,
            "recall_at_1": row.recall_at_1,
            "recall_at_5": row.recall_at_5,
            "mrr": row.mrr,
            "ndcg_at_5": row.ndcg_at_5,
            "p50_latency_ms": row.p50_latency_ms,
            "p95_latency_ms": row.p95_latency_ms,
            "failed_queries": row.failed_queries,
        } for row in results]

    def save_report(self, results: List[RAGAblationResult], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._dataset()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        payload = {
            "schema_version": 2,
            "experiment_type": "r2_retrieval_ablation",
            "mock": any(row.mock for row in results),
            "formal": self.formal and not any(row.mock for row in results),
            "dataset": {"path": str(self.dataset_path), "sha256": _sha256(self.dataset_path), "status": dataset.get("status")},
            "k": self.k,
            "results": [asdict(row) for row in results],
            "comparison_table": self.build_comparison_table(results),
        }
        path = output_dir / f"r2_retrieval_ablation_{timestamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/results"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    runner = RAGAblation(args.dataset, args.k, formal=args.formal)
    results = runner.run_all_mock() if args.mock else runner.run_all()
    print(runner.save_report(results, args.output_dir))
    return 0 if all(not row.error for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
