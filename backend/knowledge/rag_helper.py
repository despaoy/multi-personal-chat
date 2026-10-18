"""Generic knowledge retrieval using original questions and validated task views.

Scope comes only from explicit filters; keyword rules never rewrite the query.
"""

import copy
import json
import logging
import math
import os
import time
from collections import OrderedDict
from threading import RLock
from typing import Any

from infra.environment import read_bool, read_positive_int

from .reranker import get_reranker
from .task_retrieval import literal_task_views
from .vector_db import get_vector_db

logger = logging.getLogger(__name__)


def _stable_result_key(result: dict[str, Any]) -> str:
    """Use the unique source ID already validated by the vector index."""
    return f"id:{result['id']}"


class RAGHelper:
    """RAG辅助类，提供完整的检索增强生成流程。

    核心流程为两阶段检索：第一阶段用混合检索（向量+BM25）做粗排召回，
    第二阶段用Cross-Encoder重排器做精排。支持查询缓存、多查询融合、
    独立任务证据保留。
    """

    def __init__(self):
        """初始化RAG辅助类，按显式配置加载重排器，检索时获取向量数据库。"""
        self.min_score_threshold = 0.05
        self.top_k = 5

        self.reranker = None
        self.enable_reranking = read_bool(os.environ, "RERANKER_ENABLED")
        self.recall_multiplier = 4

        self._query_cache: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._cache_lock = RLock()
        self._cache_max_size = 100
        self._cache_ttl = read_positive_int(os.environ, "RAG_RETRIEVAL_CACHE_TTL", 60)

        if self.enable_reranking:
            self.reranker = get_reranker()

    def _get_from_cache(self, query: str) -> list[dict[str, Any]] | None:
        """Copy an owned snapshot outside the LRU critical section."""
        with self._cache_lock:
            cached = self._query_cache.get(query)
            if cached is None:
                return None
            expires_at, results = cached
            if expires_at <= time.monotonic():
                self._query_cache.pop(query, None)
                return None
            self._query_cache.move_to_end(query)
        return copy.deepcopy(results)

    def _add_to_cache(self, query: str, results: list[dict[str, Any]]) -> None:
        snapshot = copy.deepcopy(results)
        with self._cache_lock:
            self._query_cache[query] = (time.monotonic() + self._cache_ttl, snapshot)
            self._query_cache.move_to_end(query)
            while len(self._query_cache) > self._cache_max_size:
                self._query_cache.popitem(last=False)

    def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        enable_rerank: bool = True,
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
        additional_queries: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """检索与查询最相关的知识库文档。

        流程：原问题或已验证任务视图 -> 混合检索 -> Cross-Encoder重排 -> 分数归一化。
        支持查询缓存，重复查询直接返回缓存结果。

        Args:
            query: 用户查询文本
            top_k: 返回结果数量，默认使用self.top_k（5）
            enable_rerank: 是否启用重排器精排
            filters: 元数据过滤条件
            use_cache: 是否使用查询缓存

        Returns:
            按相关性降序排列的文档列表，每项包含normalized_score等字段
        """
        top_k = self.top_k if top_k is None else top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("RAG candidate limit must be a positive integer")
        views = literal_task_views(query, additional_queries)
        if views:
            return self._task_candidate_plan(
                query, views, top_k=top_k, enable_rerank=enable_rerank,
                filters=filters, use_cache=use_cache,
            ).results

        vector_db = get_vector_db()
        vector_generation = vector_db.cache_generation
        cache_key = json.dumps(
            [query, top_k, enable_rerank and self.enable_reranking, self.min_score_threshold,
             self.recall_multiplier, filters or {}, vector_generation],
            ensure_ascii=False, sort_keys=True,
        )
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.info("RAG缓存命中")
                return cached

        recall_results = [dict(row) for row in vector_db.hybrid_search(
            query, top_k=top_k * self.recall_multiplier,
            threshold=self.min_score_threshold, keyword_weight=0.3, filters=filters or None,
        )]
        if not recall_results:
            logger.info("RAG检索未找到相关文档")
            return []

        recall_results.sort(key=lambda row: (-row["fused_score"], _stable_result_key(row)))
        logger.info("第一阶段混合检索完成: %s 个候选文档", len(recall_results))

        if enable_rerank and self.enable_reranking:
            results = self.reranker.rerank(query, recall_results, top_k=top_k)
            if not results:
                raise RuntimeError("RAG reranker returned no results for non-empty candidates")
            score_key = "rerank_normalized_score"
        else:
            results = recall_results[:top_k]
            score_key = "fused_score"
        for result in results:
            result["normalized_score"] = result[score_key]
        logger.info("检索完成")
        if use_cache:
            self._add_to_cache(cache_key, results)
        return results

    def _task_candidate_plan(self, query, views, *, top_k, enable_rerank, filters, use_cache, threshold=0.3, question_binding=None):
        from knowledge.task_retrieval import collect_task_candidates

        resolved_filters = copy.deepcopy(filters)

        def snapshot():
            index = get_vector_db()
            return id(index), index.cache_generation

        def retrieve(question, *, top_k, filters):
            # Every validated task keeps the caller's explicit scope.
            return self.retrieve_context(
                question, top_k=top_k, enable_rerank=enable_rerank,
                filters=filters, use_cache=use_cache,
            )

        return collect_task_candidates(
            query, views, top_k=self.top_k if top_k is None else top_k, filters=resolved_filters,
            retrieve=retrieve, confidence=self.compute_confidence,
            stable_key=_stable_result_key, snapshot=snapshot, threshold=threshold, question_binding=question_binding,
        )

    @staticmethod
    def _absolute_score(result: dict[str, Any]) -> float:
        """Return a cross-query comparable score instead of per-result-list normalization."""
        value = result.get("score", result.get("fused_score"))
        message = "RAG retrieval score must be a finite number"
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if isinstance(value, bool) or not math.isfinite(score):
            raise ValueError(message)
        return max(0.0, min(1.0, score))

    def compute_confidence(self, results: list[dict[str, Any]]) -> float:
        """Estimate confidence from absolute retrieval scores, not min-max rank scores."""
        if not results:
            return 0.0
        scores = [self._absolute_score(result) for result in results[:3]]
        top_score = scores[0]
        support_score = sum(scores) / len(scores)
        return round(0.7 * top_score + 0.3 * support_score, 4)

    def build_citations(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从检索结果构建引用列表，供前端展示证据来源。"""
        citations: list[dict[str, Any]] = []
        for r in results:
            content = r["content"]
            citations.append(
                {
                    "source_id": str(r["id"]),
                    "source_title": r["title"],
                    "evidence_excerpt": content[:200] + ("..." if len(content) > 200 else ""),
                    "score": round(r["normalized_score"] if "normalized_score" in r else self._absolute_score(r), 4),
                    "kb_revision": r.get("kb_revision", ""),
                    "source_path": r.get("source_path", ""),
                    "source_line": r.get("source_line"),
                    "source_event_ids": list(r.get("source_event_ids", [])),
                    "source_lineage": list(r.get("source_lineage", [])),
                    "section": r.get("section", r.get("category", "")),
                    "version": r.get("version", "1.0"),
                }
            )
        return citations

    def should_abstain(self, confidence: float, threshold: float = 0.3) -> bool:
        """判断是否应弃答（置信度低于阈值时返回 True）。"""
        return confidence < threshold

    def retrieve_with_citations(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float = 0.3,
        filters: dict[str, Any] | None = None,
        additional_queries: tuple[str, ...] = (),
        question_binding: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """检索并返回带引用和置信度的结构化结果。

        Returns:
            {results, citations, confidence, abstained}
        """
        if question_binding is not None:
            from knowledge.public_question_binding import bound_task_search_views

            if additional_queries:
                raise ValueError("Bound task retrieval cannot accept separate query hints")
            views = bound_task_search_views(question_binding, query)
        else:
            views = literal_task_views(query, additional_queries)
        task_coverage = ()
        if views:
            plan = self._task_candidate_plan(
                query, views, top_k=top_k, enable_rerank=True,
                filters=filters, use_cache=True, threshold=threshold,
                **({"question_binding": question_binding} if question_binding is not None else {}),
            )
            results, task_coverage = plan.results, plan.coverage
        else:
            results = self.retrieve_context(query, top_k=top_k, filters=filters)
        confidence = self.compute_confidence(results)
        abstained = self.should_abstain(confidence, threshold)
        citations = self.build_citations(results) if not abstained else []
        return {
            "results": results,
            "citations": citations,
            "confidence": confidence,
            "abstained": abstained,
            **({"task_candidate_coverage": task_coverage} if views else {}),
        }

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._query_cache.clear()
        logger.info("RAG查询缓存已清除")


_rag_helper: RAGHelper | None = None
_rag_helper_init_lock = RLock()


def get_rag_helper() -> RAGHelper:
    global _rag_helper
    with _rag_helper_init_lock:
        if _rag_helper is None:
            _rag_helper = RAGHelper()
        return _rag_helper
