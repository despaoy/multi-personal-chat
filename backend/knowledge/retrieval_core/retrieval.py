"""Shared hybrid retrieval: sparse + vector + entity recall + RRF fusion.

三路召回通道：
1. sparse：BM25（归一查询 + 原始查询，取最优名次）
2. vector：FAISS 内积近邻（原始查询，语义自然形式）
3. entity：实体精确索引（保证含查询实体的文档进入候选）

融合：RRF（可解释）+ 精确实体命中加权（通用规则）+ 叙事层权重
（domain 配置的策略表 + 查询意图偏好）+ 硬性 metadata 过滤。

不使用针对具体人物/卡片/问句的 if/else 调权。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .documents import KnowledgeIndexDocument
    from .embedding import EmbeddingProvider
    from .index import DomainIndex
    from .query import QueryAnalysis

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """单条候选（保留全部分数与来源，供重排与引用使用）。"""

    row: int
    document: KnowledgeIndexDocument
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int | None = None
    sparse_rank: int | None = None
    entity_rank: int | None = None
    rrf_score: float = 0.0
    entity_boost: float = 0.0
    layer_multiplier: float = 1.0
    type_multiplier: float = 1.0
    fused_score: float = 0.0
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        doc = self.document
        return {
            "id": doc.id,
            "domain_id": doc.domain_id,
            "document_type": doc.document_type,
            "title": doc.title,
            "summary": doc.summary,
            "content": doc.content,
            "entities": list(doc.entities),
            "relations": list(doc.relations),
            "reality_status": doc.reality_status,
            "temporal_scope": doc.temporal_scope,
            "content_scope": doc.content_scope,
            "source": doc.source.to_dict(),
            "metadata": doc.metadata,
            "index_version": doc.index_version,
            "row": self.row,
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "rrf_score": round(self.rrf_score, 6),
            "entity_boost": round(self.entity_boost, 6),
            "layer_multiplier": round(self.layer_multiplier, 4),
            "type_multiplier": round(self.type_multiplier, 4),
            "fused_score": round(self.fused_score, 6),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "retrieval_rank": 0,
        }


def match_metadata_filters(
    doc: KnowledgeIndexDocument,
    filters: dict[str, Any] | None,
) -> bool:
    """通用 metadata 硬过滤。

    支持：document_type / reality_status / temporal_scope / content_scope
    （等值或列表任一匹配）、entities（任一交集）、story_unit_id /
    story_title（metadata 等值）。域自定义字段经 metadata 等值匹配。
    """
    if not filters:
        return True
    for key, value in filters.items():
        if key in ("document_type", "reality_status", "temporal_scope", "content_scope"):
            allowed = value if isinstance(value, (list, tuple, set)) else [value]
            actual = getattr(doc, key)
            if actual not in allowed:
                return False
        elif key == "entities":
            wanted = set(value if isinstance(value, (list, tuple, set)) else [value])
            if not wanted & set(doc.entities):
                return False
        elif key == "domain_id":
            if doc.domain_id != value:
                return False
        else:
            if doc.metadata.get(key) != value:
                return False
    return True


class HybridRetriever:
    """单域混合检索器。"""

    def __init__(
        self,
        domain_config,
        domain_index: DomainIndex,
        embedding_provider: EmbeddingProvider,
    ):
        self.config = domain_config
        self.index = domain_index
        self.embedding_provider = embedding_provider
        defaults = domain_config.retrieval_defaults
        self.rrf_k = defaults.rrf_k
        self.entity_boost_base = defaults.entity_boost
        self.entity_boost_cap = defaults.entity_boost_cap
        self.layer_lo, self.layer_hi = defaults.layer_boost_range
        policy = domain_config.narrative_policy
        self.reality_boost = dict(policy.reality_boost)
        self.temporal_boost = dict(policy.temporal_boost)
        self.scope_boost = dict(policy.scope_boost)

    # -- 叙事层与类型权重 ---------------------------------------------------
    def _layer_multiplier(self, doc: KnowledgeIndexDocument, analysis: QueryAnalysis) -> float:
        lo, hi = self.layer_lo, self.layer_hi
        multiplier = 1.0

        if analysis.reality_preferences:
            multiplier *= hi if doc.reality_status in analysis.reality_preferences else lo
        else:
            multiplier *= self.reality_boost.get(doc.reality_status, 1.0)

        if analysis.temporal_preferences:
            multiplier *= hi if doc.temporal_scope in analysis.temporal_preferences else lo
        else:
            multiplier *= self.temporal_boost.get(doc.temporal_scope, 1.0)

        if analysis.scope_preferences:
            multiplier *= hi if doc.content_scope in analysis.scope_preferences else lo
        else:
            multiplier *= self.scope_boost.get(doc.content_scope, 1.0)

        return max(lo, min(hi, multiplier))

    def _type_multiplier(self, doc: KnowledgeIndexDocument, analysis: QueryAnalysis) -> float:
        if not analysis.doc_type_preferences:
            return 1.0
        return 1.15 if doc.document_type in analysis.doc_type_preferences else 0.95

    def _entity_boost(self, doc: KnowledgeIndexDocument, analysis: QueryAnalysis) -> float:
        if not analysis.entities:
            return 0.0
        hits = len(set(analysis.entities) & set(doc.entities))
        if hits == 0:
            return 0.0
        return min(self.entity_boost_cap, self.entity_boost_base * hits)

    # -- 通道检索 ------------------------------------------------------------
    def _sparse_channel(
        self,
        analysis: QueryAnalysis,
        recall_k: int,
        filters: dict[str, Any] | None,
    ) -> dict[int, tuple]:
        """BM25 通道：归一查询与原始查询各查一次，取每文档最优名次。"""
        best: dict[int, tuple] = {}  # row -> (best_rank, best_score)
        query_texts = [analysis.normalized_query, analysis.original_query]
        if analysis.lexical_expansions:
            query_texts.append(f"{analysis.normalized_query} {' '.join(analysis.lexical_expansions)}")
        for query_text in dict.fromkeys(query_texts):
            for rank0, (row, score) in enumerate(self.index.search_sparse(query_text, recall_k)):
                doc = self.index.get_document(row)
                if doc is None or not match_metadata_filters(doc, filters):
                    continue
                rank = rank0 + 1
                prev = best.get(row)
                if prev is None or rank < prev[0]:
                    best[row] = (rank, score)
        return best

    def _vector_channel(
        self,
        analysis: QueryAnalysis,
        recall_k: int,
        filters: dict[str, Any] | None,
        threshold: float = 0.0,
    ) -> dict[int, tuple]:
        best: dict[int, tuple] = {}
        try:
            query_vector = self.embedding_provider.embed_query(analysis.original_query)
        except Exception as e:  # noqa: BLE001 - 向量通道失败降级为稀疏召回
            logger.warning("向量通道失败（降级 sparse-only）: %s", e)
            return best
        for rank0, (row, score) in enumerate(self.index.search_vector(query_vector, recall_k, threshold=threshold)):
            doc = self.index.get_document(row)
            if doc is None or not match_metadata_filters(doc, filters):
                continue
            best[row] = (rank0 + 1, score)
        return best

    def _entity_channel(
        self,
        analysis: QueryAnalysis,
        recall_k: int,
        filters: dict[str, Any] | None,
    ) -> dict[int, tuple]:
        """实体精确通道。

        多实体查询优先返回包含全部查询实体的文档（AND 语义，
        关系方向问题的精确候选）；无 AND 命中时回退任一命中（ANY）。
        通道内按 BM25 分数取名次。
        """
        if not analysis.entities:
            return {}

        def _filtered_rows(rows: list[int]) -> list[int]:
            result = []
            for row in rows:
                doc = self.index.get_document(row)
                if doc is None or not match_metadata_filters(doc, filters):
                    continue
                result.append(row)
            return result

        rows = _filtered_rows(self.index.entity_lookup(analysis.entities))
        if len(analysis.entities) > 1:
            # AND：文档实体覆盖全部查询实体
            wanted = set(analysis.entities)
            and_rows = [row for row in rows if wanted <= set(self.index.get_document(row).entities)]
            if and_rows:
                rows = and_rows
        if not rows:
            return {}

        # 实体行内用 BM25 精排：对候选行打一次稀疏分数
        text = analysis.normalized_query or analysis.original_query
        row_set = set(rows)
        sparse_scores: dict[int, float] = {}
        for row, score in self.index.search_sparse(text, max(len(row_set) * 2, recall_k)):
            if row in row_set:
                sparse_scores[row] = score
        ordered = sorted(row_set, key=lambda r: (-sparse_scores.get(r, 0.0), r))[:recall_k]
        return {row: (rank + 1, sparse_scores.get(row, 0.0)) for rank, row in enumerate(ordered)}

    # -- 主检索 -------------------------------------------------------------
    def search(
        self,
        analysis: QueryAnalysis,
        top_k: int,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
        recall_k: int | None = None,
        vector_threshold: float = 0.0,
    ) -> list[RetrievalCandidate]:
        """执行检索。mode: sparse / vector / hybrid。

        - sparse：仅 BM25 通道排序（分数归一化到 [0,1]）
        - vector：仅向量通道排序（内积分数）
        - hybrid：三通道 RRF 融合 + 实体加权 + 叙事层/类型权重
        """
        if self.index.count() == 0:
            return []
        defaults = self.config.retrieval_defaults
        recall_k = recall_k or defaults.recall_k
        filters = dict(filters or {})
        filters.pop("domain_id", None)  # 域内检索隐含 domain

        if mode == "sparse":
            return self._ranked_from_channel(self._sparse_channel(analysis, recall_k, filters), "bm25_score", top_k)
        if mode == "vector":
            return self._ranked_from_channel(
                self._vector_channel(analysis, recall_k, filters, threshold=vector_threshold),
                "vector_score",
                top_k,
            )

        sparse = self._sparse_channel(analysis, recall_k, filters)
        vector = self._vector_channel(analysis, recall_k, filters, threshold=vector_threshold)
        entity = self._entity_channel(analysis, recall_k, filters)

        candidates: dict[int, RetrievalCandidate] = {}
        for row, (rank, score) in sparse.items():
            doc = self.index.get_document(row)
            candidates[row] = RetrievalCandidate(row=row, document=doc, sparse_rank=rank, bm25_score=score)
        for row, (rank, score) in vector.items():
            doc = self.index.get_document(row)
            if row in candidates:
                candidates[row].vector_rank = rank
                candidates[row].vector_score = score
            else:
                candidates[row] = RetrievalCandidate(row=row, document=doc, vector_rank=rank, vector_score=score)
        for row, (rank, _score) in entity.items():
            if row in candidates:
                if candidates[row].entity_rank is None or rank < candidates[row].entity_rank:
                    candidates[row].entity_rank = rank
            else:
                doc = self.index.get_document(row)
                candidates[row] = RetrievalCandidate(row=row, document=doc, entity_rank=rank)

        for candidate in candidates.values():
            rrf = 0.0
            if candidate.vector_rank is not None:
                rrf += 1.0 / (self.rrf_k + candidate.vector_rank)
            if candidate.sparse_rank is not None:
                rrf += 1.0 / (self.rrf_k + candidate.sparse_rank)
            if candidate.entity_rank is not None:
                rrf += 1.0 / (self.rrf_k + candidate.entity_rank)
            candidate.rrf_score = rrf
            candidate.entity_boost = self._entity_boost(candidate.document, analysis)
            candidate.layer_multiplier = self._layer_multiplier(candidate.document, analysis)
            candidate.type_multiplier = self._type_multiplier(candidate.document, analysis)
            candidate.fused_score = (
                (rrf + candidate.entity_boost) * candidate.layer_multiplier * candidate.type_multiplier
            )

        ranked = sorted(candidates.values(), key=lambda c: (-c.fused_score, c.row))
        return ranked[:top_k]

    def _ranked_from_channel(
        self,
        channel: dict[int, tuple],
        score_attr: str,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if not channel:
            return []
        scores = [score for _, score in channel.values()]
        max_score = max(scores) if scores else 1.0
        result: list[RetrievalCandidate] = []
        for row, (rank, score) in sorted(channel.items(), key=lambda kv: (kv[1][0], kv[0])):
            doc = self.index.get_document(row)
            if doc is None:
                continue
            candidate = RetrievalCandidate(row=row, document=doc)
            setattr(
                candidate,
                score_attr,
                (score / max_score) if (score_attr == "bm25_score" and max_score > 0) else score,
            )
            if score_attr == "bm25_score":
                candidate.sparse_rank = rank
            else:
                candidate.vector_rank = rank
            candidate.rrf_score = 1.0 / (self.rrf_k + rank)
            candidate.fused_score = float(getattr(candidate, score_attr))
            result.append(candidate)
        return result[:top_k]
