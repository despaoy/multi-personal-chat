"""Explicit model reranking or literal relevance ranking.

Model failures propagate. The deterministic mode scores registered objects,
literal term coverage and structured relation endpoints, without intent labels,
guessed synonyms or a penalty for complete long sources.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import replace
from numbers import Real
from typing import TYPE_CHECKING, Any

from infra.environment import read_bool

from .tokenization import segment

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .query import QueryAnalysis
    from .retrieval import RetrievalCandidate

# Stop words affect lexical relevance only; complete queries remain model input.
_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "这",
    "那",
    "怎么",
    "什么",
    "为什么",
    "谁",
    "哪",
    "几",
    "哪里",
    "哪个",
    "请问",
    "一下",
    "可能",
    "应该",
    "和",
    "与",
    "跟",
    "对",
    "对于",
    "关于",
    "有",
    "没有",
    "不",
    "很",
    "都",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "how",
    "why",
}

_WEIGHTS = {
    "entity_overlap": 0.30,
    "keyword_coverage": 0.25,
    "relation_direction": 0.10,
    "story_hit": 0.05,
}


def _tokenize_query_terms(query: str, excluded_terms: set[str] | None = None) -> list[str]:
    """分词后排除停用词与已识别实体。"""
    tokens = segment(query)
    excluded = excluded_terms or set()
    return [
        token
        for token in tokens
        if token not in _STOPWORDS
        and token not in excluded
        and (len(token) >= 2 or bool(re.fullmatch(r"[\u4e00-\u9fff]", token)))
    ]


class DeterministicReranker:
    """Literal relevance ranking with no inferred query categories."""

    def rerank(
        self,
        analysis: QueryAnalysis,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[tuple]:
        query_entities = set(analysis.entities)
        query_terms = _tokenize_query_terms(analysis.normalized_query or analysis.original_query, query_entities)
        scored: list[tuple] = []
        for candidate in candidates:
            doc = candidate.document
            text = f"{doc.title} {doc.summary} {doc.embedding_text}"
            score = 0.0
            if query_entities:
                score += _WEIGHTS["entity_overlap"] * len(query_entities & set(doc.entities)) / len(query_entities)
            if query_terms:
                score += _WEIGHTS["keyword_coverage"] * sum(term in text for term in query_terms) / len(query_terms)
            if (
                doc.document_type == "relation"
                and query_entities
                and doc.metadata.get("subject") in query_entities
                and doc.metadata.get("target") in query_entities
            ):
                score += _WEIGHTS["relation_direction"]
            if analysis.story_hits and any(
                hit in doc.title or hit in (doc.metadata.get("story_title") or "") for hit in analysis.story_hits
            ):
                score += _WEIGHTS["story_hit"]
            scored.append((candidate, round(score, 4)))
        scored.sort(key=lambda pair: (-pair[1], -pair[0].fused_score, pair[0].row))
        return scored


class PipelineReranker:
    """重排门面：按配置选择模型或确定性重排，模型失败直接上报。"""

    def __init__(
        self,
        cross_encoder: Any | None = None,
        cross_encoder_enabled: bool | None = None,
        *,
        text_view: str = "content",
    ):
        if text_view not in {"content", "summary", "embedding_text"}:
            raise ValueError("unknown reranker text view")
        self.text_view = text_view
        # cross_encoder 可注入（测试）；None 时按环境变量惰性获取现有单例
        self._cross_encoder = cross_encoder
        self._cross_encoder_enabled = cross_encoder_enabled
        self.deterministic = DeterministicReranker()

    def _resolve_cross_encoder(self) -> Any | None:
        if self._cross_encoder is not None:
            return self._cross_encoder
        enabled = self._cross_encoder_enabled
        if enabled is None:
            enabled = read_bool(os.environ, "RERANKER_ENABLED")
        if not enabled:
            return None
        from knowledge.reranker import get_reranker

        return get_reranker()

    def rerank(
        self,
        analysis: QueryAnalysis,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """重排候选：输出稳定排序，保留原始分数与来源，不修改文档内容。"""
        if not candidates or top_k <= 0:
            return []

        # Scores and method diagnostics are request-local; never mutate recalled
        # objects that an index/test/parallel caller may reuse.
        candidates = [replace(candidate) for candidate in candidates]

        encoder = self._resolve_cross_encoder()
        if encoder is not None:
            return self._cross_encoder_rerank(analysis, candidates, top_k, encoder)

        scored = self.deterministic.rerank(analysis, candidates)
        reranked: list[RetrievalCandidate] = []
        for candidate, rerank_score in scored[:top_k]:
            candidate.rerank_score = rerank_score
            candidate.rerank_method = "deterministic"
            reranked.append(candidate)
        return reranked

    def _cross_encoder_rerank(
        self,
        analysis: QueryAnalysis,
        candidates: list[RetrievalCandidate],
        top_k: int,
        encoder: Any,
    ) -> list[RetrievalCandidate]:
        payload = []
        for candidate in candidates:
            item = candidate.to_dict()
            if self.text_view == "summary":
                item["content"] = f"{candidate.document.title}\n{candidate.document.summary}"
            elif self.text_view == "embedding_text":
                item["content"] = candidate.document.embedding_text
            payload.append(item)
        reranked_dicts: list[dict[str, Any]] = encoder.rerank(
            analysis.original_query, payload, top_k=max(top_k, len(candidates))
        )
        if not isinstance(reranked_dicts, list) or len(reranked_dicts) != len(candidates):
            raise RuntimeError("CrossEncoder returned incomplete candidate set")
        by_id = {candidate.document.id: candidate for candidate in candidates}
        ordered: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for item in reranked_dicts:
            if not isinstance(item, dict):
                raise RuntimeError("CrossEncoder returned an invalid candidate row")
            identity = item.get("id")
            if not isinstance(identity, str) or identity not in by_id or identity in seen:
                raise RuntimeError("CrossEncoder returned unknown or duplicate IDs")
            raw_score = item.get("rerank_score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                raise RuntimeError("CrossEncoder returned an invalid numeric score")
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("CrossEncoder returned non-finite score")
            candidate = by_id[identity]
            seen.add(identity)
            candidate.rerank_score = score
            candidate.rerank_method = "cross_encoder"
            ordered.append(candidate)
        return ordered[:top_k]
