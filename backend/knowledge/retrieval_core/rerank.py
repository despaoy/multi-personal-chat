"""Shared reranker integration.

优先接入项目现有 CrossEncoderReranker（bge-reranker-base，
RERANKER_ENABLED 控制）；显式禁用时使用确定性特征排序
（不吞掉精确关系/实体命中，不因长 evidence 获得优势，不改文档）。

确定性打分特征（全部通用规则，无作品特例）：
- 实体重合率（查询实体 ∩ 文档实体）
- 文档类型与查询意图匹配
- 查询词在文档中的覆盖率（jieba 分词，停用词剔除）
- 关系方向完整性（关系卡主体/对象同时命中查询实体）
- 叙事层与查询偏好对齐
- 超长内容轻微惩罚
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from infra.environment import read_bool

from .tokenization import segment

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .query import QueryAnalysis
    from .retrieval import RetrievalCandidate

# 查询词覆盖计算用停用词（与 corrective_rag 约定一致的通用词表）
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
    "type_match": 0.15,
    "keyword_coverage": 0.25,
    "relation_direction": 0.10,
    "layer_alignment": 0.10,
    "story_hit": 0.05,
    "identity_alignment": 0.08,
    "predicate_alignment": 0.16,
    "relation_type_alignment": 0.16,
    "causal_evidence": 0.12,
    "length_penalty": -0.10,
}

# 身份意图词（通用中文）：查询问"是谁/什么身份"时，
# 含"身份"关键词的文档获得对齐加成
_IDENTITY_INTENT_WORDS = ["是谁", "什么身份", "是什么人", "是什么人物", "个人资料"]

_LONG_CONTENT_CHARS = 1500
_CAUSAL_MARKERS = ("因为", "由于", "所以", "导致", "造成", "原因", "起因", "因而", "因此")
_TERM_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "上吊": ("自缢",),
    "自杀": ("自缢", "轻生", "主动赴死"),
    "车祸": ("交通事故",),
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
    """显式无模型重排：确定性特征打分，稳定排序。"""

    def rerank(
        self,
        analysis: QueryAnalysis,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[tuple]:
        """返回 (candidate, rerank_score) 列表，按分数降序、row 升序稳定排序。"""
        query_entities = set(analysis.entities)
        # 词覆盖基于归一化查询（别名已替换为规范名）
        query_text = analysis.normalized_query or analysis.original_query
        query_terms = _tokenize_query_terms(query_text, query_entities)
        identity_intent = any(word in query_text for word in _IDENTITY_INTENT_WORDS)
        scored: list[tuple] = []
        for candidate in candidates:
            doc = candidate.document
            text = f"{doc.title} {doc.summary} {doc.embedding_text}"
            score = 0.0

            # 实体重合率
            if query_entities:
                hits = len(query_entities & set(doc.entities))
                score += _WEIGHTS["entity_overlap"] * (hits / len(query_entities))

            # 文档类型意图匹配
            if analysis.doc_type_preferences and doc.document_type in analysis.doc_type_preferences:
                score += _WEIGHTS["type_match"]

            # 身份意图对齐：问"是谁/什么身份"时含"身份"的文档加成
            if identity_intent and "身份" in text:
                score += _WEIGHTS["identity_alignment"]

            if analysis.predicate_preferences and doc.metadata.get("predicate") in analysis.predicate_preferences:
                score += _WEIGHTS["predicate_alignment"]

            if (
                analysis.relation_type_preferences
                and doc.metadata.get("relation") in analysis.relation_type_preferences
            ):
                score += _WEIGHTS["relation_type_alignment"]

            causal_text = f"{doc.title} {doc.summary}"
            if analysis.causal_intent and any(marker in causal_text for marker in _CAUSAL_MARKERS):
                score += _WEIGHTS["causal_evidence"]

            # 查询词覆盖率
            if query_terms:
                covered = sum(
                    1
                    for term in query_terms
                    if term in text or any(alternative in text for alternative in _TERM_ALTERNATIVES.get(term, ()))
                )
                score += _WEIGHTS["keyword_coverage"] * (covered / len(query_terms))

            # 关系方向完整性：关系卡主体与对象同时命中查询实体
            if doc.document_type == "relation" and query_entities:
                subject = doc.metadata.get("subject")
                target = doc.metadata.get("target")
                if subject in query_entities and target in query_entities:
                    score += _WEIGHTS["relation_direction"]

            # 叙事层对齐
            if analysis.reality_preferences and doc.reality_status in analysis.reality_preferences:
                score += _WEIGHTS["layer_alignment"]

            # 故事标题命中
            if analysis.story_hits and any(
                hit in doc.title or hit in (doc.metadata.get("story_title") or "") for hit in analysis.story_hits
            ):
                score += _WEIGHTS["story_hit"]

            # 超长内容惩罚（避免长 evidence 主导）
            if len(doc.content) > _LONG_CONTENT_CHARS:
                score += _WEIGHTS["length_penalty"] * min(1.0, len(doc.content) / (_LONG_CONTENT_CHARS * 4))

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
        if not reranked_dicts:
            raise RuntimeError("CrossEncoder 返回空结果")
        if "rerank_score" not in reranked_dicts[0]:
            raise RuntimeError("CrossEncoder 未加载模型（原始顺序回退）")
        by_id = {candidate.document.id: candidate for candidate in candidates}
        ordered: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for item in reranked_dicts:
            candidate = by_id.get(str(item.get("id")))
            if candidate is None or candidate.document.id in seen:
                raise RuntimeError("CrossEncoder returned unknown or duplicate IDs")
            score = float(item["rerank_score"])
            if not math.isfinite(score):
                raise RuntimeError("CrossEncoder returned non-finite score")
            seen.add(candidate.document.id)
            candidate.rerank_score = score
            candidate.rerank_method = "cross_encoder"
            ordered.append(candidate)
        if len(ordered) != len(candidates):
            raise RuntimeError("CrossEncoder returned incomplete candidate set")
        return ordered[:top_k]
