"""Reranker 接入（P6）。

优先接入项目现有 CrossEncoderReranker（bge-reranker-base，
RERANKER_ENABLED 控制）；本地模型不可用时走确定性特征降级
（不吞掉精确关系/实体命中，不因长 evidence 获得优势，不改文档）。

降级打分特征（全部通用规则，无作品特例）：
- 实体重合率（查询实体 ∩ 文档实体）
- 文档类型与查询意图匹配
- 查询词在文档中的覆盖率（jieba 分词，停用词剔除）
- 关系方向完整性（关系卡主体/对象同时命中查询实体）
- 叙事层与查询偏好对齐
- 超长内容轻微惩罚
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .query import QueryAnalysis
    from .retrieval import RetrievalCandidate

logger = logging.getLogger(__name__)

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
    "length_penalty": -0.10,
}

# 身份意图词（通用中文）：查询问"是谁/什么身份"时，
# 含"身份"关键词的文档获得对齐加成
_IDENTITY_INTENT_WORDS = ["是谁", "什么身份", "是什么人", "是什么人物", "个人资料"]

_LONG_CONTENT_CHARS = 1500


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _tokenize_query_terms(query: str) -> list[str]:
    """查询词切分：优先 jieba（中文正确分词），缺失时确定性回退。"""
    try:
        import jieba

        tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
    except ImportError:
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", query)
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


class DeterministicReranker:
    """无模型降级重排：确定性特征打分，稳定排序。"""

    def rerank(
        self,
        analysis: QueryAnalysis,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[tuple]:
        """返回 (candidate, rerank_score) 列表，按分数降序、row 升序稳定排序。"""
        query_entities = set(analysis.entities)
        # 词覆盖基于归一化查询（别名已替换为规范名）
        query_text = analysis.normalized_query or analysis.original_query
        query_terms = _tokenize_query_terms(query_text)
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

            # 查询词覆盖率
            if query_terms:
                covered = sum(1 for term in query_terms if term in text)
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
    """重排门面：CrossEncoder 可用则用之，否则确定性降级。"""

    def __init__(self, cross_encoder: Any | None = None, cross_encoder_enabled: bool | None = None):
        # cross_encoder 可注入（测试）；None 时按环境变量惰性获取现有单例
        self._cross_encoder = cross_encoder
        self._cross_encoder_enabled = cross_encoder_enabled
        self._cross_encoder_unavailable = False
        self._last_used_cross_encoder = False
        self.deterministic = DeterministicReranker()

    @property
    def uses_cross_encoder(self) -> bool:
        """最近一次 rerank 是否实际使用了 CrossEncoder（logit 分数语义）。"""
        return self._last_used_cross_encoder

    def _resolve_cross_encoder(self) -> Any | None:
        if self._cross_encoder is not None:
            return self._cross_encoder
        if self._cross_encoder_unavailable:
            return None
        enabled = self._cross_encoder_enabled
        if enabled is None:
            enabled = _env_flag("RERANKER_ENABLED", "false")
        if not enabled:
            return None
        try:
            from knowledge.reranker import get_reranker

            encoder = get_reranker()
            # 探测一次模型可用性（加载失败时 rerank 返回原始顺序）
            probe = encoder.rerank("probe", [{"title": "t", "content": "c"}], top_k=1)
            if probe and "rerank_score" not in probe[0]:
                self._cross_encoder_unavailable = True
                logger.info("CrossEncoder 模型不可用，P6 使用确定性降级重排")
                return None
            return encoder
        except Exception as e:  # noqa: BLE001 - 任何失败都走降级
            self._cross_encoder_unavailable = True
            logger.info("CrossEncoder 初始化失败，P6 使用确定性降级重排: %s", e)
            return None

    def rerank(
        self,
        analysis: QueryAnalysis,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """重排候选：输出稳定排序，保留原始分数与来源，不修改文档内容。"""
        if len(candidates) <= 1:
            return list(candidates)[:top_k]

        encoder = self._resolve_cross_encoder()
        if encoder is not None:
            try:
                result = self._cross_encoder_rerank(analysis, candidates, top_k, encoder)
                self._last_used_cross_encoder = True
                return result
            except Exception as e:  # noqa: BLE001
                logger.warning("CrossEncoder 重排失败，降级确定性重排: %s", e)

        self._last_used_cross_encoder = False
        scored = self.deterministic.rerank(analysis, candidates)
        reranked: list[RetrievalCandidate] = []
        for candidate, rerank_score in scored[:top_k]:
            candidate.rerank_score = rerank_score
            reranked.append(candidate)
        return reranked

    def _cross_encoder_rerank(
        self,
        analysis: QueryAnalysis,
        candidates: list[RetrievalCandidate],
        top_k: int,
        encoder: Any,
    ) -> list[RetrievalCandidate]:
        payload = [candidate.to_dict() for candidate in candidates]
        reranked_dicts: list[dict[str, Any]] = encoder.rerank(
            analysis.original_query, payload, top_k=max(top_k, len(candidates))
        )
        if not reranked_dicts:
            raise RuntimeError("CrossEncoder 返回空结果")
        if "rerank_score" not in reranked_dicts[0]:
            # 模型未真正加载（原始顺序回退）→ 走确定性降级
            raise RuntimeError("CrossEncoder 未加载模型（原始顺序回退）")
        by_id = {candidate.document.id: candidate for candidate in candidates}
        ordered: list[RetrievalCandidate] = []
        for item in reranked_dicts[:top_k]:
            candidate = by_id.get(str(item.get("id")))
            if candidate is None:
                continue
            candidate.rerank_score = float(item.get("rerank_score", 0.0))
            ordered.append(candidate)
        return ordered
