"""回答模式识别与 abstention 策略（确定性规则，无额外模型调用）。

信号来源（复用上游意图检测、角色知识域门控与检索结果，不新增低质量分类模型）：
- 角色知识检索结果的 confidence / abstained / results / query_analysis
- 检索置信度、top1/top2 差距、sparse 与 vector 是否共同命中、实体匹配
- 调用方是否注入 persona（角色语气）

阈值均为全局配置，不含任何作品特例。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from knowledge.curated_sources import curated_source_receipts

from .models import AnswerMode, FailureKind

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AbstentionThresholds:
    """abstention / clarification 门槛（综合多信号，非单一向量分数）。"""

    # 低于 hard → abstention（检索层已输出 abstained，这里二次确认）
    hard_confidence: float = 0.25
    # [hard, soft) 且实体缺失 → clarification；否则 grounded 带限定
    soft_confidence: float = 0.45
    # top1 归一化重排分足够高且证据少 → direct_answer
    direct_top1_score: float = 0.75
    direct_max_documents: int = 2
    # top1/top2 差距过小 → 置信度打折（证据不一致信号）
    top_gap_penalty_threshold: float = 0.05
    top_gap_penalty: float = 0.05


@dataclass
class ModeDecision:
    mode: AnswerMode
    abstention_reason: str = ""  # FailureKind 值或空
    warnings: list[str] | None = None
    effective_confidence: float = 0.0


def _top_rerank_scores(results: list[Mapping[str, Any]]) -> list[float | None]:
    """Read the first two authoritative ranking scores; absence means no reranker."""
    scores: list[float | None] = []
    for item in results[:2]:
        value = item.get("rerank_score")
        score = float(value) if value is not None else None
        if isinstance(value, bool) or (score is not None and not isfinite(score)):
            raise ValueError("rerank_score must be a finite number")
        scores.append(score)
    return scores


class AnswerModeDecider:
    """确定性回答模式决策。"""

    def __init__(self, thresholds: AbstentionThresholds | None = None):
        self.thresholds = thresholds or AbstentionThresholds()

    def decide(
        self,
        bundle: Mapping[str, Any] | None,
        *,
        persona: bool = False,
    ) -> ModeDecision:
        """保留无域状态；已检索结果依据证据和置信度决定回答模式。"""
        if bundle is None:
            return ModeDecision(
                mode=AnswerMode.NO_RAG,
                abstention_reason=FailureKind.NO_DOMAIN.value,
            )

        warnings: list[str] = []
        results = list(bundle["results"])
        evidence_count = (
            len(curated_source_receipts(bundle))
            if "curated_source_catalog" in bundle else len(bundle["citations"])
        )
        confidence = float(bundle["confidence"])
        if isinstance(bundle["confidence"], bool) or not isfinite(confidence):
            raise ValueError("retrieval confidence must be a finite number")
        abstained = bundle["abstained"]
        if not isinstance(abstained, bool):
            raise TypeError("retrieval abstained must be a boolean")
        analysis = bundle.get("query_analysis") or {}
        entities = list(analysis.get("entities") or [])

        if abstained or not results or not evidence_count:
            reason = FailureKind.NO_RETRIEVAL.value if not results else FailureKind.LOW_CONFIDENCE.value
            return ModeDecision(
                mode=AnswerMode.ABSTENTION,
                abstention_reason=reason,
                warnings=warnings,
                effective_confidence=confidence,
            )

        # top1/top2 差距小 → 证据可能不一致，置信度打折（温和惩罚）
        top_scores = _top_rerank_scores(results)
        effective = confidence
        if (
            len(top_scores) >= 2
            and top_scores[0] is not None
            and top_scores[1] is not None
            and top_scores[0] >= self.thresholds.hard_confidence
        ):
            gap = top_scores[0] - top_scores[1]
            if gap < self.thresholds.top_gap_penalty_threshold:
                effective = max(0.0, confidence - self.thresholds.top_gap_penalty)
                warnings.append("ambiguous_top_candidates")

        if effective < self.thresholds.hard_confidence:
            return ModeDecision(
                mode=AnswerMode.ABSTENTION,
                abstention_reason=FailureKind.LOW_CONFIDENCE.value,
                warnings=warnings,
                effective_confidence=effective,
            )

        # sparse 与 vector 是否共同命中（channels 信号缺失时不惩罚）
        if self._no_channel_agreement(bundle):
            effective = max(0.0, effective - 0.03)
            warnings.append("single_channel_hit")

        if effective < self.thresholds.soft_confidence:
            if not entities:
                return ModeDecision(
                    mode=AnswerMode.CLARIFICATION,
                    warnings=[*warnings, "low_confidence_no_entity"],
                    effective_confidence=effective,
                )
            return ModeDecision(
                mode=AnswerMode.GROUNDED_CHARACTER_ANSWER if persona else AnswerMode.GROUNDED_ANSWER,
                warnings=[*warnings, "low_confidence_hedged"],
                effective_confidence=effective,
            )

        top1 = top_scores[0] if top_scores else None
        if (
            top1 is not None
            and top1 >= self.thresholds.direct_top1_score
            and evidence_count <= self.thresholds.direct_max_documents
            and not persona
        ):
            return ModeDecision(
                mode=AnswerMode.DIRECT_ANSWER,
                warnings=warnings,
                effective_confidence=effective,
            )

        return ModeDecision(
            mode=AnswerMode.GROUNDED_CHARACTER_ANSWER if persona else AnswerMode.GROUNDED_ANSWER,
            warnings=warnings,
            effective_confidence=effective,
        )

    @staticmethod
    def _no_channel_agreement(bundle: Mapping[str, Any]) -> bool:
        """top 结果是否只有单一通道命中（sparse/vector 未共同命中）。"""
        top = (bundle.get("results") or [])[:1]
        if not top:
            return False
        item = top[0]
        vector_hit = bool((item.get("vector_score") or 0) > 0)
        bm25_hit = bool((item.get("bm25_score") or 0) > 0)
        # 任一通道缺失分数（旧索引/降级）不视为不一致
        if item.get("vector_score") is None or item.get("bm25_score") is None:
            return False
        return not (vector_hit and bm25_hit)
