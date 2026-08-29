"""上下文组装与引用（P6）。

ContextBuilder 在字符预算内组装证据：
- 优先保留高分证据（重排分数序）
- 同一卡片不重复；等价内容（同主体+谓词 / 同关系三元组 / 同标题）去重
- 允许事实、关系、事件组合
- 每条带来源引用（文件 + 行号 + 卡片 ID）
- evidence 一律以引用数据形式呈现（「」包裹），
  不作为可执行指令解释
- 检索内容不覆盖系统指令：产物为独立参考块，由调用方注入
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .query import QueryAnalysis
    from .retrieval import RetrievalCandidate

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_CHARS = 6000
DEFAULT_MAX_ITEMS = 8
DEFAULT_PER_DOC_CHARS = 1200
DEFAULT_EVIDENCE_CHARS = 700


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _layer_label(doc) -> str:
    parts = []
    if doc.metadata.get("volume_number") is not None:
        parts.append(f"第{doc.metadata['volume_number']}卷")
    if doc.content_scope and doc.content_scope != "unknown":
        parts.append("主线" if doc.content_scope == "main_story" else "追加")
    if doc.temporal_scope and doc.temporal_scope != "unknown":
        parts.append(doc.temporal_scope)
    if doc.reality_status and doc.reality_status not in ("unknown", "objective"):
        parts.append(doc.reality_status)
    return "·".join(parts)


@dataclass
class Citation:
    """单条引用（与现有前端 citation 字段兼容并扩展）。"""

    source_id: str
    source_title: str
    domain_id: str
    document_type: str
    evidence_excerpt: str
    score: float
    source_path: str
    line_start: int | None
    line_end: int | None
    reality_status: str
    temporal_scope: str
    content_scope: str
    rerank_score: float | None = None
    index_version: str = ""
    story_unit_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "domain_id": self.domain_id,
            "document_type": self.document_type,
            "evidence_excerpt": self.evidence_excerpt,
            "score": round(self.score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "kb_revision": self.index_version,
            "source_path": self.source_path,
            "source_line": self.line_start,
            "source_line_end": self.line_end,
            "section": self.document_type,
            "version": self.index_version,
            "reality_status": self.reality_status,
            "temporal_scope": self.temporal_scope,
            "content_scope": self.content_scope,
            "story_unit_id": self.story_unit_id,
        }


@dataclass
class RetrievedContext:
    """检索组装结果：条目 + 上下文文本 + 引用。"""

    query: str
    domains: list[str]
    items: list[dict[str, Any]] = field(default_factory=list)
    context_text: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    used_chars: int = 0
    budget_chars: int = 0
    skipped_duplicates: int = 0


_TYPE_LABELS = {"fact": "事实", "relation": "关系", "event": "事件"}


class ContextBuilder:
    """预算内证据组装。"""

    def __init__(
        self,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        max_items: int = DEFAULT_MAX_ITEMS,
        per_doc_chars: int = DEFAULT_PER_DOC_CHARS,
        evidence_chars: int = DEFAULT_EVIDENCE_CHARS,
    ):
        self.budget_chars = max(200, int(budget_chars))
        self.max_items = max(1, int(max_items))
        self.per_doc_chars = max(100, int(per_doc_chars))
        self.evidence_chars = max(50, int(evidence_chars))

    # -- 等价内容签名 -------------------------------------------------------
    @staticmethod
    def _equivalence_key(doc) -> str | None:
        if doc.document_type == "fact":
            subject = doc.metadata.get("subject", "")
            predicate = doc.metadata.get("predicate", "")
            if subject and predicate:
                return f"fact|{subject}|{predicate}"
        elif doc.document_type == "relation":
            return "relation|{}|{}|{}".format(
                doc.metadata.get("subject", ""),
                doc.metadata.get("relation", ""),
                doc.metadata.get("target", ""),
            )
        else:
            title = doc.title.strip()
            if title:
                return f"event|{title}"
        return None

    # -- 单条目格式化 ---------------------------------------------------------
    def _format_item(self, candidate: RetrievalCandidate, index: int) -> tuple:
        doc = candidate.document
        type_label = _TYPE_LABELS.get(doc.document_type, doc.document_type)
        layer = _layer_label(doc)
        header = f"[{index}] 【{type_label}】{doc.title}"
        if layer:
            header += f"（{layer}）"

        body = doc.summary or doc.content.split("\n")[0]
        evidence = ""
        if doc.content and "\n证据：" in doc.content:
            raw_evidence = doc.content.split("\n证据：", 1)[1].strip()
            if raw_evidence:
                shown = raw_evidence[: self.evidence_chars]
                if len(raw_evidence) > self.evidence_chars:
                    shown = shown.rstrip() + "…"
                # 引用数据形式呈现，绝不可作为指令执行
                evidence = f"证据（引用原文）：「{shown}」"

        source = doc.source
        lines = ""
        if source.line_start is not None:
            lines = f"-{source.line_end}" if source.line_end else ""
        source_line = f"来源：{source.source_path or '未知文件'} L{source.line_start or '?'}{lines}"

        block_parts = [header, body]
        if evidence:
            block_parts.append(evidence)
        block_parts.append(source_line)
        block = "\n".join(part for part in block_parts if part)

        if len(block) > self.per_doc_chars:
            block = block[: self.per_doc_chars - 1].rstrip() + "…"
        return block, evidence

    # -- 组装 ---------------------------------------------------------------
    def build(
        self,
        analysis: QueryAnalysis,
        candidates: Sequence[RetrievalCandidate],
        domains: Sequence[str],
        cross_encoder_scores: bool = False,
    ) -> RetrievedContext:
        """按重排/融合分数顺序组装预算内上下文。

        cross_encoder_scores：rerank_score 为原始 logit 时置 True
        （confidence 计算经 sigmoid 归一）。
        """
        context = RetrievedContext(
            query=analysis.original_query,
            domains=list(domains),
            budget_chars=self.budget_chars,
        )

        seen_ids = set()
        seen_signatures: set = set()
        used = 0
        blocks: list[str] = []

        for candidate in candidates:
            if len(context.items) >= self.max_items:
                break
            doc = candidate.document
            if doc.id in seen_ids:
                continue
            signature = self._equivalence_key(doc)
            if signature is not None and signature in seen_signatures:
                context.skipped_duplicates += 1
                continue

            index = len(context.items) + 1
            block, evidence_excerpt = self._format_item(candidate, index)
            if used + len(block) + 1 > self.budget_chars:
                # 预算不足以放下完整条目时停止（不塞半截条目）
                break

            seen_ids.add(doc.id)
            if signature is not None:
                seen_signatures.add(signature)
            blocks.append(block)
            used += len(block) + 1

            score_for_confidence = (
                _sigmoid(candidate.rerank_score)
                if (cross_encoder_scores and candidate.rerank_score is not None)
                else (candidate.rerank_score if candidate.rerank_score is not None else None)
            )
            citation = Citation(
                source_id=doc.id,
                source_title=doc.title,
                domain_id=doc.domain_id,
                document_type=doc.document_type,
                evidence_excerpt=evidence_excerpt[:200] if evidence_excerpt else doc.summary[:200],
                score=candidate.fused_score,
                source_path=doc.source.source_path,
                line_start=doc.source.line_start,
                line_end=doc.source.line_end,
                reality_status=doc.reality_status,
                temporal_scope=doc.temporal_scope,
                content_scope=doc.content_scope,
                rerank_score=score_for_confidence,
                index_version=doc.index_version,
                story_unit_id=str(doc.metadata.get("story_unit_id") or ""),
            )
            context.citations.append(citation.to_dict())
            context.items.append(
                {
                    "id": doc.id,
                    "domain_id": doc.domain_id,
                    "document_type": doc.document_type,
                    "title": doc.title,
                    "summary": doc.summary,
                    "score": round(candidate.fused_score, 6),
                    "rerank_score": (round(candidate.rerank_score, 4) if candidate.rerank_score is not None else None),
                    "block": block,
                }
            )
            top_confidence = score_for_confidence
            if top_confidence is not None:
                context.confidence = max(context.confidence, float(top_confidence))

        context.context_text = "\n\n".join(blocks)
        context.used_chars = len(context.context_text)
        return context
