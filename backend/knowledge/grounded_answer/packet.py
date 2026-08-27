"""Evidence Packet 构建（P6 bundle → P7 证据包）。

- citation key 稳定分配：S1..Sn，按重排后检索排名顺序
- 只有实际进入 prompt（预算内）的文档才拥有 citation key，
  模型无法引用未提供的文档
- 证据块格式化与 P6 ContextBuilder 一致的事实来源，
  但由 P7 重新按 key 标记，供 prompt builder 包裹
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .models import AnswerMode, EvidenceItem, EvidencePacket

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# P6 上下文里证据摘录的截断长度（与 ContextBuilder 默认一致）
_EVIDENCE_CHARS = 360


def _layer_label(doc: Mapping[str, Any]) -> str:
    parts: list[str] = []
    meta = doc.get("metadata") or {}
    if meta.get("volume_number") is not None:
        parts.append(f"第{meta['volume_number']}卷")
    scope = doc.get("content_scope")
    if scope and scope != "unknown":
        parts.append("主线" if scope == "main_story" else "追加")
    temporal = doc.get("temporal_scope")
    if temporal and temporal != "unknown":
        parts.append(temporal)
    reality = doc.get("reality_status")
    if reality and reality not in ("unknown", "objective"):
        parts.append(reality)
    return "·".join(parts)


def _extract_evidence_text(content: str, limit: int = _EVIDENCE_CHARS) -> str:
    """从 canonical 文档 content 提取「证据：」段（引用原文形式）。"""
    if not content or "\n证据：" not in content:
        return ""
    raw = content.split("\n证据：", 1)[1].strip()
    if not raw:
        return ""
    shown = raw[:limit]
    if len(raw) > limit:
        shown = shown.rstrip() + "…"
    return shown


class EvidencePacketBuilder:
    """P6 bundle → EvidencePacket。"""

    def __init__(self, evidence_budget_chars: int = 2400):
        # 预算略高于 P6 context 预算（2000）：P6 已做一次预算裁剪，
        # 此处防止异常超大 bundle 直接塞满 prompt
        self.evidence_budget = max(600, int(evidence_budget_chars))

    def build(
        self,
        query: str,
        bundle: Mapping[str, Any],
        answer_mode: AnswerMode,
        *,
        warnings: list[str] | None = None,
    ) -> EvidencePacket:
        """bundle 契约见 P6 RagPipeline.retrieve 返回值。"""
        packet_warnings = list(warnings or [])
        results = list(bundle.get("results") or [])
        citations = list(bundle.get("citations") or [])
        domains = [str(d) for d in (bundle.get("domains") or [])]

        result_by_id: dict[str, Mapping[str, Any]] = {}
        for item in results:
            result_by_id[str(item.get("id"))] = item

        documents: list[EvidenceItem] = []
        bound_citations: list[dict[str, Any]] = []
        blocks: list[str] = []
        used = 0
        truncated = False

        for citation in citations:
            doc_id = str(citation.get("source_id") or "")
            if not doc_id:
                continue
            item = result_by_id.get(doc_id)
            if item is None:
                # citation 与 results 不一致（索引更新竞态）：跳过该引用，
                # 不允许引用无完整 metadata 的文档
                packet_warnings.append("citation_missing_document")
                continue

            key = f"S{len(documents) + 1}"
            # EvidenceItem 构造后用 to_block() 统一格式化（与 prompt 证据区共用）
            source = item.get("source") or {}
            evidence_item = EvidenceItem(
                citation_key=key,
                document_id=doc_id,
                document_type=str(item.get("document_type") or "unknown"),
                title=str(item.get("title") or ""),
                summary=str(item.get("summary") or ""),
                evidence_text=_extract_evidence_text(str(item.get("content") or "")),
                source_path=str(source.get("source_path") or ""),
                line_start=source.get("line_start"),
                line_end=source.get("line_end"),
                domain_id=str(item.get("domain_id") or (domains[0] if domains else "")),
                reality_status=str(item.get("reality_status") or "unknown"),
                temporal_scope=str(item.get("temporal_scope") or "unknown"),
                content_scope=str(item.get("content_scope") or "unknown"),
                story_unit_id=str((item.get("metadata") or {}).get("story_unit_id") or ""),
                index_version=str(item.get("index_version") or ""),
                retrieval_score=float(item.get("fused_score") or 0.0),
                rerank_score=(float(item["rerank_score"]) if item.get("rerank_score") is not None else None),
                layer_label=_layer_label(item),
            )
            block = evidence_item.to_block()
            if used + len(block) + 1 > self.evidence_budget:
                truncated = True
                break

            # 服务端权威 citation metadata（key 绑定；API 输出由
            # public_citation_view 再做安全视图转换）
            citation_meta = dict(citation)
            citation_meta["key"] = key
            citation_meta["document_id"] = doc_id
            citation_meta["summary"] = str(item.get("summary") or "")

            documents.append(evidence_item)
            bound_citations.append(citation_meta)
            blocks.append(block)
            used += len(block) + 1

        if truncated:
            packet_warnings.append("evidence_budget_truncated")

        return EvidencePacket(
            query=query,
            domain_id=domains[0] if domains else None,
            answer_mode=answer_mode,
            retrieval_confidence=float(bundle.get("confidence") or 0.0),
            documents=tuple(documents),
            context_text="\n\n".join(blocks),
            citations=tuple(bound_citations),
            filters=dict(bundle.get("filters") or {}),
            warnings=packet_warnings,
            evidence_budget=self.evidence_budget,
            truncated=truncated,
            abstention_reason=str(bundle.get("abstention_reason") or ""),
            query_analysis=dict(bundle.get("query_analysis") or {}),
            index_version=str(documents[0].index_version) if documents else "",
        )


def is_p6_bundle(bundle: Mapping[str, Any] | None) -> bool:
    """区分 P6 管线 bundle 与 legacy RAGHelper bundle（契约判别）。"""
    return bundle is not None and "query_analysis" in bundle and "domains" in bundle
