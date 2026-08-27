"""citation 绑定与合法性校验（确定性，不信任模型自报来源）。

生成后校验：
- citation key 是否存在于 packet（不得引用未提供文档）
- 是否重复 / 超出允许数量
- 引用位置是否可解析（[S#] 标记）
- abstention 回答不得携带 citation
- API citation metadata 与 key 一致（服务端按 key 回填，
  不采信模型生成的文件名/行号/卡片 ID）

同时执行轻量回答后检查（不重建第二套大型审核系统）：
- evidence 边界标签泄漏
- 答案超预算告警
- evidence 为空时是否输出确定性结论（abstention 结构性防护 + 模板校验）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import EvidencePacket

# [S1] / [S2] ... 引用标记（允许 1-2 位编号）
CITATION_MARKER_RE = re.compile(r"\[S(\d{1,2})\]")

DEFAULT_MAX_CITATIONS = 8

# evidence 边界标签泄漏（模型复述了结构标签而非内容）
_EVIDENCE_TAG_ARTIFACTS = ("<evidence", "</evidence>", "retrieved_evidence", 'trust="untrusted"')

_CITATION_SOURCE_FIELDS = (
    "source_id",
    "source_title",
    "document_type",
    "summary",
    "evidence_excerpt",
    "source_path",
    "source_line",
    "source_line_end",
    "domain_id",
    "reality_status",
    "temporal_scope",
    "content_scope",
    "story_unit_id",
    "kb_revision",
    "index_version",
    "score",
    "rerank_score",
)


@dataclass
class CitationValidation:
    """校验结果：合法 key（按出现顺序去重）+ 全部告警。"""

    valid_keys: list[str] = field(default_factory=list)
    invalid_keys: list[str] = field(default_factory=list)
    duplicate_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def has_invalid(self) -> bool:
        return bool(self.invalid_keys) or self.duplicate_count > 0


class CitationValidator:
    """模型输出 → 合法 citation key 集合 → 服务端回填 metadata。"""

    def __init__(self, max_citations: int = DEFAULT_MAX_CITATIONS):
        self.max_citations = max(1, int(max_citations))

    # -- 解析与校验 ---------------------------------------------------------
    def validate(self, answer_text: str, packet: EvidencePacket) -> CitationValidation:
        result = CitationValidation()
        allowed = packet.key_to_document_id()

        seen: set[str] = set()
        for match in CITATION_MARKER_RE.finditer(answer_text):
            key = f"S{match.group(1)}"
            if key not in allowed:
                result.invalid_keys.append(key)
                continue
            if key in seen:
                result.duplicate_count += 1
                continue
            seen.add(key)
            result.valid_keys.append(key)

        if result.invalid_keys:
            result.warnings.append("invalid_citation_keys:" + ",".join(sorted(set(result.invalid_keys))))
        if result.duplicate_count:
            result.warnings.append(f"duplicate_citations:{result.duplicate_count}")
        if len(result.valid_keys) > self.max_citations:
            result.warnings.append(f"citation_count_truncated:{len(result.valid_keys)}>{self.max_citations}")
            result.valid_keys = result.valid_keys[: self.max_citations]
        if packet.documents and not result.valid_keys:
            result.warnings.append("missing_valid_citation")
        if not packet.documents and CITATION_MARKER_RE.search(answer_text):
            # abstention / 空证据包下出现的任何标记都是非法引用
            result.invalid_keys.extend(m.group(0) for m in CITATION_MARKER_RE.finditer(answer_text))
            result.warnings.append("citations_without_evidence")
            result.valid_keys = []
        return result

    # -- 绑定（服务端权威回填） ----------------------------------------------
    def bind(
        self,
        validation: CitationValidation,
        packet: EvidencePacket,
    ) -> list[dict[str, Any]]:
        """按合法 key 回填 citation metadata；输出顺序 = 首次出现顺序。"""
        if not validation.valid_keys:
            return []
        by_key = {c.get("key"): c for c in packet.citations}
        bound: list[dict[str, Any]] = []
        for key in validation.valid_keys:
            meta = by_key.get(key)
            if meta is None:
                continue
            citation = {k: v for k, v in meta.items() if k in _CITATION_SOURCE_FIELDS}
            citation["key"] = key
            citation["document_id"] = str(meta.get("document_id") or meta.get("source_id") or "")
            bound.append(citation)
        return bound

    # -- 回答清理与后置检查 ---------------------------------------------------
    def sanitize_answer(self, answer_text: str, validation: CitationValidation) -> str:
        """移除全部 [S#] 标记（含非法标记），返回纯文本 answer。"""
        cleaned = CITATION_MARKER_RE.sub("", answer_text)
        cleaned = re.sub(r"[ \t]+([。！？，；、])", r"\1", cleaned)
        return cleaned.strip()

    def post_check(
        self,
        answer_text: str,
        packet: EvidencePacket,
        *,
        answer_max_chars: int = 4000,
    ) -> list[str]:
        """轻量回答后检查（只报结构化告警，不做语义裁决）。"""
        warnings: list[str] = []
        for artifact in _EVIDENCE_TAG_ARTIFACTS:
            if artifact in answer_text:
                warnings.append("evidence_tag_artifact_leak")
                break
        if answer_max_chars > 0 and len(answer_text) > answer_max_chars:
            warnings.append(f"answer_exceeds_budget:{len(answer_text)}>{answer_max_chars}")
        if not packet.documents and answer_text.strip() and _looks_confident(answer_text):
            warnings.append("confident_answer_without_evidence")
        return warnings


def _looks_confident(text: str) -> bool:
    """abstention 模板之外的"确定性结论"信号（保守：只识别强断言句式）。"""
    stripped = text.strip()
    if not stripped:
        return False
    # 弃答/不确定/请求澄清的表述不算确定性结论
    hedge_markers = (
        "不足",
        "无法",
        "没有找到",
        "暂无",
        "不确定",
        "未能确认",
        "补充",
        "澄清",
        "明确说",
        "资料有限",
    )
    if any(marker in stripped for marker in hedge_markers):
        return False
    return stripped.endswith(("。", "！", "？", ".")) and len(stripped) >= 12


def public_citation_view(citation: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    """API 安全视图：默认隐藏绝对本地路径与内部调试分数。

    - source_file：仅文件名（不含目录），产品侧不暴露绝对路径
    - debug=True 时附加 source_path / 分数（管理端诊断用）
    """
    source_path = str(citation.get("source_path") or "")
    view: dict[str, Any] = {
        "key": citation.get("key", ""),
        "document_id": citation.get("document_id", ""),
        "document_type": citation.get("document_type", ""),
        "title": citation.get("source_title") or citation.get("title", ""),
        "summary": citation.get("summary", ""),
        "evidence_excerpt": citation.get("evidence_excerpt", ""),
        "source_file": PurePath(source_path).name if source_path else "",
        "source_line": citation.get("source_line"),
        "source_line_end": citation.get("source_line_end"),
        "domain_id": citation.get("domain_id", ""),
        "reality_status": citation.get("reality_status", ""),
        "temporal_scope": citation.get("temporal_scope", ""),
        "content_scope": citation.get("content_scope", ""),
        "story_unit_id": citation.get("story_unit_id", ""),
        "index_version": citation.get("index_version") or citation.get("kb_revision", ""),
    }
    if debug:
        view["source_path"] = source_path
        view["retrieval_score"] = citation.get("score")
        view["rerank_score"] = citation.get("rerank_score")
    return view
