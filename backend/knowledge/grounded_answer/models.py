"""P7 grounded-answer 数据契约。

检索（P6）与回答（P7）分离：
- P6 负责：找到知识、排序、过滤、组装上下文、提供 citation metadata
- P7 负责：判断是否需要知识回答、把证据转成受约束的回答、
  标明事实层级、返回合法引用、证据不足时拒绝编造

EvidencePacket 是回答模型可使用知识的唯一来源；
作品专属表达规则只存在于 domain 配置（prompt_supplement），
核心模块不含任何人物名/卷名/固定问句。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerMode(str, Enum):  # noqa: UP042 - runtime 仍兼容 Python 3.10
    """回答模式（通用，domain 无关）。"""

    DIRECT_ANSWER = "direct_answer"  # 单一明确事实，证据充足
    GROUNDED_ANSWER = "grounded_answer"  # 常规证据约束回答
    GROUNDED_CHARACTER_ANSWER = "grounded_character_answer"  # 角色语气 + 事实约束
    CLARIFICATION = "clarification"  # 证据有限，回答并请求澄清
    ABSTENTION = "abstention"  # 证据不足，拒绝编造
    NO_RAG = "no_rag"  # 未命中知识域/无需检索


class FailureKind(str, Enum):  # noqa: UP042 - runtime 仍兼容 Python 3.10
    """失败分类（对外简洁，内部日志结构化）。"""

    NO_DOMAIN = "no_domain"
    NO_RETRIEVAL = "no_retrieval"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    LOW_CONFIDENCE = "low_confidence"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    GENERATION_TIMEOUT = "generation_timeout"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    INVALID_CITATION = "invalid_citation"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    CLIENT_CANCELLED = "client_cancelled"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class EvidenceItem:
    """单条证据：绑定 citation key 的 canonical 文档快照。"""

    citation_key: str  # "S1" / "S2" ...
    document_id: str
    document_type: str  # fact / relation / event（domain 定义）
    title: str
    summary: str
    evidence_text: str  # 证据摘录（引用原文形式）
    source_path: str
    line_start: int | None
    line_end: int | None
    domain_id: str
    reality_status: str
    temporal_scope: str
    content_scope: str
    story_unit_id: str
    index_version: str
    retrieval_score: float
    rerank_score: float | None
    layer_label: str  # 叙事层标签（如 第1卷·主线·current·objective）

    _TYPE_LABELS = {"fact": "事实", "relation": "关系", "event": "事件"}

    def context_block_header(self) -> str:
        type_label = self._TYPE_LABELS.get(self.document_type, self.document_type)
        header = f"[{self.citation_key}] 【{type_label}】{self.title}"
        if self.layer_label:
            header += f"（{self.layer_label}）"
        return header

    def to_block(self) -> str:
        """完整证据块文本（packet.context_text 与 prompt 证据区共用）。"""
        lines = [self.context_block_header()]
        if self.summary:
            lines.append(self.summary)
        if self.evidence_text:
            lines.append(f"证据（引用原文）：「{self.evidence_text}」")
        lines.append(
            "来源：{} L{}{}".format(
                self.source_path or "未知文件",
                self.line_start if self.line_start is not None else "?",
                f"-{self.line_end}" if self.line_end else "",
            )
        )
        return "\n".join(line for line in lines if line)


@dataclass
class EvidencePacket:
    """回答模型的唯一知识来源。"""

    query: str
    domain_id: str | None
    answer_mode: AnswerMode
    retrieval_confidence: float
    documents: tuple[EvidenceItem, ...] = ()
    context_text: str = ""  # prompt 用的证据块文本
    citations: tuple[dict[str, Any], ...] = ()  # key → 文档 metadata（服务端权威）
    filters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    evidence_budget: int = 0
    truncated: bool = False
    abstention_reason: str = ""
    query_analysis: dict[str, Any] = field(default_factory=dict)
    index_version: str = ""

    @property
    def citation_keys(self) -> list[str]:
        return [item.citation_key for item in self.documents]

    def key_to_document_id(self) -> dict[str, str]:
        return {item.citation_key: item.document_id for item in self.documents}


@dataclass
class AnswerTimings:
    """链路分阶段耗时（ms，观测用，不建性能平台）。"""

    retrieval_ms: float = 0.0
    packet_ms: float = 0.0
    prompt_ms: float = 0.0
    generation_ms: float = 0.0
    first_token_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_ms": round(self.retrieval_ms, 1),
            "packet_ms": round(self.packet_ms, 1),
            "prompt_ms": round(self.prompt_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "first_token_ms": round(self.first_token_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclass
class GroundedAnswerResult:
    """P7 回答统一结果（非流式与流式最终结构语义一致）。"""

    answer: str
    answer_mode: AnswerMode
    domain_id: str | None
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    abstained: bool = False
    warnings: list[str] = field(default_factory=list)
    failure_kind: str = ""  # FailureKind 值；成功为空
    used_rag: bool = False
    model_invoked: bool = False
    model_id: str = ""
    cache_hit: bool = False
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    timings: AnswerTimings = field(default_factory=AnswerTimings)

    def api_citations(self, debug: bool = False) -> list[dict[str, Any]]:
        """对外 citation 视图：默认不暴露绝对路径与内部调试分数。"""
        from .validator import public_citation_view  # 局部导入避免环

        return [public_citation_view(c, debug=debug) for c in self.citations]


@dataclass
class AnswerStreamEvent:
    """流式事件（SSE payload；正文完成后统一发送 citations）。"""

    type: str  # meta / delta / citations / done / error
    data: dict[str, Any] = field(default_factory=dict)
