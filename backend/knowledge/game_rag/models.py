"""月社妃全游戏文本 RAG 领域数据模型（P1 契约层）。

为 P2 原文解析、P3 场景切分、P4 知识候选提取提供稳定的类型契约：
- ScriptSegment 是 P2 解析器的输出单元（对话/叙述，逐行状态机产物）；
- 五种 DocumentType 模型是后续知识库的存储与检索单元（场景/事实/关系/事件/章节摘要）。

设计原则：
- 全部模型拒绝未知字段（extra="forbid"），避免拼写错误被静默接受；
- 优先组合（StoryContext / SourceSpan 复用），不建立复杂继承体系；
- 不包含 embedding、FAISS、BM25、数据库主键或 API 响应字段。

四个结构维度的区别（详见 docs/research/KISAKI_GAME_RAG_SCHEMA.md）：
- content_scope：内容类型（正篇 / 追加剧本 / 宣传元叙事）；
- temporal_scope：时间状态（当前 / 回忆 / 魔法重现 / 假设）；
- route：文件间结构关系的定性（主线 / 分支 / 平行），仅剧情结构审核后填写；
- continuity_id：连续性分组（不同故事单元是否处于同一连续时间线）。

None 与 unknown 的语义约定：
- route=None 表示尚未审核；RouteType.unknown 表示已审核但无法确定；
- temporal_scope=None 表示尚未审核（P3 草稿默认）；TemporalScope.unknown 表示已审核但无法判断；
- flashback 只属于 TemporalScope，绝不属于 RouteType。
"""

# ruff: noqa: UP042
# 枚举沿用项目现有的 (str, Enum) 惯例（见 inference/vllm_client.py 等），
# 不改用 enum.StrEnum 以保持对 Python 3.10 运行环境的兼容。

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


class SegmentType(str, Enum):
    """P2 解析输出的基本类型：对话或叙述。"""

    dialogue = "dialogue"
    narration = "narration"


class QuoteStyle(str, Enum):
    """台词引号样式（叙述为 none）。"""

    corner = "corner"  # 「」
    double_corner = "double_corner"  # 『』
    curly = "curly"  # ""
    none = "none"


class ContentScope(str, Enum):
    """内容类型维度：这段文本属于哪类内容。"""

    main_story = "main_story"  # 正篇
    bonus_story = "bonus_story"  # 追加剧本（日后谈第 39 行起《萤色光景》）
    promotional_meta = "promotional_meta"  # 宣传元叙事（日后谈第 1-37 行）
    unknown = "unknown"  # 已审核但无法确定；未审核应使用待定流程而非滥用此值


class TemporalScope(str, Enum):
    """时间状态维度：叙述所处的叙事时间层。

    注意：这是场景/事实的时间状态，与 RouteType 无关。
    """

    current = "current"  # 主线当前
    flashback = "flashback"  # 回忆
    reconstruction = "reconstruction"  # 魔法重现
    hypothetical = "hypothetical"  # 假设/想象
    unknown = "unknown"


class RouteType(str, Enum):
    """路线维度：文件间结构关系的定性，仅剧情结构审核后填写。

    刻意不含 flashback：回忆属于 TemporalScope（时间状态维度），
    把时间状态混入路线维度会导致后续过滤混乱。
    """

    main = "main"
    branch = "branch"
    parallel = "parallel"
    unknown = "unknown"  # 已审核但无法确定；尚未审核时字段应为 None


class RealityStatus(str, Enum):
    """事实属性维度：内容的现实/可靠性状态。"""

    objective = "objective"  # 客观事实
    character_claim = "character_claim"  # 人物观点/主张
    inferred = "inferred"  # 推测
    fictional = "fictional"  # 虚构（魔法制造、故事内故事等）
    conflicted = "conflicted"  # 存在冲突
    unknown = "unknown"


class ReviewStatus(str, Enum):
    """人工审核状态。未审核内容不得进入正式索引。"""

    draft = "draft"
    needs_review = "needs_review"
    approved = "approved"
    rejected = "rejected"


class DocumentType(str, Enum):
    """知识库文档类型。"""

    scene = "scene"
    fact = "fact"
    relation = "relation"
    event = "event"
    chapter_summary = "chapter_summary"


def _require_non_empty(value: str) -> str:
    """字符串去空白后不得为空。"""
    if not value or not value.strip():
        raise ValueError("不得为空字符串")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_require_non_empty)]


def _require_non_empty_list(items: list[str]) -> list[str]:
    """列表内每个字符串去空白后不得为空（列表本身允许为空）。"""
    for item in items:
        if not item or not item.strip():
            raise ValueError("列表元素不得为空字符串")
    return items


NonEmptyStrList = Annotated[list[str], AfterValidator(_require_non_empty_list)]


class SourceSpan(BaseModel):
    """原文出处：来源文件与行号范围（行号从 1 开始）。

    本阶段不读取文件、不校验路径是否真实存在。
    """

    model_config = ConfigDict(extra="forbid")

    source_path: NonEmptyStr
    line_start: int = Field(ge=1, description="起始行号（含），从 1 开始")
    line_end: int = Field(ge=1, description="结束行号（含）")

    @model_validator(mode="after")
    def _check_line_order(self) -> SourceSpan:
        if self.line_end < self.line_start:
            raise ValueError(f"line_end({self.line_end}) 不得小于 line_start({self.line_start})")
        return self


class StoryContext(BaseModel):
    """故事结构上下文：四个结构维度的载体。

    不根据文件名自动推断 route 或 continuity——同编号多故事单元
    的连续性关系需剧情结构人工审核后填写。
    """

    model_config = ConfigDict(extra="forbid")

    volume_number: int | None = Field(default=None, ge=1, description="卷号；None 用于《日后谈》等无卷号单元")
    story_unit_id: NonEmptyStr = Field(description="故事单元 id（P1 以文件为初始单元）")
    story_title: NonEmptyStr = Field(description="故事标题（文件名去扩展名）")
    continuity_id: str | None = Field(default=None, description="连续性组 id；None 表示尚未审核")
    sequence_order: int | None = Field(default=None, ge=0, description="连续性组内顺序；审核后填写")
    viewpoint: str | None = Field(default=None, description="视角（如琉璃第一人称），可观察特征")
    content_scope: ContentScope
    temporal_scope: TemporalScope | None = Field(
        default=None,
        description="None=尚未审核；unknown=已审核但无法确定（与 route 的 None/unknown 语义一致）",
    )
    route: RouteType | None = Field(default=None, description="None=尚未审核；unknown=已审核但无法确定")


class ScriptSegment(BaseModel):
    """P2 解析器输出契约：一条对话或一段叙述。

    只表达解析结果，不包含解析逻辑。逐行状态机的告警
    （如未闭合引号截断）通过 warnings 传递，不静默丢弃。
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    segment_type: SegmentType
    text: NonEmptyStr
    source: SourceSpan
    speaker: str | None = Field(default=None, description="说话人标签；仅 dialogue 必填")
    quote_style: QuoteStyle
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_segment_consistency(self) -> ScriptSegment:
        """跨字段一致性：segment_type 与 speaker / quote_style 的组合约束。

        - dialogue：必须有非空 speaker，quote_style 不得为 none
          （未闭合引号的台词仍是 dialogue，保留原始开引号文本并经 warnings 记录）；
        - narration：必须 speaker=None 且 quote_style=none。
        """
        if self.segment_type is SegmentType.dialogue:
            if self.speaker is None or not self.speaker.strip():
                raise ValueError("dialogue 必须提供 speaker")
            if self.quote_style is QuoteStyle.none:
                raise ValueError("dialogue 的 quote_style 不得为 none")
        else:
            if self.speaker is not None:
                raise ValueError("narration 必须不携带 speaker")
            if self.quote_style is not QuoteStyle.none:
                raise ValueError("narration 必须使用 quote_style=none")
        return self


class SceneDocument(BaseModel):
    """场景文档：P3 切分产物，保留完整场景原文与人物在场信息。

    人物缺席状态不存储，由 speakers / mentioned_characters /
    present_characters 在运行时推导。
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    document_type: Literal[DocumentType.scene] = DocumentType.scene
    title: NonEmptyStr
    text: NonEmptyStr = Field(description="场景完整原文（evidence 用）")
    story: StoryContext
    source: SourceSpan
    speakers: NonEmptyStrList = Field(default_factory=list, description="有台词的说话人")
    mentioned_characters: NonEmptyStrList = Field(default_factory=list, description="被提及的人物")
    present_characters: NonEmptyStrList = Field(default_factory=list, description="实际在场的人物")
    reality_status: RealityStatus = RealityStatus.unknown
    review_status: ReviewStatus = ReviewStatus.draft


class FactDocument(BaseModel):
    """事实卡：一条记录只表达一个事实（subject-predicate-value 三元组）。"""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    document_type: Literal[DocumentType.fact] = DocumentType.fact
    title: NonEmptyStr
    subject: NonEmptyStr
    predicate: NonEmptyStr
    value: NonEmptyStr
    summary: NonEmptyStr
    evidence_text: NonEmptyStr = Field(description="原作原文证据，用于回答与引用")
    story: StoryContext
    source: SourceSpan
    reality_status: RealityStatus
    review_status: ReviewStatus


class RelationDocument(BaseModel):
    """关系卡：subject 与 target 的关系及证据。"""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    document_type: Literal[DocumentType.relation] = DocumentType.relation
    title: NonEmptyStr
    subject: NonEmptyStr
    relation: NonEmptyStr
    target: NonEmptyStr
    summary: NonEmptyStr
    evidence_text: NonEmptyStr
    story: StoryContext
    source: SourceSpan
    reality_status: RealityStatus
    review_status: ReviewStatus


class EventDocument(BaseModel):
    """事件卡：起因、过程、结果、参与者。"""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    document_type: Literal[DocumentType.event] = DocumentType.event
    title: NonEmptyStr
    summary: NonEmptyStr
    participants: NonEmptyStrList = Field(default_factory=list)
    causes: NonEmptyStrList = Field(default_factory=list)
    outcomes: NonEmptyStrList = Field(default_factory=list)
    evidence_text: NonEmptyStr
    story: StoryContext
    source: SourceSpan
    reality_status: RealityStatus
    review_status: ReviewStatus


class ChapterSummaryDocument(BaseModel):
    """章节摘要：只服务跨章节全局问题，不能作为唯一事实证据。"""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    document_type: Literal[DocumentType.chapter_summary] = DocumentType.chapter_summary
    title: NonEmptyStr
    summary: NonEmptyStr
    story: StoryContext
    source: SourceSpan
    review_status: ReviewStatus
