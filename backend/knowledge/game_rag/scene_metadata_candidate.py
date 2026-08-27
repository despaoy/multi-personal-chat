"""P4B 场景元数据候选生成与断点续审基础设施。

职责：在 P4A 冻结场景包（FrozenSceneBundle，manifest/scenes/bundle 三摘要绑定）
之上，通过可注入的模型客户端为每个冻结场景生成**待人工审核的元数据候选**
（viewpoint / temporal_scope / reality_status / mentioned_characters /
present_characters / evidence / reasons / warnings），维护可断点续跑的候选
运行状态，并把合法候选安全合并进 P4A 审核文档。

明确不做：不初始化模型服务器、不下载模型、不读取环境密钥（模型客户端由
调用方按 CandidateModelClient 协议注入）；不调用 apply_approved_scene_metadata
或 write_enriched_scenes（候选路径绝不产出 approved/enriched 正式产物）；
不修改游戏原文、SFT、实验数据、通用 RAG、数据库、embedding 或 API。

四类产物的区别（不得混淆）：
- 模型候选（SceneMetadataCandidate）：模型输出，仅供人工参考；
- 人工审核决定（P4A SceneMetadataDecision）：人工填写/复核后的记录；
- approved enriched 输出：P4A apply/write 流程的正式产物，候选路径不可达；
- 运行 manifest（CandidateRunManifest）：唯一允许携带时间戳等非确定性信息
  的产物；候选运行状态与 P4A 审核状态保持确定性（无时间戳/随机数）。

语义（与 P4A 契约一致，不得混用）：
- None = 尚未审核（人工侧初始状态）；unknown = 已审核但无法判断；
  空人物数组 = 已审核且确认无人；
- 候选合并进审核文档后，记录最多置为 needs_review，绝不自动 approved；
- 已有人工字段不被候选覆盖：默认冲突即跳过并报告，
  显式 overwrite 策略也绝不触碰 reviewer / review_status / approved 记录。

详见 docs/research/KISAKI_GAME_RAG_SCENE_METADATA_CANDIDATE.md。
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from knowledge.game_rag.models import (
    NonEmptyStr,
    NonEmptyStrList,
    RealityStatus,
    ReviewStatus,
    SceneDocument,
    SourceSpan,
    TemporalScope,
)
from knowledge.game_rag.scene_metadata_review import (
    FrozenSceneBundle,
    SceneMetadataDecision,
    SceneMetadataReviewDocument,
    SourceManifestRef,
    _atomic_write_pair,
    _atomic_write_text,
    _bundle_integrity_errors,
    _coerce_review_document,
    _normalize_character_names,
    _normalize_viewpoint,
    _summarize_validation_error,
    validate_scene_metadata_review,
)

CANDIDATE_RUN_SCHEMA_VERSION = 1
CANDIDATE_RUN_MANIFEST_SCHEMA_VERSION = 1
GENERATOR_ID = "knowledge.game_rag.scene_metadata_candidate"

# 长场景分片阈值：超过该行数的场景分片生成候选；分片只影响候选生成，
# evidence 与决定归并回原 scene_id，绝不重切 P3 场景。
CANDIDATE_CHUNK_MAX_LINES = 150

# 失败摘要 detail 上限：不得把完整 prompt、模型全文或密钥写进运行状态。
FAILURE_DETAIL_MAX_CHARS = 200

DEFAULT_RUN_NOTES = (
    "P4B 场景元数据候选运行状态。候选由模型生成，仅供人工审核参考；"
    "合并进 P4A 审核文档后记录最多为 needs_review，绝不自动 approved。"
    "运行状态绑定冻结包三摘要（manifest_sha256/scenes_sha256/bundle_sha256），"
    "跨 bundle 恢复或合并会被拒绝。scene_id / source / story_unit_id 创建后不得更换。"
)

DEFAULT_MANIFEST_NOTE = (
    "P4B candidate run manifest: 记录模型标识、参数、输入 bundle 摘要与计数；"
    "时间戳等非确定性信息只记录在本 manifest，不进入候选运行状态或 P4A 审核状态。"
)

DEFAULT_GENERATION_PARAMS: dict[str, int | float | str | bool] = {
    "chunk_max_lines": CANDIDATE_CHUNK_MAX_LINES,
}

# 候选合并涉及的审核记录字段（reviewer / review_status 永不由候选改写）。
CANDIDATE_MERGE_FIELDS = (
    "viewpoint",
    "temporal_scope",
    "reality_status",
    "mentioned_characters",
    "present_characters",
    "evidence",
    "reasons",
    "warnings",
)


class CandidateParseError(ValueError):
    """模型输出解析失败（markdown fence / JSON 非法 / 契约违反 / evidence 越界）。

    error_kind 用于失败摘要分类；异常消息不含 prompt 内容。
    """

    def __init__(self, message: str, *, error_kind: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


@runtime_checkable
class CandidateModelClient(Protocol):
    """模型客户端最小协议：输入 prompt 文本，返回模型原始输出文本。

    领域模块不初始化服务器、不下载模型、不读取环境密钥；
    超时/连接错误由客户端以异常形式抛出（TimeoutError 或其他异常），
    本模块负责把异常收敛为可重试的失败摘要。
    """

    def __call__(self, prompt: str) -> str: ...


# ---------- 提示契约 ----------


class CategoryHint(BaseModel):
    """提示契约条目：一类叙事现象 → 候选字段倾向。

    只描述类别定义与判断规则，不预设任何真实语料的结论；
    最终判断一律以人工审核为准。
    """

    model_config = ConfigDict(extra="forbid")

    label: NonEmptyStr
    description: NonEmptyStr
    temporal_scope_hints: list[TemporalScope] = Field(min_length=1)
    reality_status_hints: list[RealityStatus] = Field(min_length=1)


DEFAULT_HINT_CONTRACT: tuple[CategoryHint, ...] = (
    CategoryHint(
        label="梦境",
        description="人物睡眠中的梦境叙述，不属于现实时间线上发生的事件",
        temporal_scope_hints=[TemporalScope.hypothetical],
        reality_status_hints=[RealityStatus.fictional],
    ),
    CategoryHint(
        label="回忆",
        description="对过去事件的回顾叙述；事件本身可能真实，也可能带叙述者主观色彩",
        temporal_scope_hints=[TemporalScope.flashback],
        reality_status_hints=[RealityStatus.objective, RealityStatus.character_claim],
    ),
    CategoryHint(
        label="书中故事",
        description="剧情内人物讲述、阅读或演绎的虚构故事（故事内故事）",
        temporal_scope_hints=[TemporalScope.hypothetical],
        reality_status_hints=[RealityStatus.fictional],
    ),
    CategoryHint(
        label="宣传元叙事",
        description="面向读者的宣传性、推广性叙述，不是剧情内人物经历的事实",
        temporal_scope_hints=[TemporalScope.unknown],
        reality_status_hints=[RealityStatus.fictional],
    ),
    CategoryHint(
        label="魔法重现",
        description="以魔法手段重现的人物、场景或事件",
        temporal_scope_hints=[TemporalScope.reconstruction],
        reality_status_hints=[RealityStatus.fictional],
    ),
    CategoryHint(
        label="无法判断",
        description="证据不足或现象混合，无法归入上述类别",
        temporal_scope_hints=[TemporalScope.unknown],
        reality_status_hints=[RealityStatus.unknown],
    ),
)


# ---------- 候选模型与运行状态 ----------


class SceneMetadataCandidate(BaseModel):
    """单个场景的模型候选（非人工决定；合并后最多 needs_review）。

    - evidence 的 source_path 由解析器从场景补齐（模型只输出行号范围），
      因此 evidence 与场景 source_path 一致由构造保证；
    - 人物数组与 viewpoint 沿用 P4A 规范：去空白、去重、稳定排序，
      viewpoint 须为「<人物名>第一人称」/「第三人称」/「多视角」/「unknown」。
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: NonEmptyStr
    viewpoint: NonEmptyStr
    temporal_scope: TemporalScope
    reality_status: RealityStatus
    mentioned_characters: list[str]
    present_characters: list[str]
    evidence: list[SourceSpan] = Field(min_length=1)
    reasons: NonEmptyStrList = Field(min_length=1)
    warnings: NonEmptyStrList = Field(default_factory=list)

    @field_validator("viewpoint")
    @classmethod
    def _validate_viewpoint(cls, value: str) -> str:
        return _normalize_viewpoint(value)

    @field_validator("mentioned_characters", "present_characters")
    @classmethod
    def _validate_characters(cls, value: list[str]) -> list[str]:
        return _normalize_character_names(value)


class FailureSummary(BaseModel):
    """失败摘要：只存分类与截断 detail，不存 prompt、模型全文或密钥。"""

    model_config = ConfigDict(extra="forbid")

    error_kind: NonEmptyStr
    detail: str = ""
    attempts: int = Field(ge=1)


class CandidateGenerationStatus(str, Enum):  # noqa: UP042  # 沿用项目 (str, Enum) 惯例（Python 3.10 兼容）
    """候选生成状态（运行状态层；与人工 ReviewStatus 语义无关）。"""

    pending = "pending"
    success = "success"
    failed = "failed"


class RerunRecord(BaseModel):
    """显式重跑审计记录（确定性：不含时间戳）。"""

    model_config = ConfigDict(extra="forbid")

    rerun_reason: NonEmptyStr
    previous_status: CandidateGenerationStatus
    outcome: Literal["success", "failed"]


class SceneCandidateState(BaseModel):
    """单个场景的候选生成状态。

    - status=success 必须携带候选；status=failed 必须携带失败摘要；
    - last_failure 表示该场景**最近一次**生成尝试的失败信息：
      生成成功即清空；显式重跑失败时保留（status 保持 success，旧候选不损坏）；
    - attempts 为累计模型调用次数（含分片调用与成功调用）；
    - rerun_history 记录全部显式重跑（覆盖成功结果必须先留原因）。
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: NonEmptyStr
    story_unit_id: NonEmptyStr
    source: SourceSpan
    status: CandidateGenerationStatus = CandidateGenerationStatus.pending
    candidate: SceneMetadataCandidate | None = None
    attempts: int = Field(default=0, ge=0)
    last_failure: FailureSummary | None = None
    rerun_history: list[RerunRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> SceneCandidateState:
        if self.status is CandidateGenerationStatus.success and self.candidate is None:
            raise ValueError("status=success 必须携带候选结果")
        if self.status is not CandidateGenerationStatus.success and self.candidate is not None:
            raise ValueError("只有 status=success 才能携带候选结果")
        if self.status is CandidateGenerationStatus.failed and self.last_failure is None:
            raise ValueError("status=failed 必须携带失败摘要")
        if self.candidate is not None and self.candidate.scene_id != self.scene_id:
            raise ValueError(f"候选 scene_id {self.candidate.scene_id!r} 与记录 scene_id {self.scene_id!r} 不一致")
        return self


class CandidateRunState(BaseModel):
    """P4B 候选生成运行状态（版本化顶层文档）。

    确定性：不含时间戳/随机 UUID；同一 bundle 连续创建两次结果完全一致。
    绑定冻结包三摘要：跨 bundle 恢复或合并由 validate_candidate_run 拒绝。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source_manifest: SourceManifestRef
    total_source_scenes: int = Field(ge=1)
    model_id: NonEmptyStr
    generation_params: dict[str, int | float | str | bool] = Field(default_factory=dict)
    scene_states: list[SceneCandidateState] = Field(default_factory=list)
    notes: str = DEFAULT_RUN_NOTES
    created_by: NonEmptyStr = GENERATOR_ID

    @model_validator(mode="after")
    def _check_scene_states(self) -> CandidateRunState:
        if len(self.scene_states) != self.total_source_scenes:
            raise ValueError(
                f"scene_states 数量 {len(self.scene_states)} 与 total_source_scenes "
                f"{self.total_source_scenes} 不一致（每个输入场景必须恰好一条状态）"
            )
        counts = Counter(state.scene_id for state in self.scene_states)
        if duplicates := sorted(scene_id for scene_id, n in counts.items() if n > 1):
            raise ValueError(f"scene_states 存在重复 scene_id: {duplicates}")
        chunk_max = self.generation_params.get("chunk_max_lines")
        if isinstance(chunk_max, bool) or not isinstance(chunk_max, int) or chunk_max < 1:
            raise ValueError("generation_params.chunk_max_lines 必须为 >= 1 的整数")
        return self


class CandidateRunResult(BaseModel):
    """一次 generate_scene_candidates 调用的结果（含更新后的运行状态）。"""

    model_config = ConfigDict(extra="forbid")

    new_state: CandidateRunState
    attempted_scene_ids: list[str]
    succeeded_scene_ids: list[str]
    failed_scene_ids: list[str]
    skipped_scene_ids: list[str]

    @model_validator(mode="after")
    def _check_result_partition(self) -> CandidateRunResult:
        groups = {
            "attempted_scene_ids": self.attempted_scene_ids,
            "succeeded_scene_ids": self.succeeded_scene_ids,
            "failed_scene_ids": self.failed_scene_ids,
            "skipped_scene_ids": self.skipped_scene_ids,
        }
        for name, values in groups.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name} 不得包含重复 scene id")
        outcomes = self.succeeded_scene_ids + self.failed_scene_ids + self.skipped_scene_ids
        if len(outcomes) != len(set(outcomes)):
            raise ValueError("succeeded/failed/skipped 三组 scene id 必须互斥")
        if set(outcomes) != set(self.attempted_scene_ids):
            raise ValueError("attempted_scene_ids 必须恰好等于 succeeded/failed/skipped 的并集")
        return self


# ---------- 创建与校验 ----------


def create_candidate_run(
    bundle: FrozenSceneBundle,
    *,
    model_id: str,
    generation_params: dict[str, int | float | str | bool] | None = None,
) -> CandidateRunState:
    """从冻结场景包创建初始候选运行状态：全部 pending、无候选、零尝试。

    - 不修改输入 bundle；记录按冻结场景顺序创建，scene_id / source /
      story_unit_id 原样复制；
    - 绑定冻结包三摘要（manifest_sha256 / scenes_sha256 / bundle_sha256）；
    - model_id 只是模型标识字符串，不得包含密钥；
    - generation_params 是调用方提供的参数快照（确定性标量），
      与模块内置默认值合并（默认值可被覆盖）；
    - 同一 bundle 连续创建两次结果完全一致（无时间戳/随机数）。
    """
    if integrity_errors := _bundle_integrity_errors(bundle):
        raise ValueError("冻结场景包完整性校验失败:\n- " + "\n- ".join(integrity_errors))
    model_id_value = str(model_id or "").strip()
    if not model_id_value:
        raise ValueError("model_id 不得为空")
    params: dict[str, int | float | str | bool] = dict(DEFAULT_GENERATION_PARAMS)
    if generation_params is not None:
        params.update(generation_params)
    scene_states = [
        SceneCandidateState(
            scene_id=scene.id,
            story_unit_id=scene.story.story_unit_id,
            source=scene.source.model_copy(deep=True),
        )
        for scene in bundle.scenes
    ]
    return CandidateRunState(
        schema_version=CANDIDATE_RUN_SCHEMA_VERSION,
        source_manifest=SourceManifestRef(
            schema_version=bundle.manifest.schema_version,
            boundary_review_status=bundle.manifest.boundary_review_status,
            reviewer=bundle.manifest.reviewer,
            total_scenes=bundle.manifest.total_scenes,
            manifest_sha256=bundle.manifest_digest,
            scenes_sha256=bundle.scenes_digest,
            bundle_sha256=bundle.bundle_digest,
        ),
        total_source_scenes=len(bundle.scenes),
        model_id=model_id_value,
        generation_params=params,
        scene_states=scene_states,
    )


def _coerce_run_state(
    run_state: CandidateRunState | dict[str, Any],
) -> tuple[CandidateRunState | None, list[str]]:
    if isinstance(run_state, CandidateRunState):
        try:
            return CandidateRunState.model_validate(run_state.model_dump(mode="json")), []
        except ValidationError as exc:
            return None, [f"候选运行状态结构非法: {_summarize_validation_error(exc)}"]
    if isinstance(run_state, dict):
        try:
            return CandidateRunState.model_validate(run_state), []
        except ValidationError as exc:
            return None, [f"候选运行状态结构非法: {_summarize_validation_error(exc)}"]
    return None, [f"候选运行状态类型非法: {type(run_state).__name__}（须为 CandidateRunState 或 dict）"]


def validate_candidate_run(
    run_state: CandidateRunState | dict[str, Any],
    bundle: FrozenSceneBundle,
) -> list[str]:
    """校验候选运行状态与冻结场景包的跨字段一致性，返回错误列表（空=通过）。

    - schema_version 必须为当前版本（不静默迁移旧版本）；
    - source_manifest 与 bundle 逐字段一致（三摘要钉住冻结双文件及组合）；
    - 场景集合完整：不缺失、无未知、无重复（重复由文档模型自身拒绝）；
    - scene_id / source / story_unit_id 不可篡改；
    - 已存候选的 evidence 必须与场景 source_path 一致且落在场景 span 内
      （防加载后的文件级篡改）。
    """
    doc, errors = _coerce_run_state(run_state)
    if doc is None:
        return errors

    errors.extend(_bundle_integrity_errors(bundle))

    if doc.schema_version != CANDIDATE_RUN_SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {CANDIDATE_RUN_SCHEMA_VERSION}，当前为 {doc.schema_version!r}")

    ref = doc.source_manifest
    if ref.manifest_sha256 != bundle.manifest_digest:
        errors.append(
            "source_manifest.manifest_sha256 与 bundle 不一致：候选运行状态不是针对当前冻结场景包创建的"
            f"（{ref.manifest_sha256[:12]}… != {bundle.manifest_digest[:12]}…）"
        )
    if ref.scenes_sha256 != bundle.scenes_digest:
        errors.append(
            "source_manifest.scenes_sha256 与 bundle 不一致：候选运行状态不是针对当前 scenes.jsonl 创建的"
            f"（{ref.scenes_sha256[:12]}… != {bundle.scenes_digest[:12]}…）"
        )
    if ref.bundle_sha256 != bundle.bundle_digest:
        errors.append(
            "source_manifest.bundle_sha256 与 bundle 不一致：冻结双文件组合已变化"
            f"（{ref.bundle_sha256[:12]}… != {bundle.bundle_digest[:12]}…）"
        )
    if ref.schema_version != bundle.manifest.schema_version:
        errors.append(f"source_manifest.schema_version 与 bundle.manifest 不一致: {ref.schema_version!r}")
    if ref.boundary_review_status != bundle.manifest.boundary_review_status:
        errors.append(
            f"source_manifest.boundary_review_status 与 bundle.manifest 不一致: {ref.boundary_review_status!r}"
        )
    if ref.reviewer != bundle.manifest.reviewer:
        errors.append(f"source_manifest.reviewer 与 bundle.manifest 不一致: {ref.reviewer!r}")
    if ref.total_scenes != bundle.total_scenes:
        errors.append(
            f"source_manifest.total_scenes 与 bundle 场景数不一致: {ref.total_scenes} != {bundle.total_scenes}"
        )
    if doc.total_source_scenes != bundle.total_scenes:
        errors.append(f"total_source_scenes 与 bundle 场景数不一致: {doc.total_source_scenes} != {bundle.total_scenes}")

    bundle_by_id = {scene.id: scene for scene in bundle.scenes}
    states_by_id = {state.scene_id: state for state in doc.scene_states}
    if unknown := sorted(set(states_by_id) - set(bundle_by_id)):
        errors.append(f"scene_states 含未知 scene: {unknown}")
    if missing := sorted(set(bundle_by_id) - set(states_by_id)):
        errors.append(f"scene_states 缺少 scene: {missing}")

    for scene_id in sorted(set(states_by_id) & set(bundle_by_id)):
        state = states_by_id[scene_id]
        scene = bundle_by_id[scene_id]
        if state.story_unit_id != scene.story.story_unit_id:
            errors.append(
                f"{scene_id}: story_unit_id 被篡改（{state.story_unit_id!r} != {scene.story.story_unit_id!r}）"
            )
        if state.source != scene.source:
            errors.append(
                f"{scene_id}: source 被篡改（L{state.source.line_start}-{state.source.line_end} != "
                f"L{scene.source.line_start}-{scene.source.line_end}）"
            )
        if state.candidate is not None:
            for span in state.candidate.evidence:
                if span.source_path != scene.source.source_path:
                    errors.append(
                        f"{scene_id}: 候选 evidence 出处 {span.source_path!r} 与场景 source "
                        f"{scene.source.source_path!r} 不一致"
                    )
                if span.line_start < scene.source.line_start or span.line_end > scene.source.line_end:
                    errors.append(
                        f"{scene_id}: 候选 evidence L{span.line_start}-{span.line_end} 超出场景范围 "
                        f"L{scene.source.line_start}-{scene.source.line_end}"
                    )
    return errors


# ---------- 原子保存与读取 ----------


def save_candidate_run(path: Path | str, run_state: CandidateRunState | dict[str, Any]) -> None:
    """原子保存候选运行状态；写出前先做结构校验（含一致性复查），拒绝写出非法状态。"""
    doc, errors = _coerce_run_state(run_state)
    if doc is None:
        raise ValueError("候选运行状态非法，拒绝写出:\n- " + "\n- ".join(errors))
    # 状态在运行期会被原地更新，落盘前经 model_dump → model_validate 复查全部约束。
    revalidated = CandidateRunState.model_validate(doc.model_dump(mode="json"))
    payload = json.dumps(revalidated.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(Path(path), payload)


def load_candidate_run(path: Path | str) -> CandidateRunState:
    """读取候选运行状态文件（JSON → CandidateRunState）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return CandidateRunState.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"候选运行状态文件结构非法: {_summarize_validation_error(exc)}") from exc


# ---------- 场景选择 ----------


def select_candidate_scenes(
    run_state: CandidateRunState,
    *,
    scene_ids: list[str] | tuple[str, ...] | None = None,
    retry_failed: bool = True,
) -> list[str]:
    """按冻结场景顺序返回待处理 scene id 列表。

    - scene_ids=None：全部 pending 场景，外加 retry_failed=True 时的 failed 场景
      （失败重试不需要显式原因，因为不会覆盖任何成功结果）；
    - scene_ids 给定：仅指定场景（仍按 scene_states 记录顺序，即冻结顺序），
      不论状态——已成功场景由 generate_scene_candidates 按 rerun 规则处理；
    - 指定不存在的 scene id 直接拒绝。
    """
    states_by_id = {state.scene_id: state for state in run_state.scene_states}
    if scene_ids is None:
        return [
            state.scene_id
            for state in run_state.scene_states
            if state.status is CandidateGenerationStatus.pending
            or (retry_failed and state.status is CandidateGenerationStatus.failed)
        ]
    requested = list(scene_ids)
    if unknown := sorted({scene_id for scene_id in requested if scene_id not in states_by_id}):
        raise ValueError(f"指定的 scene id 不在候选运行状态中: {unknown}")
    wanted = set(requested)
    return [state.scene_id for state in run_state.scene_states if state.scene_id in wanted]


# ---------- prompt 构建与严格解析 ----------


def build_candidate_prompt(
    scene: SceneDocument,
    *,
    span: SourceSpan | None = None,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
    hint_contract: tuple[CategoryHint, ...] | list[CategoryHint] = DEFAULT_HINT_CONTRACT,
) -> str:
    """为冻结场景（或其分片）构建确定性候选生成 prompt。

    - 行号使用绝对行号（相对 source span），evidence 允许范围显式给出；
    - 分片 prompt 声明片段序号，并要求模型仅基于本片段判断；
    - 提示契约只给出类别定义与字段倾向，不预设真实语料结论；
    - text 行数与 span 行数不一致时拒绝（数据异常，不猜测行号映射）。
    """
    target = span if span is not None else scene.source
    if target.source_path != scene.source.source_path:
        raise ValueError("分片 source_path 必须与场景一致")
    if target.line_start < scene.source.line_start or target.line_end > scene.source.line_end:
        raise ValueError(
            f"分片范围 L{target.line_start}-{target.line_end} 必须落在场景 span "
            f"L{scene.source.line_start}-{scene.source.line_end} 内"
        )
    text_lines = scene.text.split("\n")
    span_lines = scene.source.line_end - scene.source.line_start + 1
    if len(text_lines) != span_lines:
        raise ValueError(f"scene {scene.id}: text 行数 {len(text_lines)} 与 span 行数 {span_lines} 不一致")
    offset = target.line_start - scene.source.line_start
    body = text_lines[offset : offset + (target.line_end - target.line_start + 1)]

    lines: list[str] = [
        "你是游戏文本场景元数据审核助手。请阅读以下冻结场景（或分片），为人工审核生成候选元数据。",
        "候选仅供人工复核，不是最终决定。",
        "",
        "## 场景信息",
        f"- scene_id: {scene.id}",
        f"- story_unit_id: {scene.story.story_unit_id}",
        f"- story_title: {scene.story.story_title}",
        f"- span: {scene.source.source_path} L{scene.source.line_start}-{scene.source.line_end}",
    ]
    if total_chunks is not None and total_chunks > 1:
        lines.append(
            f"- 本片段为场景的第 {chunk_index}/{total_chunks} 片（L{target.line_start}-{target.line_end}）；"
            "请仅基于本片段判断，分片候选由调用方归并回原 scene_id"
        )
    lines.extend(
        [
            f"- speakers（仅供参考，不等于在场人物）: {'、'.join(scene.speakers) or '（无）'}",
            "",
            "## 原文（行号为绝对行号）",
        ]
    )
    lines.extend(f"L{target.line_start + index}: {line}" for index, line in enumerate(body))
    lines.extend(
        [
            "",
            "## 输出要求",
            "只输出一个 JSON 对象；不要 markdown 代码围栏（```），不要输出 JSON 以外的任何文本。",
            f'- scene_id: 字符串，必须等于 "{scene.id}"',
            '- viewpoint: "<人物名>第一人称"（第一人称叙述必须写明叙述者人物名，如 "琉璃第一人称"；'
            '无法判断叙述者是谁则填 "unknown"）/ "第三人称" / "多视角" / "unknown" 之一',
            "  注意：以「我」等第一人称口吻叙述的场景属于第一人称，不要误标为第三人称",
            '- temporal_scope: "current" / "flashback" / "reconstruction" / "hypothetical" / "unknown" 之一',
            '- reality_status: "objective" / "character_claim" / "inferred" / "fictional" / "conflicted" / "unknown" 之一',
            "- mentioned_characters: 被提及人物数组（去重排序；已判断且确认无则 []）",
            "- present_characters: 当前叙事层实际在场人物数组（去重排序；已判断且确认无则 []）",
            "- evidence: 行号范围数组（如 "
            f'{{"line_start": {target.line_start}, "line_end": {target.line_start}}}），'
            f"至少一条，必须落在 L{target.line_start}-L{target.line_end} 内",
            "- reasons: 判断理由数组，至少一条",
            "- warnings: 告警数组（可为 []）",
            "",
            "## 判断提示契约（仅供候选参考，最终以人工审核为准）",
        ]
    )
    for hint in hint_contract:
        temporal = "/".join(value.value for value in hint.temporal_scope_hints)
        reality = "/".join(value.value for value in hint.reality_status_hints)
        lines.append(
            f"- {hint.label}: {hint.description} → temporal_scope 倾向 {temporal}；reality_status 倾向 {reality}"
        )
    lines.extend(
        [
            "",
            "## 人物集合提醒",
            "- speakers ≠ 在场人物：书中故事、回忆、梦境、转述中的台词不属于当前叙事层在场人物。",
            "- mentioned 是文本中被谈到的人物；present 是当前叙事层实际在场的人物。",
            "- 第一人称叙述者（「我」「妾」等）本身属于当前叙事层在场人物，应计入 present_characters。",
        ]
    )
    return "\n".join(lines)


class _PayloadEvidenceSpan(BaseModel):
    """模型输出的 evidence 行号范围（source_path 由解析器从场景补齐）。"""

    model_config = ConfigDict(extra="forbid")

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_line_order(self) -> _PayloadEvidenceSpan:
        if self.line_end < self.line_start:
            raise ValueError(f"line_end({self.line_end}) 不得小于 line_start({self.line_start})")
        return self


class _CandidatePayload(BaseModel):
    """模型输出 JSON 的严格契约（extra=forbid：额外字段直接拒绝）。"""

    model_config = ConfigDict(extra="forbid")

    scene_id: NonEmptyStr
    viewpoint: NonEmptyStr
    temporal_scope: TemporalScope
    reality_status: RealityStatus
    mentioned_characters: list[str]
    present_characters: list[str]
    evidence: list[_PayloadEvidenceSpan] = Field(min_length=1)
    reasons: list[str] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


def parse_scene_candidate(
    raw: str,
    scene: SceneDocument,
    *,
    span: SourceSpan | None = None,
) -> SceneMetadataCandidate:
    """严格解析模型 JSON 输出为场景候选。

    拒绝：markdown fence、JSON 以外文本、非对象 JSON、额外字段、缺失必填字段、
    非法枚举、非法 viewpoint、空白人物名、scene_id 不匹配、空 evidence/reasons、
    evidence 越界（相对场景 span；分片调用时相对分片 span）。
    """
    if not isinstance(raw, str):
        raise CandidateParseError(f"模型输出必须是字符串，得到 {type(raw).__name__}", error_kind="invalid_output")
    text = raw.strip()
    if text.startswith("```"):
        raise CandidateParseError("模型输出包含 markdown 代码围栏（```），要求纯 JSON", error_kind="markdown_fence")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateParseError(f"模型 JSON 含重复键: {key!r}", error_kind="duplicate_json_key")
            result[key] = value
        return result

    try:
        data = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CandidateParseError(f"模型输出不是合法 JSON: {exc}", error_kind="invalid_json") from exc
    if not isinstance(data, dict):
        raise CandidateParseError(f"模型输出必须是 JSON 对象，得到 {type(data).__name__}", error_kind="invalid_json")
    try:
        payload = _CandidatePayload.model_validate(data)
    except ValidationError as exc:
        raise CandidateParseError(
            f"模型输出违反候选契约: {_summarize_validation_error(exc)}", error_kind="schema_violation"
        ) from exc
    if payload.scene_id != scene.id:
        raise CandidateParseError(
            f"模型输出 scene_id {payload.scene_id!r} 与请求场景 {scene.id!r} 不一致",
            error_kind="scene_id_mismatch",
        )
    allowed = span if span is not None else scene.source
    evidence: list[SourceSpan] = []
    for item in payload.evidence:
        if item.line_start < allowed.line_start or item.line_end > allowed.line_end:
            raise CandidateParseError(
                f"evidence L{item.line_start}-{item.line_end} 超出允许范围 L{allowed.line_start}-{allowed.line_end}",
                error_kind="evidence_out_of_range",
            )
        evidence.append(
            SourceSpan(source_path=scene.source.source_path, line_start=item.line_start, line_end=item.line_end)
        )
    try:
        return SceneMetadataCandidate(
            scene_id=payload.scene_id,
            viewpoint=payload.viewpoint,
            temporal_scope=payload.temporal_scope,
            reality_status=payload.reality_status,
            mentioned_characters=payload.mentioned_characters,
            present_characters=payload.present_characters,
            evidence=evidence,
            reasons=payload.reasons,
            warnings=payload.warnings,
        )
    except ValidationError as exc:
        raise CandidateParseError(
            f"模型输出违反候选契约: {_summarize_validation_error(exc)}", error_kind="schema_violation"
        ) from exc


# ---------- 候选生成（含分片归并） ----------


def _truncate_detail(detail: str) -> str:
    """压缩空白并截断失败 detail，防止多行 prompt/长响应进入运行状态。"""
    text = " ".join(str(detail or "").split())
    if len(text) <= FAILURE_DETAIL_MAX_CHARS:
        return text
    return text[:FAILURE_DETAIL_MAX_CHARS] + "…"


def _failure_summary(error_kind: str, detail: str, attempts: int) -> FailureSummary:
    return FailureSummary(error_kind=error_kind, detail=_truncate_detail(detail), attempts=attempts)


def _chunk_spans(scene: SceneDocument, *, chunk_max_lines: int) -> list[SourceSpan]:
    """把长场景切为连续分片（仅用于候选生成，不重切 P3 场景）。"""
    total_lines = scene.source.line_end - scene.source.line_start + 1
    if total_lines <= chunk_max_lines:
        return [scene.source]
    spans: list[SourceSpan] = []
    start = scene.source.line_start
    while start <= scene.source.line_end:
        end = min(start + chunk_max_lines - 1, scene.source.line_end)
        spans.append(SourceSpan(source_path=scene.source.source_path, line_start=start, line_end=end))
        start = end + 1
    return spans


def _merge_chunk_candidates(
    scene: SceneDocument,
    chunk_candidates: list[SceneMetadataCandidate],
) -> SceneMetadataCandidate:
    """把分片候选归并回原 scene_id。

    - 分类字段（viewpoint / temporal_scope / reality_status）：分片意见一致取该值，
      不一致取 unknown 并追加告警（ disagreement 交由人工复核，不擅自择优）；
    - 人物数组取并集（沿用 P4A 规范化：去空白、去重、稳定排序）；
    - evidence 取并集（各分片 evidence 已被限制在分片 span 内，并集必在场景内）；
    - reasons / warnings 按分片顺序拼接（reasons 去重保持顺序）。
    """
    viewpoints = [candidate.viewpoint for candidate in chunk_candidates]
    temporals = [candidate.temporal_scope for candidate in chunk_candidates]
    realities = [candidate.reality_status for candidate in chunk_candidates]
    warnings: list[str] = []
    viewpoint = viewpoints[0] if len(set(viewpoints)) == 1 else "unknown"
    if len(set(viewpoints)) > 1:
        warnings.append(f"分片 viewpoint 意见不一致: {'/'.join(sorted(set(viewpoints)))}")
    temporal = temporals[0] if len(set(temporals)) == 1 else TemporalScope.unknown
    if len(set(temporals)) > 1:
        warnings.append(f"分片 temporal_scope 意见不一致: {'/'.join(sorted({v.value for v in temporals}))}")
    reality = realities[0] if len(set(realities)) == 1 else RealityStatus.unknown
    if len(set(realities)) > 1:
        warnings.append(f"分片 reality_status 意见不一致: {'/'.join(sorted({v.value for v in realities}))}")
    mentioned = _normalize_character_names(
        [name for candidate in chunk_candidates for name in candidate.mentioned_characters]
    )
    present = _normalize_character_names(
        [name for candidate in chunk_candidates for name in candidate.present_characters]
    )
    evidence = [span for candidate in chunk_candidates for span in candidate.evidence]
    reasons = list(dict.fromkeys(reason for candidate in chunk_candidates for reason in candidate.reasons))
    warnings.extend(warning for candidate in chunk_candidates for warning in candidate.warnings)
    return SceneMetadataCandidate(
        scene_id=scene.id,
        viewpoint=viewpoint,
        temporal_scope=temporal,
        reality_status=reality,
        mentioned_characters=mentioned or [],
        present_characters=present or [],
        evidence=evidence,
        reasons=reasons,
        warnings=warnings,
    )


def _generate_for_scene(
    scene: SceneDocument,
    model_client: CandidateModelClient,
    *,
    max_attempts: int,
    chunk_max_lines: int,
    hint_contract: tuple[CategoryHint, ...] | list[CategoryHint],
) -> tuple[SceneMetadataCandidate | None, FailureSummary | None, int]:
    """为单个场景生成候选：分片 → 逐片重试 → 归并。

    返回 (候选, 失败摘要, 模型调用次数)；候选与失败摘要至多一个非 None。
    """
    spans = _chunk_spans(scene, chunk_max_lines=chunk_max_lines)
    chunk_candidates: list[SceneMetadataCandidate] = []
    attempts = 0
    failure: FailureSummary | None = None
    for index, span in enumerate(spans, start=1):
        chunked = len(spans) > 1
        prompt = build_candidate_prompt(
            scene,
            span=span if chunked else None,
            chunk_index=index if chunked else None,
            total_chunks=len(spans) if chunked else None,
            hint_contract=hint_contract,
        )
        for _ in range(max_attempts):
            attempts += 1
            try:
                raw = model_client(prompt)
                candidate = parse_scene_candidate(raw, scene, span=span)
            except TimeoutError as exc:
                # 异常消息可能携带 URL（如 httpx 超时）：只落盘类型名，原文不进运行状态。
                failure = _failure_summary("timeout", type(exc).__name__, attempts)
                continue
            except CandidateParseError as exc:
                # 自定义校验消息可能包含模型提供的非法值；持久化时仅保留错误分类。
                failure = _failure_summary(exc.error_kind, type(exc).__name__, attempts)
                continue
            except Exception as exc:  # 模型客户端任意异常都收敛为可重试失败
                # 异常消息可能携带密钥、URL 或回显 prompt：只落盘类型名，原文由调用方日志处理。
                failure = _failure_summary("model_error", type(exc).__name__, attempts)
                continue
            chunk_candidates.append(candidate)
            break
        else:
            # 单个分片重试耗尽：整场景判失败，已成功分片的结果不产出（无完整归并）。
            return None, failure, attempts
    final = chunk_candidates[0] if len(chunk_candidates) == 1 else _merge_chunk_candidates(scene, chunk_candidates)
    return final, None, attempts


def generate_scene_candidates(
    bundle: FrozenSceneBundle,
    run_state: CandidateRunState | dict[str, Any],
    model_client: CandidateModelClient,
    *,
    scene_ids: list[str] | tuple[str, ...] | None = None,
    rerun_reasons: dict[str, str] | None = None,
    max_attempts: int = 3,
    state_path: Path | str | None = None,
    hint_contract: tuple[CategoryHint, ...] | list[CategoryHint] = DEFAULT_HINT_CONTRACT,
) -> CandidateRunResult:
    """对选中场景生成候选（断点可续：提供 state_path 时逐场景原子保存进度）。

    门禁（任何模型调用之前完成，任一失败抛 ValueError 且零模型调用）：
    - bundle 完整性与三摘要绑定校验（跨 bundle 的运行状态一律拒绝）；
    - 运行状态结构/一致性校验；
    - rerun_reasons 合法性：只允许指向已成功场景、必须在本次选择范围内、原因非空白。

    处理规则：
    - 按冻结场景顺序处理选中场景；
    - 已成功场景无显式重跑原因 → 跳过（skipped，成功结果不被无意覆盖）；
    - 显式重跑：记录 RerunRecord（原因 + 前状态 + 结果）；重跑失败时保留旧候选
      与 success 状态（不损坏旧结果），仅更新 last_failure；
    - 单场景（分片）最多 max_attempts 次尝试，耗尽置 failed 并保留失败摘要；
    - 失败不影响其他场景；state_path 提供时每个已处理场景之后原子保存一次。
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts 必须 >= 1，当前为 {max_attempts}")
    state, coerce_errors = _coerce_run_state(run_state)
    if state is None:
        raise ValueError("候选运行状态非法:\n- " + "\n- ".join(coerce_errors))
    errors = validate_candidate_run(state, bundle)
    if errors:
        raise ValueError("候选运行状态未通过校验（拒绝在任何模型调用之前）:\n- " + "\n- ".join(errors))

    selected = select_candidate_scenes(state, scene_ids=scene_ids)
    selected_set = set(selected)
    reasons = {str(key): str(value) for key, value in (rerun_reasons or {}).items()}
    if unknown_reasons := sorted(set(reasons) - selected_set):
        raise ValueError(f"rerun_reasons 指向未选中的场景: {unknown_reasons}")
    states_by_id = {item.scene_id: item for item in state.scene_states}
    if invalid_rerun := sorted(
        scene_id
        for scene_id, _ in reasons.items()
        if states_by_id[scene_id].status is not CandidateGenerationStatus.success
    ):
        raise ValueError(f"rerun 原因只适用于已成功场景，以下场景尚未成功: {invalid_rerun}")
    if blank_reasons := sorted(scene_id for scene_id, value in reasons.items() if not value.strip()):
        raise ValueError(f"rerun 原因不得为空白字符串: {blank_reasons}")

    new_state = state.model_copy(deep=True)
    new_states_by_id = {item.scene_id: item for item in new_state.scene_states}
    attempted: list[str] = []
    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for scene in bundle.scenes:
        scene_id = scene.id
        if scene_id not in selected_set:
            continue
        item = new_states_by_id[scene_id]
        attempted.append(scene_id)
        rerun_reason = reasons.get(scene_id)
        if item.status is CandidateGenerationStatus.success and rerun_reason is None:
            skipped.append(scene_id)
            continue
        candidate, failure, attempts = _generate_for_scene(
            scene,
            model_client,
            max_attempts=max_attempts,
            chunk_max_lines=int(state.generation_params["chunk_max_lines"]),
            hint_contract=hint_contract,
        )
        item.attempts += attempts
        if candidate is not None:
            item.candidate = candidate
            item.status = CandidateGenerationStatus.success
            item.last_failure = None
            if rerun_reason is not None:
                item.rerun_history.append(
                    RerunRecord(
                        rerun_reason=rerun_reason,
                        previous_status=CandidateGenerationStatus.success,
                        outcome="success",
                    )
                )
            succeeded.append(scene_id)
        elif rerun_reason is not None:
            # 显式重跑失败：旧候选与 success 状态保留（不损坏旧结果），审计 outcome=failed。
            item.rerun_history.append(
                RerunRecord(
                    rerun_reason=rerun_reason,
                    previous_status=CandidateGenerationStatus.success,
                    outcome="failed",
                )
            )
            item.last_failure = failure
            failed.append(scene_id)
        else:
            item.status = CandidateGenerationStatus.failed
            item.last_failure = failure
            failed.append(scene_id)
        if state_path is not None:
            save_candidate_run(state_path, new_state)
    return CandidateRunResult(
        new_state=new_state,
        attempted_scene_ids=attempted,
        succeeded_scene_ids=succeeded,
        failed_scene_ids=failed,
        skipped_scene_ids=skipped,
    )


# ---------- 候选合并进 P4A 审核文档 ----------


class CandidateMergeReport(BaseModel):
    """候选合并报告：合并进审核文档的结果与新文档（原输入不被修改）。"""

    model_config = ConfigDict(extra="forbid")

    review_doc: SceneMetadataReviewDocument
    on_conflict: Literal["skip", "overwrite"]
    merged_scene_ids: list[str]
    filled_fields: dict[str, list[str]]
    overwritten_fields: dict[str, list[str]]
    skipped_conflict: dict[str, list[str]]
    skipped_final_scene_ids: list[str]
    no_candidate_scene_ids: list[str]


def merge_candidates_into_review(
    bundle: FrozenSceneBundle,
    review_doc: SceneMetadataReviewDocument | dict[str, Any],
    run_state: CandidateRunState | dict[str, Any],
    *,
    on_conflict: Literal["skip", "overwrite"] = "skip",
) -> CandidateMergeReport:
    """把运行状态中的合法候选安全合并进 P4A 审核文档。

    门禁（合并前完整校验，任一失败抛 ValueError）：
    - 运行状态必须通过 validate_candidate_run（三摘要绑定同一 bundle）；
    - 审核文档必须通过 validate_scene_metadata_review（schema v1 等旧状态在此被
      拒绝，不做静默迁移）；
    - 顶层 review_status=approved 的审核文档拒绝合并（候选路径不得触碰 approved）。

    合并规则：
    - 只填充尚为空的字段（None 或空数组）；已有人工字段值相同则幂等跳过；
    - 值冲突时默认（on_conflict="skip"）跳过该场景并报告冲突字段；
    - on_conflict="overwrite" 为显式字段级覆盖策略：仅对 draft/needs_review 记录
      生效，覆盖后追加审计告警；reviewer / review_status 绝不被候选改写；
    - approved / rejected 记录一律跳过（skipped_final）；
    - 合并后的记录最多置为 needs_review，绝不自动 approved；
    - 顶层 review_status 保持 draft（P4A 只允许人工置为 approved）。
    """
    if on_conflict not in ("skip", "overwrite"):
        raise ValueError(f"on_conflict 必须为 'skip' 或 'overwrite'，当前为 {on_conflict!r}")
    state, coerce_errors = _coerce_run_state(run_state)
    if state is None:
        raise ValueError("候选运行状态非法:\n- " + "\n- ".join(coerce_errors))
    state_errors = validate_candidate_run(state, bundle)
    if state_errors:
        raise ValueError("候选运行状态未通过校验:\n- " + "\n- ".join(state_errors))
    review_errors = validate_scene_metadata_review(review_doc, bundle)
    if review_errors:
        raise ValueError("审核状态未通过校验（含 schema 版本检查，不做静默迁移）:\n- " + "\n- ".join(review_errors))
    doc, _ = _coerce_review_document(review_doc)
    if doc is None:  # pragma: no cover - validate 通过后必为合法模型
        raise ValueError("审核状态文档非法")
    if doc.review_status == "approved":
        raise ValueError("顶层 review_status=approved 的审核文档不接受候选合并（候选路径不得触碰 approved 状态）")

    candidates_by_id = {
        item.scene_id: item.candidate
        for item in state.scene_states
        if item.status is CandidateGenerationStatus.success and item.candidate is not None
    }
    decisions_by_id = {decision.scene_id: decision for decision in doc.scene_decisions}

    new_decisions: list[SceneMetadataDecision] = []
    merged: list[str] = []
    filled_fields: dict[str, list[str]] = {}
    overwritten_fields: dict[str, list[str]] = {}
    skipped_conflict: dict[str, list[str]] = {}
    skipped_final: list[str] = []
    no_candidate: list[str] = []

    for scene in bundle.scenes:
        decision = decisions_by_id[scene.id]
        candidate = candidates_by_id.get(scene.id)
        if candidate is None:
            no_candidate.append(scene.id)
            new_decisions.append(decision)
            continue
        if decision.review_status in (ReviewStatus.approved, ReviewStatus.rejected):
            skipped_final.append(scene.id)
            new_decisions.append(decision)
            continue

        fills: dict[str, Any] = {}
        conflicts: list[str] = []
        for field in CANDIDATE_MERGE_FIELDS:
            human_value = getattr(decision, field)
            candidate_value = getattr(candidate, field)
            if field == "warnings":
                combined = list(human_value)
                combined.extend(value for value in candidate_value if value not in combined)
                if combined != human_value:
                    fills[field] = combined
                continue
            # P4A 三态：人物字段的 [] 表示人工已审核且确认无人，不得当作“未填写”。
            # evidence/reasons 的默认空列表仍可由候选补齐。
            human_empty = human_value is None or (field in ("evidence", "reasons") and human_value == [])
            if human_empty:
                fills[field] = candidate_value
            elif human_value == candidate_value:
                continue  # 幂等：值一致则不动
            else:
                conflicts.append(field)

        if conflicts and on_conflict == "skip":
            skipped_conflict[scene.id] = conflicts
            new_decisions.append(decision)
            continue

        update: dict[str, Any] = dict(fills)
        if conflicts:  # 仅显式 overwrite 策略会走到这里
            for field in conflicts:
                update[field] = getattr(candidate, field)
            overwritten_fields[scene.id] = list(conflicts)
            audit = f"P4B 候选显式覆盖人工字段（model_id={state.model_id}）: {', '.join(sorted(conflicts))}"
            preserved_warnings = list(decision.warnings)
            preserved_warnings.extend(value for value in candidate.warnings if value not in preserved_warnings)
            preserved_warnings.append(audit)
            update["warnings"] = preserved_warnings
        if update or decision.review_status is ReviewStatus.draft:
            update["review_status"] = ReviewStatus.needs_review
            new_decisions.append(decision.model_copy(update=update))
        else:
            new_decisions.append(decision)
        merged.append(scene.id)
        filled_fields[scene.id] = sorted(fills)

    new_doc = SceneMetadataReviewDocument.model_validate(
        {
            **doc.model_dump(mode="json"),
            "scene_decisions": [decision.model_dump(mode="json") for decision in new_decisions],
        }
    )
    residual = validate_scene_metadata_review(new_doc, bundle)
    if residual:  # pragma: no cover - 合并产物必须始终通过 P4A 校验（内部不变量）
        raise RuntimeError(f"候选合并产物未通过 P4A 校验（内部不变量被破坏）: {residual}")

    return CandidateMergeReport(
        review_doc=new_doc,
        on_conflict=on_conflict,
        merged_scene_ids=merged,
        filled_fields=filled_fields,
        overwritten_fields=overwritten_fields,
        skipped_conflict=skipped_conflict,
        skipped_final_scene_ids=skipped_final,
        no_candidate_scene_ids=no_candidate,
    )


# ---------- 运行 manifest ----------


class CandidateRunManifest(BaseModel):
    """P4B 运行 manifest：唯一允许携带非确定性信息（时间戳）的产物。

    记录模型标识、参数快照、输入 bundle 三摘要、场景状态计数与本次运行计数；
    候选运行状态与 P4A 审核状态不复制时间戳（确定性约束）。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    generator: NonEmptyStr
    model_id: NonEmptyStr
    generation_params: dict[str, int | float | str | bool]
    source_bundle: SourceManifestRef
    total_scenes: int = Field(ge=1)
    scene_status_counts: dict[str, int]
    run_counts: dict[str, int]
    attempted_scene_ids: list[str]
    started_at: NonEmptyStr
    completed_at: NonEmptyStr
    note: str = DEFAULT_MANIFEST_NOTE


def build_candidate_run_manifest(
    run_state: CandidateRunState,
    run_result: CandidateRunResult,
    *,
    started_at: str,
    completed_at: str,
) -> CandidateRunManifest:
    """从运行状态与运行结果构建 manifest（时间戳由调用方显式提供）。"""
    state, state_errors = _coerce_run_state(run_state)
    if state is None:
        raise ValueError("候选运行状态非法，无法构建 manifest:\n- " + "\n- ".join(state_errors))
    try:
        result = CandidateRunResult.model_validate(run_result.model_dump(mode="json"))
    except ValidationError as exc:
        raise ValueError(f"候选运行结果非法，无法构建 manifest: {_summarize_validation_error(exc)}") from exc
    if result.new_state.model_dump(mode="json") != state.model_dump(mode="json"):
        raise ValueError("run_result.new_state 与用于构建 manifest 的 run_state 不一致")
    counts = Counter(item.status.value for item in state.scene_states)
    return CandidateRunManifest(
        schema_version=CANDIDATE_RUN_MANIFEST_SCHEMA_VERSION,
        generator=GENERATOR_ID,
        model_id=state.model_id,
        generation_params=dict(state.generation_params),
        source_bundle=state.source_manifest.model_copy(deep=True),
        total_scenes=state.total_source_scenes,
        scene_status_counts={status: counts.get(status, 0) for status in ("pending", "success", "failed")},
        run_counts={
            "attempted": len(result.attempted_scene_ids),
            "succeeded": len(result.succeeded_scene_ids),
            "failed": len(result.failed_scene_ids),
            "skipped": len(result.skipped_scene_ids),
        },
        attempted_scene_ids=list(result.attempted_scene_ids),
        started_at=started_at,
        completed_at=completed_at,
    )


def write_candidate_run_manifest(path: Path | str, manifest: CandidateRunManifest) -> None:
    """原子写出运行 manifest（单文件原子提交）。"""
    payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(Path(path), payload)


def save_candidate_run_with_manifest(
    state_path: Path | str,
    manifest_path: Path | str,
    run_state: CandidateRunState,
    run_result: CandidateRunResult,
    *,
    started_at: str,
    completed_at: str,
) -> CandidateRunManifest:
    """状态 + manifest 两文件整体原子提交（复用 P3/P4A 冻结对提交协议）。

    - primary = 状态文件（先备份旧版再替换），secondary = manifest（最后替换，
      作为整体提交完成标志）；secondary 失败时回滚 primary，不留混合版本；
    - 两文件必须位于同一目录（提交协议要求）；
    - 发现既有 `.tmp.old` 恢复副本时拒绝覆盖（人工恢复优先）；
    - 时间戳只写入 manifest，状态文件保持确定性。
    """
    state_path = Path(state_path)
    manifest_path = Path(manifest_path)
    if state_path.parent != manifest_path.parent:
        raise ValueError("状态文件与 manifest 必须位于同一目录（两文件整体提交协议要求）")
    if state_path == manifest_path:
        raise ValueError("状态文件与 manifest 必须使用两个不同路径")
    doc, coerce_errors = _coerce_run_state(run_state)
    if doc is None:
        raise ValueError("候选运行状态非法，拒绝写出:\n- " + "\n- ".join(coerce_errors))
    revalidated = CandidateRunState.model_validate(doc.model_dump(mode="json"))
    manifest = build_candidate_run_manifest(revalidated, run_result, started_at=started_at, completed_at=completed_at)
    _atomic_write_pair(
        state_path.parent,
        primary_name=state_path.name,
        primary_payload=json.dumps(revalidated.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        secondary_name=manifest_path.name,
        secondary_payload=json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    return manifest
