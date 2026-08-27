"""P4A 场景元数据审核基础设施。

职责：冻结场景包输入门禁（scenes.jsonl + boundary_manifest.json）、
P4 场景元数据审核状态的创建/校验/原子保存/读取、人工审核 Markdown 包生成、
approved 元数据应用，以及 enriched 场景的可选原子输出。

明确不做：场景切分（scene_segmenter）、LLM 调用与 prompt 管理、embedding、
检索、数据库写入、API、事实卡抽取——这些属于 P4B 及以后。

三态语义（与 P1 Schema 的 None/unknown 约定一致，不得混用）：
- None = 尚未审核（初始状态）；
- unknown = 已审核但无法判断（viewpoint 用字符串 "unknown"，
  temporal_scope/reality_status 用枚举 unknown）；
- 空人物数组 = 已审核且确认没有对应人物；不得用空数组冒充未审核状态。

审核状态不含时间戳或随机 UUID：同一 bundle 连续创建两次结果完全一致；
运行期非确定性信息（如生成时间）如需记录，应放入独立的运行 manifest。

详见 docs/research/KISAKI_GAME_RAG_SCENE_METADATA_REVIEW.md。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

SCENE_METADATA_REVIEW_SCHEMA_VERSION = 2
SUPPORTED_BOUNDARY_MANIFEST_SCHEMA_VERSIONS = (1,)
ENRICHED_MANIFEST_SCHEMA_VERSION = 1
GENERATOR_ID = "knowledge.game_rag.scene_metadata_review"

# viewpoint 规范："<人物名>第一人称" / "第三人称" / "多视角" / "unknown"。
# 不假定人物名单——人物名不做白名单，只做格式约束；None = 尚未审核。
VIEWPOINT_UNKNOWN = "unknown"
_VIEWPOINT_SPECIAL_VALUES = frozenset({"第三人称", "多视角", VIEWPOINT_UNKNOWN})
_VIEWPOINT_FIRST_PERSON_RE = re.compile(r"^\S+第一人称$")

# 审核包原文展示规则：短场景完整展示，长场景只给首尾摘录（摘录不构成完整审核）。
REVIEW_PACK_FULL_TEXT_MAX_LINES = 40
REVIEW_PACK_EXCERPT_EDGE_LINES = 8

DEFAULT_REVIEW_NOTES = (
    "场景元数据审核状态（P4A）。待审核字段为 None（尚未审核）；"
    "已审核但无法判断填 unknown（viewpoint 用字符串，temporal_scope/reality_status 用枚举）；"
    "空人物数组表示已审核且确认没有对应人物。"
    "scene_id / source / story_unit_id 创建后不得更换。"
)


def _summarize_validation_error(exc: ValidationError, *, max_items: int = 5) -> str:
    """把 Pydantic ValidationError 压缩为可读摘要（用于拒绝信息）。"""
    errors = exc.errors()
    parts = []
    for error in errors[:max_items]:
        loc = ".".join(str(item) for item in error["loc"]) or "(root)"
        parts.append(f"{loc}: {error['msg']}")
    summary = "; ".join(parts)
    if len(errors) > max_items:
        summary += f"；另有 {len(errors) - max_items} 项错误"
    return summary


# ---------- 冻结场景包输入门禁 ----------


class FrozenUnitSummary(BaseModel):
    """P3 boundary_manifest.units[uid] 的结构契约（字段全集，extra=forbid）。"""

    model_config = ConfigDict(extra="forbid")

    story_title: NonEmptyStr
    boundaries: list[int]
    decisions: dict[str, str]
    adds: list[int]
    scenes: int = Field(ge=0)


class FrozenBoundaryManifest(BaseModel):
    """P3 boundary_manifest.json 的结构契约（字段全集，extra=forbid）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    boundary_review_status: str
    reviewer: NonEmptyStr
    min_dialogue_turns: int
    source_prefix: NonEmptyStr
    units: dict[str, FrozenUnitSummary]
    total_scenes: int = Field(ge=1)
    scene_review_status: str
    note: str


@dataclass(frozen=True)
class FrozenSceneBundle:
    """通过输入门禁的冻结场景包（不可变视图）。"""

    scenes: tuple[SceneDocument, ...]
    manifest: FrozenBoundaryManifest
    manifest_digest: str
    scenes_digest: str
    bundle_digest: str

    @property
    def total_scenes(self) -> int:
        return len(self.scenes)


def _manifest_digest(manifest: FrozenBoundaryManifest) -> str:
    """manifest 内容摘要（规范化 JSON 的 sha256）：审核状态与冻结包的稳定关联键。"""
    canonical = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scenes_digest(scenes: list[SceneDocument]) -> str:
    """场景内容摘要：绑定记录顺序与 SceneDocument 全字段。"""
    digest = hashlib.sha256()
    for scene in scenes:
        canonical = json.dumps(scene.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _bundle_digest(manifest_digest: str, scenes_digest: str) -> str:
    """冻结包整体摘要：防止合法形状的 scenes 与另一份 manifest 被重新配对。"""
    return hashlib.sha256(f"{manifest_digest}:{scenes_digest}".encode()).hexdigest()


def _bundle_integrity_errors(bundle: FrozenSceneBundle) -> list[str]:
    """重算摘要，拒绝加载后被原地修改或手工拼装的 bundle。"""
    errors: list[str] = []
    current_manifest = _manifest_digest(bundle.manifest)
    current_scenes = _scenes_digest(list(bundle.scenes))
    current_bundle = _bundle_digest(current_manifest, current_scenes)
    if current_manifest != bundle.manifest_digest:
        errors.append("FrozenSceneBundle.manifest 在加载后被修改，缓存摘要已失效")
    if current_scenes != bundle.scenes_digest:
        errors.append("FrozenSceneBundle.scenes 在加载后被修改，缓存摘要已失效")
    if current_bundle != bundle.bundle_digest:
        errors.append("FrozenSceneBundle 双文件组合摘要无效")
    return errors


def load_frozen_scene_bundle(scene_path: Path | str, manifest_path: Path | str) -> FrozenSceneBundle:
    """冻结场景包输入门禁：全部校验通过才返回 bundle，供创建审核状态使用。

    校验（创建审核状态之前全部完成）：
    1. 两文件均存在（不接受孤立 scenes.jsonl 或孤立 manifest）；
    2. manifest 是合法 JSON；3. schema 版本受支持；4. boundary_review_status=approved；
    5. scene_review_status=draft；6. scenes.jsonl 每个非空行可解析；7. 每条记录通过
    SceneDocument 校验；8. scene id 唯一；9. 场景数 = manifest.total_scenes；
    10. 每个单元场景数与 manifest 一致；11. source_path 与 story_unit_id 对应
    （同一单元的场景共享同一 source_path，且单元须在 manifest.units 登记）；
    12. 同一单元 span 有序不重叠；13. text 不含 \\x1A；14. 输入场景均为 draft。
    """
    scene_path = Path(scene_path)
    manifest_path = Path(manifest_path)
    if not scene_path.is_file():
        raise ValueError(f"scenes.jsonl 不存在: {scene_path}（不接受缺少 scenes 的孤立 manifest）")
    if not manifest_path.is_file():
        raise ValueError(f"boundary_manifest.json 不存在: {manifest_path}（不接受缺少 manifest 的孤立 scenes.jsonl）")

    try:
        raw_manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"boundary_manifest.json 不是合法 JSON: {exc}") from exc

    version = raw_manifest.get("schema_version") if isinstance(raw_manifest, dict) else None
    if version not in SUPPORTED_BOUNDARY_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            f"boundary manifest schema_version 不受支持: {version!r}"
            f"（支持: {list(SUPPORTED_BOUNDARY_MANIFEST_SCHEMA_VERSIONS)}）"
        )
    try:
        manifest = FrozenBoundaryManifest.model_validate(raw_manifest)
    except ValidationError as exc:
        raise ValueError(f"boundary_manifest.json 结构校验失败: {_summarize_validation_error(exc)}") from exc

    if manifest.boundary_review_status != "approved":
        raise ValueError(f"boundary_review_status 必须为 approved，当前为 {manifest.boundary_review_status!r}")
    if manifest.scene_review_status != "draft":
        raise ValueError(
            f"manifest.scene_review_status 必须为 draft（P4A 只消费未做内容审核的冻结场景），"
            f"当前为 {manifest.scene_review_status!r}"
        )

    try:
        scene_lines = scene_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"scenes.jsonl 读取失败: {exc}") from exc

    scenes: list[SceneDocument] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(scene_lines, start=1):
        if not line.strip():
            continue
        try:
            scene = SceneDocument.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(
                f"scenes.jsonl 第 {lineno} 行不是合法 SceneDocument: {_summarize_validation_error(exc)}"
            ) from exc
        if scene.id in seen_ids:
            raise ValueError(f"scenes.jsonl 存在重复 scene id: {scene.id}")
        seen_ids.add(scene.id)
        scenes.append(scene)

    if len(scenes) != manifest.total_scenes:
        raise ValueError(f"场景数量 {len(scenes)} 与 manifest.total_scenes {manifest.total_scenes} 不一致")

    scenes_by_unit: dict[str, list[SceneDocument]] = {}
    for scene in scenes:
        unit_id = scene.story.story_unit_id
        if unit_id not in manifest.units:
            raise ValueError(f"scene {scene.id} 的 story_unit_id {unit_id!r} 未在 manifest.units 中登记")
        scenes_by_unit.setdefault(unit_id, []).append(scene)

    for unit_id, summary in manifest.units.items():
        unit_scenes = scenes_by_unit.get(unit_id, [])
        if len(unit_scenes) != summary.scenes:
            raise ValueError(f"{unit_id}: 场景数 {len(unit_scenes)} 与 manifest 记录的 {summary.scenes} 不一致")
        if any(scene.story.story_title != summary.story_title for scene in unit_scenes):
            raise ValueError(f"{unit_id}: SceneDocument.story_title 与 manifest.units.story_title 不一致")
        paths = {scene.source.source_path for scene in unit_scenes}
        if len(paths) > 1:
            raise ValueError(f"{unit_id}: 同一 story unit 的场景出现多个 source_path: {sorted(paths)}")
        if paths and not next(iter(paths)).startswith(f"{manifest.source_prefix}/"):
            raise ValueError(f"{unit_id}: source_path 不在 manifest.source_prefix {manifest.source_prefix!r} 下")
        prev_end = 0
        for scene in unit_scenes:
            if scene.source.line_start <= prev_end:
                raise ValueError(
                    f"{unit_id}: 场景 span 无序或重叠（L{scene.source.line_start}-{scene.source.line_end} "
                    f"未严格晚于上一场景结束行 L{prev_end}）"
                )
            prev_end = scene.source.line_end
        actual_boundaries = [scene.source.line_start for scene in unit_scenes[1:]]
        if actual_boundaries != summary.boundaries:
            raise ValueError(
                f"{unit_id}: 场景起点 {actual_boundaries} 与 manifest.boundaries {summary.boundaries} 不一致"
            )

    for scene in scenes:
        if "\x1a" in scene.text:
            raise ValueError(f"scene {scene.id} 的 text 含有 \\x1a（DOS EOF 标记不得进入场景文本）")
        if scene.review_status is not ReviewStatus.draft:
            raise ValueError(
                f"scene {scene.id} 的 review_status 必须为 draft（P4A 输入门禁只接受未审核场景），"
                f"当前为 {scene.review_status.value!r}"
            )

    manifest_digest = _manifest_digest(manifest)
    scenes_digest = _scenes_digest(scenes)
    return FrozenSceneBundle(
        scenes=tuple(scenes),
        manifest=manifest,
        manifest_digest=manifest_digest,
        scenes_digest=scenes_digest,
        bundle_digest=_bundle_digest(manifest_digest, scenes_digest),
    )


# ---------- 审核记录与审核状态文档 ----------


def _normalize_viewpoint(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("viewpoint 不得为空白字符串")
    if stripped in _VIEWPOINT_SPECIAL_VALUES or _VIEWPOINT_FIRST_PERSON_RE.match(stripped):
        return stripped
    raise ValueError(
        f"viewpoint {value!r} 不符合规范：须为「<人物名>第一人称」「第三人称」「多视角」或 'unknown'（None=尚未审核）"
    )


def _normalize_character_names(value: list[str] | None) -> list[str] | None:
    if value is None:
        return None
    names: list[str] = []
    for name in value:
        stripped = name.strip()
        if not stripped:
            raise ValueError("人物名去除首尾空白后不得为空字符串")
        names.append(stripped)
    return sorted(set(names))


class SceneMetadataDecision(BaseModel):
    """单个冻结场景的 P4 元数据审核记录。

    - 待审核字段（viewpoint / temporal_scope / reality_status / mentioned_characters /
      present_characters / evidence）初始为 None：None=尚未审核，空数组=已审核且确认无人；
    - approved 记录不得保留任何 None 待审核字段，且 evidence 与 reasons 非空、reviewer 非空；
    - scene_id / source / story_unit_id 创建时从冻结场景复制，审核过程中不得更换
      （由 validate_scene_metadata_review 对照 bundle 校验）。
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: NonEmptyStr
    story_unit_id: NonEmptyStr
    source: SourceSpan
    viewpoint: str | None = Field(default=None, description="视角；None=尚未审核，'unknown'=已审核但无法判断")
    temporal_scope: TemporalScope | None = Field(default=None, description="None=尚未审核；unknown=已审核但无法判断")
    reality_status: RealityStatus | None = Field(default=None, description="None=尚未审核；unknown=已审核但无法判断")
    mentioned_characters: list[str] | None = Field(
        default=None, description="被提及人物；None=尚未审核，[]=已审核且确认无"
    )
    present_characters: list[str] | None = Field(
        default=None, description="当前叙事层实际在场人物；None=尚未审核，[]=已审核且确认无"
    )
    evidence: list[SourceSpan] | None = Field(
        default=None, description="场景内证据行号范围；approved 记录必须至少一条且落在场景范围内"
    )
    reasons: NonEmptyStrList = Field(default_factory=list, description="决定理由；approved 记录必须非空")
    warnings: NonEmptyStrList = Field(default_factory=list, description="审核中的告警备注")
    review_status: ReviewStatus = ReviewStatus.draft
    reviewer: str = ""

    @field_validator("viewpoint")
    @classmethod
    def _validate_viewpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_viewpoint(value)

    @field_validator("mentioned_characters", "present_characters")
    @classmethod
    def _validate_characters(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_character_names(value)

    @model_validator(mode="after")
    def _check_consistency(self) -> SceneMetadataDecision:
        for span in self.evidence or []:
            if span.source_path != self.source.source_path:
                raise ValueError(f"evidence 出处 {span.source_path!r} 与场景 source {self.source.source_path!r} 不一致")
            if span.line_start < self.source.line_start or span.line_end > self.source.line_end:
                raise ValueError(
                    f"evidence L{span.line_start}-{span.line_end} 超出场景范围 "
                    f"L{self.source.line_start}-{self.source.line_end}"
                )
        if self.review_status is ReviewStatus.approved:
            missing = [
                name
                for name, value in (
                    ("viewpoint", self.viewpoint),
                    ("temporal_scope", self.temporal_scope),
                    ("reality_status", self.reality_status),
                    ("mentioned_characters", self.mentioned_characters),
                    ("present_characters", self.present_characters),
                    ("evidence", self.evidence),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"approved 记录不得保留未审核字段（None）: {missing}")
            if not self.evidence:
                raise ValueError("approved 记录必须提供至少一条场景内 evidence")
            if not self.reasons:
                raise ValueError("approved 记录必须提供非空 reasons")
            if not self.reviewer.strip():
                raise ValueError("approved 记录的 reviewer 不得为空")
        return self


class SourceManifestRef(BaseModel):
    """审核状态对源冻结包的稳定标识（双文件摘要 + 关键字段快照）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    boundary_review_status: str
    reviewer: NonEmptyStr
    total_scenes: int = Field(ge=1)
    manifest_sha256: str
    scenes_sha256: str
    bundle_sha256: str


class SceneMetadataReviewDocument(BaseModel):
    """P4A 场景元数据审核状态（版本化顶层文档）。

    确定性：不含时间戳/随机 UUID，同一 bundle 连续创建两次完全一致。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source_manifest: SourceManifestRef
    total_source_scenes: int = Field(ge=1)
    reviewer: str = ""
    review_status: Literal["draft", "approved"] = "draft"
    scene_decisions: list[SceneMetadataDecision] = Field(default_factory=list)
    notes: str = DEFAULT_REVIEW_NOTES
    created_by: NonEmptyStr = GENERATOR_ID

    @model_validator(mode="after")
    def _check_decisions(self) -> SceneMetadataReviewDocument:
        if len(self.scene_decisions) != self.total_source_scenes:
            raise ValueError(
                f"scene_decisions 数量 {len(self.scene_decisions)} 与 total_source_scenes "
                f"{self.total_source_scenes} 不一致（每个输入场景必须恰好一条决定）"
            )
        counts = Counter(decision.scene_id for decision in self.scene_decisions)
        if duplicates := sorted(scene_id for scene_id, n in counts.items() if n > 1):
            raise ValueError(f"scene_decisions 存在重复 scene_id: {duplicates}")
        return self


# ---------- 创建与校验 ----------


def create_scene_metadata_review(bundle: FrozenSceneBundle, *, reviewer: str = "") -> SceneMetadataReviewDocument:
    """从冻结场景包创建初始审核状态：全部 draft、待审核字段全部 None。

    - 不修改输入 bundle 或其 SceneDocument；
    - 记录按冻结场景顺序创建，scene_id / source / story_unit_id 原样复制；
    - 不自动用 speakers 填充 present_characters，不推断视角/时间状态/现实状态；
    - reviewer 允许为空（draft 阶段），记录级 reviewer 继承该值（approved 时必须非空）；
    - 同一 bundle 连续创建两次结果完全一致（无时间戳/随机数）。
    """
    if integrity_errors := _bundle_integrity_errors(bundle):
        raise ValueError("冻结场景包完整性校验失败:\n- " + "\n- ".join(integrity_errors))
    reviewer_value = str(reviewer or "")
    decisions = [
        SceneMetadataDecision(
            scene_id=scene.id,
            story_unit_id=scene.story.story_unit_id,
            source=scene.source.model_copy(deep=True),
            reviewer=reviewer_value,
        )
        for scene in bundle.scenes
    ]
    return SceneMetadataReviewDocument(
        schema_version=SCENE_METADATA_REVIEW_SCHEMA_VERSION,
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
        reviewer=reviewer_value,
        review_status="draft",
        scene_decisions=decisions,
    )


def _coerce_review_document(
    review_doc: SceneMetadataReviewDocument | dict[str, Any],
) -> tuple[SceneMetadataReviewDocument | None, list[str]]:
    if isinstance(review_doc, SceneMetadataReviewDocument):
        try:
            # Pydantic 模型默认允许属性赋值；跨入口使用前必须重放模型校验，
            # 不能把“曾经合法”误当作“当前仍合法”。
            return SceneMetadataReviewDocument.model_validate(review_doc.model_dump(mode="json")), []
        except ValidationError as exc:
            return None, [f"审核状态文档结构非法: {_summarize_validation_error(exc)}"]
    if isinstance(review_doc, dict):
        try:
            return SceneMetadataReviewDocument.model_validate(review_doc), []
        except ValidationError as exc:
            return None, [f"审核状态文档结构非法: {_summarize_validation_error(exc)}"]
    return None, [f"审核状态文档类型非法: {type(review_doc).__name__}（须为 SceneMetadataReviewDocument 或 dict）"]


def validate_scene_metadata_review(
    review_doc: SceneMetadataReviewDocument | dict[str, Any],
    bundle: FrozenSceneBundle,
    *,
    require_complete: bool = False,
) -> list[str]:
    """校验审核状态与冻结场景包的跨字段一致性，返回错误列表（空=通过）。

    - schema_version / 顶层结构 / review_status 合法性；
    - source_manifest 与 bundle 一致（manifest_sha256 钉住审核状态对应的冻结包）；
    - scene 集合完整：不缺失、无未知、无重复（重复由文档模型自身拒绝）；
    - scene_id / source / story_unit_id 不可篡改；
    - 枚举、人物数组、evidence 范围、reasons 结构、reviewer 语义由记录模型保证；
    - 顶层 approved 时不得存在 draft/needs_review/rejected 场景，reviewer 不得为空；
    - require_complete=True 时全部场景必须明确审核（review_status 不得为 draft）。
    """
    doc, errors = _coerce_review_document(review_doc)
    if doc is None:
        return errors

    errors.extend(_bundle_integrity_errors(bundle))

    if doc.schema_version != SCENE_METADATA_REVIEW_SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCENE_METADATA_REVIEW_SCHEMA_VERSION}，当前为 {doc.schema_version!r}")

    ref = doc.source_manifest
    if ref.manifest_sha256 != bundle.manifest_digest:
        errors.append(
            "source_manifest.manifest_sha256 与 bundle 不一致：审核状态不是针对当前冻结场景包创建的"
            f"（{ref.manifest_sha256[:12]}… != {bundle.manifest_digest[:12]}…）"
        )
    if ref.scenes_sha256 != bundle.scenes_digest:
        errors.append(
            "source_manifest.scenes_sha256 与 bundle 不一致：审核状态不是针对当前 scenes.jsonl 创建的"
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
    decisions_by_id = {decision.scene_id: decision for decision in doc.scene_decisions}
    if unknown := sorted(set(decisions_by_id) - set(bundle_by_id)):
        errors.append(f"scene_decisions 含未知 scene: {unknown}")
    if missing := sorted(set(bundle_by_id) - set(decisions_by_id)):
        errors.append(f"scene_decisions 缺少 scene: {missing}")

    for scene_id in sorted(set(decisions_by_id) & set(bundle_by_id)):
        decision = decisions_by_id[scene_id]
        scene = bundle_by_id[scene_id]
        if decision.story_unit_id != scene.story.story_unit_id:
            errors.append(
                f"{scene_id}: story_unit_id 被篡改（{decision.story_unit_id!r} != {scene.story.story_unit_id!r}）"
            )
        if decision.source != scene.source:
            errors.append(
                f"{scene_id}: source 被篡改（L{decision.source.line_start}-{decision.source.line_end} != "
                f"L{scene.source.line_start}-{scene.source.line_end}）"
            )

    if require_complete:
        for decision in doc.scene_decisions:
            if decision.review_status is ReviewStatus.draft:
                errors.append(f"{decision.scene_id}: review_status 仍为 draft（require_complete 要求全部场景明确审核）")

    if doc.review_status == "approved":
        if not doc.reviewer.strip():
            errors.append("顶层 review_status=approved 时 reviewer 不得为空")
        for decision in doc.scene_decisions:
            if decision.review_status is not ReviewStatus.approved:
                errors.append(
                    f"顶层 review_status=approved 时不得存在 {decision.review_status.value} 场景: {decision.scene_id}"
                )
    return errors


# ---------- 原子保存与读取 ----------


def _atomic_write_text(path: Path, payload: str) -> None:
    """单文件原子写：内存序列化 → 同目录确定性 .tmp → os.replace。

    - tmp 写入失败或替换失败时旧文件保持不变；
    - 失败后尽量清理未完成的 tmp（清理失败不得掩盖原始异常）；
    - 成功后无 tmp 残留。不使用随机临时路径。
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(payload, encoding="utf-8", newline="\n")
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    try:
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def save_scene_metadata_review(path: Path | str, review_doc: SceneMetadataReviewDocument | dict[str, Any]) -> None:
    """原子保存审核状态；写出前先做结构校验，拒绝写出非法状态。"""
    doc, errors = _coerce_review_document(review_doc)
    if doc is None:
        raise ValueError("审核状态文档非法，拒绝写出:\n- " + "\n- ".join(errors))
    payload = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(Path(path), payload)


def load_scene_metadata_review(path: Path | str) -> SceneMetadataReviewDocument:
    """读取审核状态文件（JSON → SceneMetadataReviewDocument）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return SceneMetadataReviewDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"审核状态文件结构非法: {_summarize_validation_error(exc)}") from exc


# ---------- 人工审核包（Markdown） ----------


def _render_scene_form(scene: SceneDocument) -> list[str]:
    span_lines = scene.source.line_end - scene.source.line_start + 1
    text_lines = scene.text.split("\n")
    parts = [
        f"### {scene.id}｜L{scene.source.line_start}-{scene.source.line_end}｜"
        f"{scene.story.story_title}｜{span_lines} 行｜speakers: {'、'.join(scene.speakers) or '（无）'}",
        "",
    ]
    parts.append("```text")
    if len(text_lines) <= REVIEW_PACK_FULL_TEXT_MAX_LINES:
        parts.extend(text_lines)
    else:
        omitted = len(text_lines) - 2 * REVIEW_PACK_EXCERPT_EDGE_LINES
        parts.extend(text_lines[:REVIEW_PACK_EXCERPT_EDGE_LINES])
        parts.append(
            f"……（中间省略 {omitted} 行；以上为摘录，摘录不构成完整人工审核，"
            f"完整原文见 source span L{scene.source.line_start}-{scene.source.line_end}）……"
        )
        parts.extend(text_lines[-REVIEW_PACK_EXCERPT_EDGE_LINES:])
    parts.append("```")
    parts.extend(
        [
            "",
            "- viewpoint: ______（<人物名>第一人称 / 第三人称 / 多视角 / unknown）",
            "- temporal_scope: ______（current / flashback / reconstruction / hypothetical / unknown）",
            "- reality_status: ______（objective / character_claim / inferred / fictional / conflicted / unknown）",
            "- mentioned_characters: ______（分隔符列举；已审核且确认无则填 []）",
            "- present_characters: ______（当前叙事层实际在场；已审核且确认无则填 []）",
            f"- evidence: ______（场景内行号范围，如 L{scene.source.line_start}-L{scene.source.line_start}）",
            "- reason: ______",
            "- review_status: ______（draft / needs_review / approved / rejected）",
            "",
        ]
    )
    return parts


def generate_scene_metadata_review_pack(
    bundle: FrozenSceneBundle,
    *,
    out_path: Path | str | None = None,
) -> str:
    """生成人工审核 Markdown 包（确定性输出；提供 out_path 时原子写出）。

    - 按 story unit（首次出现顺序）与 source 顺序展示全部冻结场景；
    - speakers 仅作人工参考：有台词者不一定属于现实层在场人物；
    - 长场景仅展示首尾摘录并显式声明摘录不构成完整审核。
    """
    if integrity_errors := _bundle_integrity_errors(bundle):
        raise ValueError("冻结场景包完整性校验失败:\n- " + "\n- ".join(integrity_errors))
    lines: list[str] = [
        "# 场景元数据审核包（P4A）",
        "",
        f"- 源 boundary manifest：schema_version={bundle.manifest.schema_version}，"
        f"reviewer={bundle.manifest.reviewer}，total_scenes={bundle.total_scenes}，"
        f"bundle_sha256={bundle.bundle_digest[:16]}…",
        "- 填写语义：待审核字段留空（None=尚未审核）；已审核但无法判断填 unknown"
        "（viewpoint 用字符串 'unknown'，temporal_scope/reality_status 用枚举 unknown）；"
        "已审核且确认无人物填 []，不得用空数组冒充未审核",
        "- speakers 仅为人工参考（书中故事/回忆/梦境/转述可打破「有台词=现实在场」假设）",
        f"- 原文展示：≤{REVIEW_PACK_FULL_TEXT_MAX_LINES} 行完整展示；更长场景仅展示首尾各 "
        f"{REVIEW_PACK_EXCERPT_EDGE_LINES} 行摘录，长场景审核必须对照完整 source span 原文",
        "",
    ]
    unit_scenes: dict[str, list[SceneDocument]] = {}
    for scene in bundle.scenes:
        unit_scenes.setdefault(scene.story.story_unit_id, []).append(scene)
    for unit_id, scenes in unit_scenes.items():
        lines.append(f"## {unit_id}（{scenes[0].story.story_title}｜{len(scenes)} 个场景）")
        lines.append("")
        for scene in scenes:
            lines.extend(_render_scene_form(scene))
        lines.append("")
    content = "\n".join(lines).rstrip("\n") + "\n"
    if out_path is not None:
        _atomic_write_text(Path(out_path), content)
    return content


# ---------- 应用 approved 元数据 ----------


def apply_approved_scene_metadata(
    bundle: FrozenSceneBundle,
    review_doc: SceneMetadataReviewDocument | dict[str, Any],
) -> list[SceneDocument]:
    """应用 approved 元数据，返回新的 SceneDocument 列表（输入 bundle 与审核状态均不被修改）。

    门禁：先完整校验；顶层 review_status 必须为 approved（draft/未完成一律拒绝），
    由校验保证全部记录 approved、reviewer 非空、无 None 待审核字段。
    写入：story.viewpoint / story.temporal_scope / reality_status /
    mentioned_characters / present_characters，review_status 变为 approved；
    text / id / source / speakers / story_unit_id / 顺序全部守恒。
    """
    errors = validate_scene_metadata_review(review_doc, bundle)
    if errors:
        raise ValueError("场景元数据审核状态未通过校验:\n- " + "\n- ".join(errors))
    doc, _ = _coerce_review_document(review_doc)
    if doc is None:  # pragma: no cover - validate 通过后必为合法模型
        raise ValueError("审核状态文档非法")
    if doc.review_status != "approved":
        raise ValueError(f"顶层 review_status 必须为 approved 才能应用元数据，当前为 {doc.review_status!r}")

    decisions_by_id = {decision.scene_id: decision for decision in doc.scene_decisions}
    enriched: list[SceneDocument] = []
    for scene in bundle.scenes:
        decision = decisions_by_id[scene.id]
        story = scene.story.model_copy(
            update={"viewpoint": decision.viewpoint, "temporal_scope": decision.temporal_scope}
        )
        enriched.append(
            scene.model_copy(
                update={
                    "story": story,
                    "reality_status": decision.reality_status,
                    "mentioned_characters": list(decision.mentioned_characters or []),
                    "present_characters": list(decision.present_characters or []),
                    "review_status": ReviewStatus.approved,
                }
            )
        )
    return enriched


# ---------- enriched 场景输出门禁（可选原子输出） ----------


def _atomic_restore_primary(primary: Path, backup: Path, *, had_old: bool) -> bool:
    """把旧版 primary 恢复原位（无旧版则删除已提交的新版），返回是否成功。"""
    try:
        if had_old:
            os.replace(backup, primary)
        else:
            primary.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _atomic_write_pair(
    out_dir: Path,
    *,
    primary_name: str,
    primary_payload: str,
    secondary_name: str,
    secondary_payload: str,
) -> None:
    """两文件整体原子提交（沿用 P3 冻结对提交协议）。

    1. 两份内容全部写入各自 .tmp（失败发生在提交前，旧文件未动）；
    2. primary 先备份旧版（若存在）再替换，secondary 最后替换作为完成标志；
    3. secondary 替换失败时回滚 primary，不留混合版本；
    4. 回滚失败时保留备份（旧版唯一可恢复副本）并在错误信息中报告路径；
    5. 提交成功或回滚成功后清理备份与 tmp，成功后无残留。
    """
    primary = out_dir / primary_name
    secondary = out_dir / secondary_name
    primary_tmp = primary.with_suffix(primary.suffix + ".tmp")
    secondary_tmp = secondary.with_suffix(secondary.suffix + ".tmp")
    backup = primary.with_suffix(primary.suffix + ".tmp.old")

    if backup.exists():
        raise ValueError(f"检测到未恢复的旧版备份 {backup}；请先人工恢复或移走后重试")

    try:
        primary_tmp.write_text(primary_payload, encoding="utf-8", newline="\n")
        secondary_tmp.write_text(secondary_payload, encoding="utf-8", newline="\n")
    except BaseException:
        for tmp in (primary_tmp, secondary_tmp):
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        raise

    had_old = primary.exists()
    if had_old:
        try:
            os.replace(primary, backup)
        except OSError:
            for tmp in (primary_tmp, secondary_tmp):
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            raise
    committed = False
    rollback_done = False
    rollback_attempted = False
    try:
        os.replace(primary_tmp, primary)
        try:
            os.replace(secondary_tmp, secondary)
        except OSError as exc:
            rollback_attempted = True
            rollback_done = _atomic_restore_primary(primary, backup, had_old=had_old)
            if not rollback_done:
                # 瞬时文件占用可能只影响第一次恢复；明确重试一次后再判定最终状态。
                rollback_done = _atomic_restore_primary(primary, backup, had_old=had_old)
            if rollback_done:
                raise ValueError(f"{secondary_name} 替换失败，已回滚 {primary_name}（不留混合版本）: {exc}") from exc
            raise ValueError(
                f"{secondary_name} 替换失败且 {primary_name} 回滚失败；"
                f"旧版本备份保留于 {backup}（唯一可恢复副本，请手动恢复后重试）: {exc}"
            ) from exc
        committed = True
    except BaseException:
        if not rollback_attempted:
            rollback_done = _atomic_restore_primary(primary, backup, had_old=had_old)
        for tmp in (primary_tmp, secondary_tmp):
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        raise
    finally:
        if committed or rollback_done:
            with contextlib.suppress(OSError):
                backup.unlink(missing_ok=True)


def write_enriched_scenes(
    bundle: FrozenSceneBundle,
    review_doc: SceneMetadataReviewDocument | dict[str, Any],
    out_dir: Path | str,
) -> dict[str, Any]:
    """写出 approved 应用结果（enriched_scenes.jsonl + enriched_manifest.json，整体原子提交）。

    输出门禁：
    - 必须提供与 bundle 双摘要绑定的 approved 审核文档；
    - 场景只在函数内部由 apply_approved_scene_metadata 生成，不接受外部可篡改列表；
    - 使用独立输出目录，不覆盖 P3 原始冻结产物；
    - manifest 记录源 boundary manifest 的稳定标识（sha256 摘要 + 关键字段）；
    - 场景数量与输入 bundle 相同。
    """
    scenes = apply_approved_scene_metadata(bundle, review_doc)

    manifest: dict[str, Any] = {
        "schema_version": ENRICHED_MANIFEST_SCHEMA_VERSION,
        "generator": GENERATOR_ID,
        "source_boundary_manifest": {
            "schema_version": bundle.manifest.schema_version,
            "boundary_review_status": bundle.manifest.boundary_review_status,
            "reviewer": bundle.manifest.reviewer,
            "total_scenes": bundle.manifest.total_scenes,
            "manifest_sha256": bundle.manifest_digest,
            "scenes_sha256": bundle.scenes_digest,
            "bundle_sha256": bundle.bundle_digest,
        },
        "total_scenes": len(scenes),
        "scene_review_status": "approved",
        "note": (
            "P4A enriched scenes: approved scene metadata applied on top of the frozen P3 bundle "
            "identified by source_boundary_manifest.manifest_sha256; P3 frozen artifacts stay untouched."
        ),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_pair(
        out_dir,
        primary_name="enriched_scenes.jsonl",
        primary_payload="".join(scene.model_dump_json() + "\n" for scene in scenes),
        secondary_name="enriched_manifest.json",
        secondary_payload=json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest
