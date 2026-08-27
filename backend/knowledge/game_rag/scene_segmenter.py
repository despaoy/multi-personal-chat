"""保守场景切分器（P3/P3.1）。

流程：边界候选检测 → 人工决定（approved 决定逐条显式）→ SceneDocument 组装 → 冻结。
本模块只产出内存对象、审核材料与内部冻结实现；公开冻结入口位于
low_review.freeze_reviewed_scenes，必须同时验证 high/medium 与 low 两层审核，
再从原始语料**重新构建**场景并写出 scenes.jsonl 与 boundary_manifest.json。

P3.1 关键口径：
- SceneDocument.text = 场景行号范围内的**原文物理行逐字拼接**（dialogue 保留 [说话人] 标签，
  空行按行号差自动恢复，未闭合台词/重复闭引号原样保留）；
- temporal_scope=None（尚未审核），unknown 保留给"已审核但无法判断"；
- 边界审核决定 schema v2：每个 high/medium 候选必须显式 boundary / no_boundary，
  自动合并项同样需要决定；low 候选与人工新边界经 adds 提升；
- boundary_review_status=approved 只表示**边界冻结**；SceneDocument.review_status 保持
  draft（人物/时间状态/知识内容待 P4 审核），这些场景不得直接进入正式索引。

详见 docs/research/KISAKI_GAME_RAG_SCENES.md。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from knowledge.game_rag.models import (
    RealityStatus,
    ReviewStatus,
    SceneDocument,
    ScriptSegment,
    SegmentType,
    SourceSpan,
    StoryContext,
)
from knowledge.game_rag.story_units import SOURCE_PREFIX, STORY_UNITS, StoryUnit, split_segments_by_unit

# ---------- 边界信号与置信度（集中定义，稳定字符串） ----------

SIGNAL_TIME_JUMP = "time_jump"  # 明确时间跳跃（次日/数日后/与此同时…）→ high，生效
SIGNAL_TIME_OF_DAY = "time_of_day"  # 时段/场景词（清晨/傍晚/深夜…）→ medium，生效
SIGNAL_BLANK_LINE = "blank_line"  # 段间空行（行号间隔 ≥2）→ medium，生效
SIGNAL_TRANSITION = "transition_marker"  # 整行分隔线或 ※ 注记行 → high，生效
SIGNAL_LONG_NARRATION = "long_narration_gap"  # 连续 ≥5 行叙述 → low，仅记录不生效

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

DECISION_BOUNDARY = "boundary"
DECISION_NO_BOUNDARY = "no_boundary"
_VALID_DECISIONS = {DECISION_BOUNDARY, DECISION_NO_BOUNDARY}

DEFAULT_MIN_DIALOGUE_TURNS = 6
LONG_NARRATION_MIN_LINES = 5
OVERSIZED_SPAN_LINES = 300
OVERSIZED_DIALOGUE_TURNS = 150

# 扩展自 scripts/extract_character_dialogues.py 的 SCENE_RESET_RE（经全语料验证的词表）
_TIME_JUMP_RE = re.compile(
    r"^(?:次日|翌日|第二天|数日后|几天后|几日后|过了几天|隔天|隔日|"
    r"数周后|几周后|一周后|数月后|几个月后|数年后|几年后|多年后|"
    r"与此同时|另一方面|同一时间|回忆的讲述(?:翻到下一页|已经结束))"
)
_TIME_OF_DAY_RE = re.compile(
    r"^(?:当天(?:早上|上午|下午|晚上|夜晚)?|当晚|那一夜|那天|放学后|"
    r"清晨|早晨|傍晚|黄昏|夜晚|深夜|夜里|梦中)"
)
# 转场标记必须整行成分隔线，或以 ※ 开头（制作注记行）。
# 全语料实测：整行分隔线 0 条、※ 行 1 条（日后谈:39）；
# 322 条「——」前缀行是行文破折号而非转场，前缀匹配会造成大量误报（P3 修正）。
_TRANSITION_RE = re.compile(r"^\s*(?:——+|――+|~~~+|\*{3,}|＝{3,}|={3,})\s*$")
_NOTE_MARKER_RE = re.compile(r"^\s*※")

# 去重优先级：生效 > 不生效；high > medium > low
_SIGNAL_PRIORITY: dict[str, tuple[int, int]] = {
    SIGNAL_TIME_JUMP: (1, 3),
    SIGNAL_TRANSITION: (1, 3),
    SIGNAL_TIME_OF_DAY: (1, 2),
    SIGNAL_BLANK_LINE: (1, 2),
    SIGNAL_LONG_NARRATION: (0, 1),
}

_TRIGGER_LIMIT = 30


@dataclass(frozen=True)
class BoundaryCandidate:
    """一个场景边界候选：锚点行号 = 新场景起始物理行。"""

    unit_id: str
    source_path: str
    line: int
    signal: str
    confidence: str
    effective_default: bool  # 是否默认自动生效（low 候选为 False）
    trigger_text: str


@dataclass
class ScenePlan:
    """边界解析结果：最终生效锚点 + 审计信息。"""

    boundaries: list[int] = field(default_factory=list)  # 排序后的生效锚点
    auto_merged: list[int] = field(default_factory=list)  # 因低于最小轮数被合并的锚点
    overridden_removed: list[int] = field(default_factory=list)  # 人工移除（预览模式）
    overridden_added: list[int] = field(default_factory=list)  # 人工新增（预览模式；决定模式=adds）
    unit_id: str = ""


# ---------- 检测 ----------


def detect_scene_boundaries(unit: StoryUnit, segments: list[ScriptSegment]) -> list[BoundaryCandidate]:
    """在一个故事单元内检测边界候选（只扫叙述行与段间隙，绝不锚进 dialogue 内部）。"""
    if not segments:
        return []
    source_path = unit.source_path
    unit_first_line = segments[0].source.line_start
    by_line: dict[int, BoundaryCandidate] = {}

    def _put(candidate: BoundaryCandidate) -> None:
        if candidate.line <= unit_first_line:
            return  # 单元首段前不切分
        existing = by_line.get(candidate.line)
        if existing is None or _SIGNAL_PRIORITY[candidate.signal] > _SIGNAL_PRIORITY[existing.signal]:
            by_line[candidate.line] = candidate

    # 信号 1：段间空行（行号间隔 ≥2，P2.1 已证明间隔只能来自空行）
    for cur, nxt in zip(segments, segments[1:]):
        gap = nxt.source.line_start - cur.source.line_end
        if gap >= 2:
            first_line = nxt.text.split("\n", 1)[0].strip()
            _put(
                BoundaryCandidate(
                    unit_id=unit.unit_id,
                    source_path=source_path,
                    line=nxt.source.line_start,
                    signal=SIGNAL_BLANK_LINE,
                    confidence=CONFIDENCE_MEDIUM,
                    effective_default=True,
                    trigger_text=first_line[:_TRIGGER_LIMIT],
                )
            )

    # 信号 2/3/4：叙述段逐行扫描（dialogue 段整体跳过，含其吞并的叙述行）
    for seg in segments:
        if seg.segment_type is not SegmentType.narration:
            continue
        lines = seg.text.split("\n")
        span_lines = seg.source.line_end - seg.source.line_start + 1
        if span_lines >= LONG_NARRATION_MIN_LINES:
            _put(
                BoundaryCandidate(
                    unit_id=unit.unit_id,
                    source_path=source_path,
                    line=seg.source.line_start,
                    signal=SIGNAL_LONG_NARRATION,
                    confidence=CONFIDENCE_LOW,
                    effective_default=False,
                    trigger_text=lines[0].strip()[:_TRIGGER_LIMIT],
                )
            )
        for idx, line_text in enumerate(lines):
            line_no = seg.source.line_start + idx
            stripped = line_text.strip()
            signal: str | None = None
            confidence: str | None = None
            if _TIME_JUMP_RE.match(stripped):
                signal, confidence = SIGNAL_TIME_JUMP, CONFIDENCE_HIGH
            elif _TIME_OF_DAY_RE.match(stripped):
                signal, confidence = SIGNAL_TIME_OF_DAY, CONFIDENCE_MEDIUM
            elif _TRANSITION_RE.match(line_text) or _NOTE_MARKER_RE.match(line_text):
                signal, confidence = SIGNAL_TRANSITION, CONFIDENCE_HIGH
            if signal is not None:
                _put(
                    BoundaryCandidate(
                        unit_id=unit.unit_id,
                        source_path=source_path,
                        line=line_no,
                        signal=signal,
                        confidence=confidence or "",
                        effective_default=True,
                        trigger_text=stripped[:_TRIGGER_LIMIT],
                    )
                )

    return [by_line[line] for line in sorted(by_line)]


# ---------- 边界解析 ----------


def _scene_index(anchors: list[int], line_start: int) -> int:
    """行号所属场景下标：锚点行本身属于新场景（计数 ≤ line_start 的锚点）。"""
    return bisect_right(anchors, line_start)


def _dialogue_turns_per_scene(segments: list[ScriptSegment], anchors: list[int]) -> list[int]:
    """按锚点统计各场景的 dialogue 段数（叙述内部切分不影响台词计数）。"""
    if not anchors:
        return [sum(1 for s in segments if s.segment_type is SegmentType.dialogue)]
    turns = [0] * (len(anchors) + 1)
    for seg in segments:
        if seg.segment_type is SegmentType.dialogue:
            turns[_scene_index(anchors, seg.source.line_start)] += 1
    return turns


def _validate_added_anchor(anchor: int, segments: list[ScriptSegment], unit_first_line: int) -> None:
    """add 锚点必须落在可切分位置：任意段起始行或叙述段内部行。

    锚进 dialogue 内部或落在空白行（不属于任何段）均抛错。
    """
    if anchor <= unit_first_line:
        raise ValueError(f"add 锚点 L{anchor} 不合法：不得位于单元首段之前")
    for seg in segments:
        start, end = seg.source.line_start, seg.source.line_end
        if anchor == start:
            return
        if seg.segment_type is SegmentType.narration and start < anchor <= end:
            return
        if seg.segment_type is SegmentType.dialogue and start < anchor <= end:
            raise ValueError(f"add 锚点 L{anchor} 不合法：落在 dialogue 段内部（L{start}-{end}）")
    raise ValueError(f"add 锚点 L{anchor} 不合法：不在任何已解析段的覆盖范围内（空白行请改用后一段起始行）")


def plan_scene_boundaries(
    unit: StoryUnit,
    segments: list[ScriptSegment],
    candidates: list[BoundaryCandidate],
    *,
    overrides: dict | None = None,
    min_dialogue_turns: int = DEFAULT_MIN_DIALOGUE_TURNS,
) -> ScenePlan:
    """草稿预览模式：自动生效候选 → 覆盖 → 最小轮数合并。

    该计划仅供审核预览；approved 冻结必须走 plan_scene_boundaries_from_decisions。
    """
    decisions = (overrides or {}).get("unit_decisions", {}).get(unit.unit_id, {})
    remove = {int(x) for x in decisions.get("remove", [])}
    add = {int(x) for x in decisions.get("add", [])}
    unit_first_line = segments[0].source.line_start if segments else 0
    for anchor in sorted(add):
        _validate_added_anchor(anchor, segments, unit_first_line)

    base = {c.line for c in candidates if c.effective_default}
    base |= add
    base -= remove
    protected = add - remove

    plan = ScenePlan(
        boundaries=sorted(base),
        auto_merged=[],
        overridden_removed=sorted(remove & {c.line for c in candidates}),
        overridden_added=sorted(add),
        unit_id=unit.unit_id,
    )
    if min_dialogue_turns <= 0 or not base:
        return plan

    merged: list[int] = []
    changed = True
    while changed:
        changed = False
        anchors = sorted(base)
        turns = _dialogue_turns_per_scene(segments, anchors)
        # 场景 idx（0..n）：scene_{idx>=1} 由 anchors[idx-1] 开启；
        # 移除 anchors[j] 即合并 scene_j 与 scene_{j+1}。
        for idx, count in enumerate(turns):
            if count >= min_dialogue_turns:
                continue
            if idx >= 1 and anchors[idx - 1] not in protected:
                drop = anchors[idx - 1]  # 小场景并入前一场景
            elif idx == 0 and len(anchors) >= 1 and anchors[0] not in protected:
                drop = anchors[0]  # 首场景过小：与后一场景合并
            elif idx < len(anchors) and anchors[idx] not in protected:
                drop = anchors[idx]  # 起点受保护：改为与后一场景合并
            else:
                continue  # 两侧边界均受保护：保留小场景（人工决定优先）
            base.discard(drop)
            merged.append(drop)
            changed = True
            break
    plan.boundaries = sorted(base)
    plan.auto_merged = sorted(set(merged))
    return plan


def plan_scene_boundaries_from_decisions(
    unit: StoryUnit,
    segments: list[ScriptSegment],
    candidates: list[BoundaryCandidate],
    overrides_doc: dict,
) -> ScenePlan:
    """决定模式：边界完全由 approved 决定生成（decisions.boundary + adds），不做自动合并。"""
    decisions = overrides_doc.get("candidate_decisions", {}).get(unit.unit_id, {})
    adds = {int(x) for x in overrides_doc.get("adds", {}).get(unit.unit_id, [])}
    keep = {int(line) for line, dec in decisions.items() if dec == DECISION_BOUNDARY}
    boundaries = sorted(keep | adds)
    return ScenePlan(
        boundaries=boundaries,
        auto_merged=[],
        overridden_removed=sorted(int(line) for line, dec in decisions.items() if dec == DECISION_NO_BOUNDARY),
        overridden_added=sorted(adds),
        unit_id=unit.unit_id,
    )


# ---------- SceneDocument 组装 ----------


def _scene_id(source_path: str, line_start: int, line_end: int) -> str:
    digest = hashlib.sha256(f"{source_path}|scene|{line_start}|{line_end}".encode()).hexdigest()
    return f"scene_{digest[:16]}"


@dataclass
class _Piece:
    """原子片段：dialogue 整段一块；narration 可按内部锚点切分。仅承载行号与说话人。"""

    line_start: int
    line_end: int
    speaker: str | None


def _split_into_pieces(segments: list[ScriptSegment], anchors: list[int]) -> list[_Piece]:
    pieces: list[_Piece] = []
    for seg in segments:
        start, end = seg.source.line_start, seg.source.line_end
        if seg.segment_type is SegmentType.dialogue:
            pieces.append(_Piece(start, end, seg.speaker))
            continue
        interior = [a for a in anchors if start < a <= end]
        if not interior:
            pieces.append(_Piece(start, end, None))
            continue
        cursor = 0
        piece_start = start
        for anchor in interior + [end + 1]:
            cut = anchor - start
            chunk_lines = cut - cursor
            if chunk_lines > 0:
                pieces.append(_Piece(piece_start, piece_start + chunk_lines - 1, None))
            cursor = cut
            piece_start = start + cut
    return pieces


def build_scene_documents(
    unit: StoryUnit,
    segments: list[ScriptSegment],
    plan: ScenePlan,
    *,
    source_lines: list[str],
) -> list[SceneDocument]:
    """按最终边界组装 SceneDocument（内存对象，不写盘）。

    P3.1：text 取场景行号范围内的**原文物理行逐字拼接**（LF 规范化后），
    dialogue 的 [说话人] 标签、空行、未闭合台词、重复闭引号全部原样保留。
    """
    if not segments:
        return []
    anchors = plan.boundaries
    pieces = _split_into_pieces(segments, anchors)
    groups: list[list[_Piece]] = [[] for _ in range(len(anchors) + 1)]
    for piece in pieces:
        groups[_scene_index(anchors, piece.line_start)].append(piece)

    documents: list[SceneDocument] = []
    for group in groups:
        if not group:
            continue
        line_start = group[0].line_start
        line_end = group[-1].line_end
        speakers = sorted({p.speaker for p in group if p.speaker})
        text = "\n".join(source_lines[line_start - 1 : line_end])
        documents.append(
            SceneDocument(
                id=_scene_id(unit.source_path, line_start, line_end),
                title=f"{unit.story_title} L{line_start}-{line_end}",
                text=text,
                story=StoryContext(
                    volume_number=unit.volume_number,
                    story_unit_id=unit.unit_id,
                    story_title=unit.story_title,
                    continuity_id=None,
                    sequence_order=None,
                    viewpoint=unit.viewpoint,
                    content_scope=unit.content_scope,
                    temporal_scope=None,  # P3.1：None=尚未审核；unknown 留给已审核但无法判断
                    route=None,
                ),
                source=SourceSpan(source_path=unit.source_path, line_start=line_start, line_end=line_end),
                speakers=speakers,
                mentioned_characters=[],  # P4 提取
                present_characters=[],  # P4 提取
                reality_status=RealityStatus.unknown,
                review_status=ReviewStatus.draft,  # 边界冻结≠内容审核通过；正式索引需 P4 后 approved
            )
        )
    return documents


def _load_source_lines(game_root: Path, source_path: str, source_prefix: str) -> list[str]:
    """读取源文件并做 LF 规范化（与解析器唯一允许的文本规范化一致）。"""
    from knowledge.game_rag.parser import _split_lines

    prefix = f"{source_prefix}/"
    rel = source_path[len(prefix) :] if source_path.startswith(prefix) else source_path
    return _split_lines((game_root / rel).read_text(encoding="utf-8"))


# ---------- 决定验证（approved 门禁） ----------


def _parse_canonical_line(value: object) -> int | None:
    """解析规范十进制行号：接受 int 或 '123' 形式 str；'01'/' 1'/'+1'/1.0 等一律拒绝。

    规范化拒绝的动机：'01' 与 '1' 若都解析为 1，会绕过"同一行重复决定"的键唯一性
    （JSON 对象中它们是两个不同键），造成冲突决定被静默接受。
    """
    if isinstance(value, bool):  # bool 是 int 子类，显式排除
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit() and (value == "0" or not value.startswith("0")):
        return int(value)
    return None


def validate_boundary_overrides(
    overrides_doc: dict,
    grouped: dict[str, list[ScriptSegment]],
    candidates_by_unit: dict[str, list[BoundaryCandidate]],
    *,
    expected_min_dialogue_turns: int | None = None,
    allow_draft: bool = False,
) -> list[str]:
    """验证边界决定文档，返回错误列表（空 = 通过）。

    检查项：
    1. schema_version == 2（P3.2：版本门禁）；
    2. boundary_review_status == "approved"（allow_draft=True 时也接受 "draft"，
       供决定齐备但尚未批准冻结的中间环节使用，如 oversized 审核包重建）；
    3. reviewer 非空；
    4. candidate_decisions / adds 覆盖全部 18 个登记单元且无未知单元；
    5. 结构校验：candidate_decisions 及各单元决定必须是对象，adds 各项必须是数组；
    6. 所有 high/medium 候选均有合法决定（boundary / no_boundary，null 即缺失）；
    7. 决定中的行号确实属于该单元的 high/medium 候选；行号必须为规范十进制（"01"/"+1" 拒绝）；
    8. add 行号可切分且不在 dialogue/空白行内部，且必须为规范十进制；
    9. 同一行不得同时出现在决定与 add 中（冲突）；
    10. min_dialogue_turns 与生成审核包时一致（提供期望值时）。
    """
    errors: list[str] = []
    status = overrides_doc.get("boundary_review_status")
    allowed_status = ("approved", "draft") if allow_draft else ("approved",)
    if status not in allowed_status:
        expected = " 或 ".join(repr(s) for s in allowed_status)
        errors.append(f"boundary_review_status 必须为 {expected}，当前为 {status!r}")
        return errors

    if overrides_doc.get("schema_version") != 2:
        errors.append(f"schema_version 必须为 2（P3.1 决定文件格式），当前为 {overrides_doc.get('schema_version')!r}")

    decisions_map = overrides_doc.get("candidate_decisions", {})
    adds_map = overrides_doc.get("adds", {})
    structure_invalid = False
    if not isinstance(decisions_map, dict):
        errors.append(f"candidate_decisions 必须是对象，当前为 {type(decisions_map).__name__}")
        decisions_map = {}
        structure_invalid = True
    if not isinstance(adds_map, dict):
        errors.append(f"adds 必须是对象，当前为 {type(adds_map).__name__}")
        adds_map = {}
        structure_invalid = True
    for uid, unit_decisions in decisions_map.items():
        if not isinstance(unit_decisions, dict):
            errors.append(f"{uid}: 单元决定必须是对象，当前为 {type(unit_decisions).__name__}")
            structure_invalid = True
    for uid, unit_adds in adds_map.items():
        if not isinstance(unit_adds, list):
            errors.append(f"{uid}: adds 必须是数组，当前为 {type(unit_adds).__name__}")
            structure_invalid = True
    if structure_invalid:
        return errors  # 结构错误时不继续逐行校验，避免类型异常

    reviewer = str(overrides_doc.get("reviewer", "") or "").strip()
    if not reviewer:
        errors.append("reviewer 不得为空")

    known_units = {u.unit_id for u in STORY_UNITS}
    for label, mapping in (("candidate_decisions", decisions_map), ("adds", adds_map)):
        keys = set(mapping)
        missing = sorted(known_units - keys)
        unknown = sorted(keys - known_units)
        if missing:
            errors.append(f"{label} 缺少单元: {missing}")
        if unknown:
            errors.append(f"{label} 含未知单元: {unknown}")

    if expected_min_dialogue_turns is not None:
        actual = overrides_doc.get("min_dialogue_turns")
        if actual != expected_min_dialogue_turns:
            errors.append(f"min_dialogue_turns={actual!r} 与审核包生成参数 {expected_min_dialogue_turns!r} 不一致")

    for unit in STORY_UNITS:
        unit_id = unit.unit_id
        segments = grouped.get(unit_id, [])
        candidates = candidates_by_unit.get(unit_id, [])
        must_review = {c.line for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM)}
        decisions_raw = decisions_map.get(unit_id, {})
        adds_raw = adds_map.get(unit_id, [])

        decided: dict[int, str] = {}
        for key, value in decisions_raw.items():
            line = _parse_canonical_line(key)
            if line is None:
                errors.append(f"{unit_id}: 决定行号 {key!r} 不是规范十进制整数（如 '01'/' 1'/'+1' 拒绝）")
                continue
            if value is None:
                continue  # null = 未决定，由下方 must_review 检查统一报缺失
            if value not in _VALID_DECISIONS:
                errors.append(f"{unit_id} L{line}: 决定值 {value!r} 非法（须为 boundary/no_boundary）")
                continue
            decided[line] = value

        for line in sorted(must_review):
            if line not in decided:
                errors.append(f"{unit_id} L{line}: high/medium 候选缺少明确决定（null 或缺失）")
        for line in sorted(decided):
            if line not in must_review:
                errors.append(f"{unit_id} L{line}: 决定的行号不属于该单元的 high/medium 候选")

        add_lines: set[int] = set()
        parsed_add_count = 0
        for value in adds_raw:
            line = _parse_canonical_line(value)
            if line is None:
                errors.append(f"{unit_id}: add 行号 {value!r} 不是规范十进制整数（如 '01'/' 1'/'+1' 拒绝）")
                continue
            parsed_add_count += 1
            add_lines.add(line)
        if parsed_add_count != len(add_lines):
            errors.append(f"{unit_id}: adds 含重复行号")
        conflicts = sorted(add_lines & set(decided))
        if conflicts:
            errors.append(f"{unit_id}: 行 {conflicts} 同时出现在决定与 add 中，存在冲突")
        unit_first_line = segments[0].source.line_start if segments else 0
        for anchor in sorted(add_lines):
            if anchor <= unit_first_line:
                errors.append(f"{unit_id}: add 锚点 L{anchor} 不合法：不得位于单元首段之前")
                continue
            try:
                _validate_added_anchor(anchor, segments, unit_first_line)
            except ValueError as exc:
                errors.append(f"{unit_id}: {exc}")
    return errors


# ---------- 冻结入口（从原始语料重建） ----------


def _freeze_scenes_from_overrides(
    game_root: Path,
    overrides_doc: dict,
    out_dir: Path,
    *,
    review_dir: Path | None = None,
) -> dict:
    """内部冻结实现；公开调用必须经过 low_review.freeze_reviewed_scenes。

    - 写出 scenes.jsonl（SceneDocument，review_status 仍为 draft）与 boundary_manifest.json；
    - 不接收现成场景列表：任何场景都必须由当前语料 + 决定重新构建；
    - review_dir 提供时（含 boundary_stats.json），校验 min_dialogue_turns 与审核包一致。
    """
    from knowledge.game_rag.parser import parse_script_directory

    if overrides_doc.get("boundary_review_status") != "approved":
        raise ValueError(
            f"boundary_review_status 必须为 approved，当前为 {overrides_doc.get('boundary_review_status')!r}"
        )
    source_prefix = str(overrides_doc.get("source_prefix") or SOURCE_PREFIX)
    segments = parse_script_directory(game_root, source_prefix=source_prefix)
    grouped = split_segments_by_unit(segments)
    candidates_by_unit = {unit.unit_id: detect_scene_boundaries(unit, grouped[unit.unit_id]) for unit in STORY_UNITS}

    expected_min: int | None = None
    if review_dir is not None:
        stats = json.loads((review_dir / "boundary_stats.json").read_text(encoding="utf-8"))
        expected_min = stats.get("min_dialogue_turns")

    errors = validate_boundary_overrides(
        overrides_doc, grouped, candidates_by_unit, expected_min_dialogue_turns=expected_min
    )
    if errors:
        raise ValueError("边界审核决定未通过验证:\n- " + "\n- ".join(errors))

    all_scenes: list[SceneDocument] = []
    units_manifest: dict[str, dict] = {}
    source_lines_cache: dict[str, list[str]] = {}
    for unit in STORY_UNITS:
        unit_segments = grouped[unit.unit_id]
        plan = plan_scene_boundaries_from_decisions(
            unit, unit_segments, candidates_by_unit[unit.unit_id], overrides_doc
        )
        if unit.source_path not in source_lines_cache:
            source_lines_cache[unit.source_path] = _load_source_lines(game_root, unit.source_path, source_prefix)
        scenes = build_scene_documents(unit, unit_segments, plan, source_lines=source_lines_cache[unit.source_path])
        all_scenes.extend(scenes)
        decisions = overrides_doc.get("candidate_decisions", {}).get(unit.unit_id, {})
        units_manifest[unit.unit_id] = {
            "story_title": unit.story_title,
            "boundaries": plan.boundaries,
            "decisions": {str(k): v for k, v in sorted(decisions.items())},
            "adds": sorted(int(x) for x in overrides_doc.get("adds", {}).get(unit.unit_id, [])),
            "scenes": len(scenes),
        }

    manifest = {
        "schema_version": 1,
        "boundary_review_status": "approved",
        "reviewer": overrides_doc.get("reviewer"),
        "min_dialogue_turns": overrides_doc.get("min_dialogue_turns"),
        "source_prefix": source_prefix,
        "units": units_manifest,
        "total_scenes": len(all_scenes),
        "scene_review_status": "draft",
        "note": (
            "boundary_review_status=approved 仅表示场景边界已冻结；"
            "SceneDocument.review_status=draft 表示人物、时间状态与知识内容尚待 P4 审核，"
            "这些场景不得直接进入正式索引（正式索引要求 P4 后 review_status=approved）。"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _freeze_pair_atomic(
        out_dir,
        scenes_payload="".join(scene.model_dump_json() + "\n" for scene in all_scenes),
        manifest_payload=json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def _rollback_scenes_after_commit(scenes_path: Path, scenes_backup: Path, *, had_old_scenes: bool) -> bool:
    """把旧版 scenes 恢复原位（覆盖已提交的新版本），返回回滚是否成功。

    - 有旧版本：备份移回原位；
    - 无旧版本（首次冻结）：删除已提交的新 scenes 即视为回滚成功；
    - 任何 OSError 都视为回滚失败：调用方必须**保留备份**（唯一可恢复副本）并报告路径。
    """
    try:
        if had_old_scenes:
            os.replace(scenes_backup, scenes_path)
        else:
            scenes_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _freeze_pair_atomic(out_dir: Path, *, scenes_payload: str, manifest_payload: str) -> None:
    """两份冻结文件的整体原子提交（P3.2）。

    提交协议：
    1. 先把两份内容**全部**写入各自 .tmp（任何写失败都发生在提交前，旧文件未动）；
    2. 两份 tmp 均就绪后进入提交阶段：先替换 scenes.jsonl，再替换 boundary_manifest.json；
    3. manifest 最后替换，作为本次冻结完成标志；
    4. 第二次替换失败时**回滚** scenes.jsonl 到旧版本（备份于 .tmp.old），
       不得留下"新 scenes + 旧 manifest"的混合版本；
    5. 回滚也失败属极端情况（磁盘/权限双重故障）：**保留备份文件**（旧版本唯一
       可恢复副本）并在错误信息中报告其路径；只有提交成功或回滚成功后才删除备份。
    """
    scenes_path = out_dir / "scenes.jsonl"
    manifest_path = out_dir / "boundary_manifest.json"
    scenes_tmp = scenes_path.with_suffix(scenes_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    scenes_backup = scenes_path.with_suffix(scenes_path.suffix + ".tmp.old")

    if scenes_backup.exists():
        raise ValueError(f"检测到未恢复的旧版备份 {scenes_backup}；请先人工恢复或移走后重试")

    # 阶段 1：完整生成两份临时文件（失败时旧文件未动）
    try:
        scenes_tmp.write_text(scenes_payload, encoding="utf-8", newline="\n")
        manifest_tmp.write_text(manifest_payload, encoding="utf-8", newline="\n")
    except BaseException:
        for tmp in (scenes_tmp, manifest_tmp):
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        raise

    # 阶段 2：提交。scenes 先备份旧版本（若存在），替换，再替换 manifest
    had_old_scenes = scenes_path.exists()
    if had_old_scenes:
        try:
            os.replace(scenes_path, scenes_backup)
        except OSError:
            for tmp in (scenes_tmp, manifest_tmp):
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            raise
    committed = False
    rollback_done = False  # True=旧 scenes 已恢复原位（或确认无需恢复），备份可删除
    rollback_attempted = False
    try:
        os.replace(scenes_tmp, scenes_path)
        try:
            os.replace(manifest_tmp, manifest_path)
        except OSError as exc:
            # manifest 替换失败：回滚 scenes 到旧版本，不留混合版本
            rollback_attempted = True
            rollback_done = _rollback_scenes_after_commit(scenes_path, scenes_backup, had_old_scenes=had_old_scenes)
            if not rollback_done:
                # 瞬时文件占用可能只影响第一次恢复；明确重试一次后再判定最终状态。
                rollback_done = _rollback_scenes_after_commit(scenes_path, scenes_backup, had_old_scenes=had_old_scenes)
            if rollback_done:
                raise ValueError(
                    f"boundary_manifest.json 替换失败，已回滚 scenes.jsonl（不留混合版本）: {exc}"
                ) from exc
            raise ValueError(
                f"boundary_manifest.json 替换失败，且 scenes.jsonl 回滚失败；"
                f"旧版本备份保留于 {scenes_backup}（唯一可恢复副本，请手动恢复后重试）: {exc}"
            ) from exc
        committed = True
    except BaseException:
        # 阶段 2 其他失败（含 scenes 替换自身失败、上方 ValueError 传播）：
        # 同样恢复旧 scenes，清理 tmp
        if not rollback_attempted:
            rollback_done = _rollback_scenes_after_commit(scenes_path, scenes_backup, had_old_scenes=had_old_scenes)
        for tmp in (scenes_tmp, manifest_tmp):
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        raise
    finally:
        # 仅在提交成功或回滚成功后删除备份；
        # 回滚失败时备份是旧版本唯一可恢复副本，必须保留
        if committed or rollback_done:
            with contextlib.suppress(OSError):
                scenes_backup.unlink(missing_ok=True)


def _dump_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------- 审核材料生成（CLI） ----------


def _excerpt(line_map: dict[int, str], line: int, radius: int = 2) -> str:
    parts = []
    for no in range(line - radius, line + radius + 1):
        mark = ">>" if no == line else "  "
        text = line_map.get(no)
        shown = "（空行）" if text is None or not text.strip() else (text.strip()[:40])
        parts.append(f"  {mark}L{no} {shown}")
    return "\n".join(parts)


def generate_review_materials(
    game_root: Path,
    source_prefix: str,
    out_dir: Path,
    *,
    min_dialogue_turns: int = DEFAULT_MIN_DIALOGUE_TURNS,
) -> dict:
    """生成边界审核包。P3.1 策略：
    - high/medium 候选 100% 列入必审清单（含自动合并项，逐条显式决定）；
    - 超过 300 行或 150 轮的场景列入 oversized_scene_review，并优先展示其中的 low 候选；
    - 其余 low 候选分层抽样建议（隔 5 取 1）；
    - overrides 模板的待决定值为 null，禁止预填为通过。
    """
    from knowledge.game_rag.parser import parse_script_directory

    segments = parse_script_directory(game_root, source_prefix=source_prefix)
    grouped = split_segments_by_unit(segments)

    stats_units: dict[str, dict] = {}
    md: list[str] = [
        "# 场景边界审核包（P3.1）",
        "",
        f"- 生成参数：min_dialogue_turns={min_dialogue_turns}，oversized 阈值："
        f">{OVERSIZED_SPAN_LINES} 行或 >{OVERSIZED_DIALOGUE_TURNS} 轮",
        "- **必审清单**：全部 high/medium 候选（含 effective 与 auto_merged）需要逐条显式决定，"
        "null 视为未决定，approved 验证会拒绝",
        "- **oversized_scene_review**：超限场景（>300 行或 >150 轮）**内部的 low 候选逐条展示**"
        "（±2 行原文 + 提升决定栏），优先审核",
        "- **其余 low 候选抽样**：排除 oversized 场景内候选后的分层抽样建议（隔 5 取 1），两组互斥",
        "- 决定填写：overrides 中 candidate_decisions[unit][行号] = boundary / no_boundary；"
        "low 候选提升与人工新边界写入 adds[unit]",
        "- boundary_review_status=approved 仅冻结边界；场景内容（人物/时间状态）待 P4 审核",
        "",
        "## oversized_scene_review（超限场景）",
        "",
    ]
    sample_lines: list[str] = ["## low 候选分层抽样建议（隔 5 取 1）", ""]
    must_review_lines: list[str] = ["## 必审清单（high/medium 候选，100% 覆盖）", ""]

    for unit in STORY_UNITS:
        unit_segments = grouped[unit.unit_id]
        candidates = detect_scene_boundaries(unit, unit_segments)
        plan = plan_scene_boundaries(unit, unit_segments, candidates, min_dialogue_turns=min_dialogue_turns)
        source_lines = _load_source_lines(game_root, unit.source_path, source_prefix)
        scenes = build_scene_documents(unit, unit_segments, plan, source_lines=source_lines)
        line_map = {i: text for i, text in enumerate(source_lines, 1)}
        dialogues = sum(1 for s in unit_segments if s.segment_type is SegmentType.dialogue)
        scene_turn_counts = _dialogue_turns_per_scene(unit_segments, plan.boundaries)[: len(scenes)]
        merged_set = set(plan.auto_merged)

        must_review = {c.line: c for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM)}
        low_candidates = sorted((c for c in candidates if c.confidence == CONFIDENCE_LOW), key=lambda c: c.line)

        oversized: list[dict] = []
        oversized_low_lines: set[int] = set()
        for scene, turns in zip(scenes, scene_turn_counts):
            span = scene.source.line_end - scene.source.line_start + 1
            if span > OVERSIZED_SPAN_LINES or turns > OVERSIZED_DIALOGUE_TURNS:
                inner_low = [c for c in low_candidates if scene.source.line_start <= c.line <= scene.source.line_end]
                oversized_low_lines.update(c.line for c in inner_low)
                oversized.append(
                    {
                        "line_start": scene.source.line_start,
                        "line_end": scene.source.line_end,
                        "span_lines": span,
                        "dialogue_turns": turns,
                        "low_candidates": [c.line for c in inner_low],
                    }
                )

        by_signal: dict[str, int] = {}
        by_conf: dict[str, int] = {}
        for cand in candidates:
            by_signal[cand.signal] = by_signal.get(cand.signal, 0) + 1
            by_conf[cand.confidence] = by_conf.get(cand.confidence, 0) + 1
        spans = [s.source.line_end - s.source.line_start + 1 for s in scenes]
        stats_units[unit.unit_id] = {
            "story_title": unit.story_title,
            "source_path": unit.source_path,
            "content_scope": unit.content_scope.value,
            "volume_number": unit.volume_number,
            "line_range": [unit.line_start, unit.line_end],
            "segments": len(unit_segments),
            "dialogues": dialogues,
            "candidates_total": len(candidates),
            "candidates_by_signal": dict(sorted(by_signal.items())),
            "candidates_by_confidence": dict(sorted(by_conf.items())),
            "must_review_candidates": {
                str(line): {
                    "signal": cand.signal,
                    "confidence": cand.confidence,
                    "state": "auto_merged" if line in merged_set else "effective",
                }
                for line, cand in sorted(must_review.items())
            },
            "boundaries_effective": len(plan.boundaries),
            "boundaries_auto_merged": len(plan.auto_merged),
            "scenes": len(scenes),
            "scene_dialogue_turns": scene_turn_counts,
            "scene_span_lines_min": min(spans) if spans else 0,
            "scene_span_lines_max": max(spans) if spans else 0,
            "oversized_scenes": oversized,
            "oversized_low_candidate_count": len(oversized_low_lines),
            "remaining_low_candidate_count": len(low_candidates) - len(oversized_low_lines),
        }

        # 必审清单（全部 high/medium）
        must_review_lines.append(
            f"### {unit.unit_id}（{unit.story_title}｜L{unit.line_start}-{unit.line_end or 'EOF'}｜"
            f"必审 {len(must_review)}｜low {len(low_candidates)}｜预览场景 {len(scenes)}）"
        )
        must_review_lines.append("")
        for line, cand in sorted(must_review.items()):
            state = "auto_merged" if line in merged_set else "effective"
            must_review_lines.append(
                f"- L{line}｜{cand.signal}｜{cand.confidence}｜预览状态: {state}"
                f"｜触发: {cand.trigger_text or '（空行间隔）'}｜决定: ______"
            )
            must_review_lines.append(_excerpt(line_map, line))
            must_review_lines.append("")
        must_review_lines.append("")

        # oversized 场景：内部 low 候选逐条展示（±2 行原文 + add 决定栏）
        if oversized:
            md.append(f"### {unit.unit_id}（{unit.story_title}）")
            md.append("")
            for item in oversized:
                md.append(
                    f"#### 场景 L{item['line_start']}-{item['line_end']}｜{item['span_lines']} 行｜"
                    f"{item['dialogue_turns']} 轮"
                )
                md.append("")
                inner = [c for c in low_candidates if c.line in set(item["low_candidates"])]
                if not inner:
                    md.append("- （场景内无 low 候选；如需切分请人工指定边界行号）")
                    md.append("")
                    continue
                for cand in inner:
                    md.append(
                        f"- L{cand.line}｜{cand.signal}｜low｜触发: {cand.trigger_text or '（长叙述）'}"
                        f"｜提升为边界（写入 adds）: ______"
                    )
                    md.append(_excerpt(line_map, cand.line))
                    md.append("")
            md.append("")

        # 其余 low 候选抽样：排除 oversized 内的候选（两组互斥，P3.2）
        remaining_low = [c for c in low_candidates if c.line not in oversized_low_lines]
        picked = [c.line for c in remaining_low[::5]]
        if remaining_low:
            sample_lines.append(
                f"- {unit.unit_id} / {SIGNAL_LONG_NARRATION}: 候选 {len(remaining_low)}（已排除 "
                f"oversized 场景内 {len(oversized_low_lines)} 条），"
                f"建议抽检 L{', L'.join(map(str, picked)) or '无'}"
            )

    sample_lines.append("")
    md += sample_lines + ["", *must_review_lines]

    overrides_template = {
        "schema_version": 2,
        "boundary_review_status": "draft",
        "reviewer": "",
        "min_dialogue_turns": min_dialogue_turns,
        "source_prefix": source_prefix,
        "candidate_decisions": {
            unit.unit_id: {str(line): None for line in stats_units[unit.unit_id]["must_review_candidates"]}
            for unit in STORY_UNITS
        },
        "adds": {unit.unit_id: [] for unit in STORY_UNITS},
        "notes": "",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene_boundary_review.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    _dump_atomic_json(
        out_dir / "boundary_stats.json",
        {
            "schema_version": 2,
            "min_dialogue_turns": min_dialogue_turns,
            "oversized_thresholds": {"span_lines": OVERSIZED_SPAN_LINES, "dialogue_turns": OVERSIZED_DIALOGUE_TURNS},
            "units": stats_units,
        },
    )
    _dump_atomic_json(out_dir / "overrides_template.json", overrides_template)
    return stats_units


def generate_oversized_review_pack(
    game_root: Path,
    overrides_doc: dict,
    out_dir: Path,
    *,
    source_prefix: str | None = None,
) -> dict:
    """P3.3：按最终人工决定重建场景，生成新口径的 oversized 审核包。

    与 generate_review_materials（预览口径）不同：
    - 用 plan_scene_boundaries_from_decisions（决定模式，无自动合并）重建全部场景；
    - 允许 boundary_review_status=draft（本包生成先于 approved 冻结），但要求
      candidate_decisions 已覆盖全部 high/medium 候选（复用 allow_draft 校验）；
    - 在**新场景口径**下重新统计 oversized（>300 行或 >150 轮）及其内部 low 候选，
      旧预览口径（51 场景 / 567 low 候选）作废；
    - 输出 oversized_review.md（场景内 low 候选 ±2 行原文 + 提升决定栏）与
      oversized_stats.json；需要提升为边界的行号写入 overrides 的 adds[unit]。
    """
    from knowledge.game_rag.parser import parse_script_directory

    source_prefix = str(source_prefix or overrides_doc.get("source_prefix") or SOURCE_PREFIX)
    segments = parse_script_directory(game_root, source_prefix=source_prefix)
    grouped = split_segments_by_unit(segments)
    candidates_by_unit = {unit.unit_id: detect_scene_boundaries(unit, grouped[unit.unit_id]) for unit in STORY_UNITS}

    errors = validate_boundary_overrides(overrides_doc, grouped, candidates_by_unit, allow_draft=True)
    if errors:
        raise ValueError("边界审核决定未通过验证（重建 oversized 审核包要求决定完整）:\n- " + "\n- ".join(errors))

    header: list[str] = [
        "# oversized 场景审核包（最终决定口径，P3.3）",
        "",
        "- 本包按 candidate_decisions + adds（决定模式，无自动合并）重建场景后统计，",
        "  取代预览口径审核包中的 oversized_scene_review（旧口径作废）",
        f"- oversized 阈值：>{OVERSIZED_SPAN_LINES} 行或 >{OVERSIZED_DIALOGUE_TURNS} 轮",
        "- 场景内 low 候选逐条展示（±2 行原文）；需要提升为边界的行号写入 overrides 的 adds[unit]",
        "",
    ]
    body: list[str] = []
    stats_units: dict[str, dict] = {}
    total_scenes = total_oversized = total_low_in_oversized = 0

    for unit in STORY_UNITS:
        unit_segments = grouped[unit.unit_id]
        candidates = candidates_by_unit[unit.unit_id]
        plan = plan_scene_boundaries_from_decisions(unit, unit_segments, candidates, overrides_doc)
        source_lines = _load_source_lines(game_root, unit.source_path, source_prefix)
        scenes = build_scene_documents(unit, unit_segments, plan, source_lines=source_lines)
        line_map = {i: text for i, text in enumerate(source_lines, 1)}
        scene_turns = _dialogue_turns_per_scene(unit_segments, plan.boundaries)[: len(scenes)]
        low_candidates = sorted((c for c in candidates if c.confidence == CONFIDENCE_LOW), key=lambda c: c.line)

        oversized: list[dict] = []
        oversized_low_lines: set[int] = set()
        for scene, turns in zip(scenes, scene_turns):
            span = scene.source.line_end - scene.source.line_start + 1
            if span > OVERSIZED_SPAN_LINES or turns > OVERSIZED_DIALOGUE_TURNS:
                inner_low = [c for c in low_candidates if scene.source.line_start <= c.line <= scene.source.line_end]
                oversized_low_lines.update(c.line for c in inner_low)
                oversized.append(
                    {
                        "line_start": scene.source.line_start,
                        "line_end": scene.source.line_end,
                        "span_lines": span,
                        "dialogue_turns": turns,
                        "low_candidates": [c.line for c in inner_low],
                    }
                )

        stats_units[unit.unit_id] = {
            "story_title": unit.story_title,
            "boundaries": plan.boundaries,
            "scenes": len(scenes),
            "oversized_scenes": oversized,
            "oversized_low_candidate_count": len(oversized_low_lines),
            "remaining_low_candidate_count": len(low_candidates) - len(oversized_low_lines),
        }
        total_scenes += len(scenes)
        total_oversized += len(oversized)
        total_low_in_oversized += len(oversized_low_lines)

        if not oversized:
            continue
        body.append(f"### {unit.unit_id}（{unit.story_title}）")
        body.append("")
        for item in oversized:
            body.append(
                f"#### 场景 L{item['line_start']}-{item['line_end']}｜{item['span_lines']} 行｜"
                f"{item['dialogue_turns']} 轮"
            )
            body.append("")
            inner = [c for c in low_candidates if c.line in set(item["low_candidates"])]
            if not inner:
                body.append("- （场景内无 low 候选；如需切分请人工指定边界行号）")
                body.append("")
                continue
            for cand in inner:
                body.append(
                    f"- L{cand.line}｜{cand.signal}｜low｜触发: {cand.trigger_text or '（长叙述）'}"
                    f"｜提升为边界（写入 adds）: ______"
                )
                body.append(_excerpt(line_map, cand.line))
                body.append("")
        body.append("")

    header.insert(
        2,
        f"- 决定口径统计：场景 {total_scenes}｜oversized {total_oversized}｜其中 low 候选 {total_low_in_oversized}",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oversized_review.md").write_text("\n".join(header + body), encoding="utf-8", newline="\n")
    _dump_atomic_json(
        out_dir / "oversized_stats.json",
        {
            "schema_version": 1,
            "mode": "decisions",
            "oversized_thresholds": {
                "span_lines": OVERSIZED_SPAN_LINES,
                "dialogue_turns": OVERSIZED_DIALOGUE_TURNS,
            },
            "total_scenes": total_scenes,
            "total_oversized_scenes": total_oversized,
            "total_oversized_low_candidates": total_low_in_oversized,
            "units": stats_units,
        },
    )
    return stats_units


if __name__ == "__main__":  # pragma: no cover
    import argparse

    cli = argparse.ArgumentParser(description="P3 场景边界审核材料工具（不冻结 scenes）")
    sub = cli.add_subparsers(dest="command", required=True)

    p_preview = sub.add_parser("review-pack", help="生成预览口径审核包（P3.1）")
    p_preview.add_argument("--game-root", type=Path, required=True, help="gametext/纸上魔法使 目录")
    p_preview.add_argument("--source-prefix", default=SOURCE_PREFIX, help=f"便携溯源前缀（默认 {SOURCE_PREFIX}）")
    p_preview.add_argument("--out-dir", type=Path, required=True, help="审核材料输出目录")
    p_preview.add_argument("--min-dialogue-turns", type=int, default=DEFAULT_MIN_DIALOGUE_TURNS)

    p_oversized = sub.add_parser("oversized-pack", help="按最终决定重建场景并生成 oversized 审核包（P3.3）")
    p_oversized.add_argument("--game-root", type=Path, required=True, help="gametext/纸上魔法使 目录")
    p_oversized.add_argument(
        "--overrides", type=Path, required=True, help="boundary_overrides.json（draft 即可，决定须完整）"
    )
    p_oversized.add_argument("--out-dir", type=Path, required=True, help="审核材料输出目录")

    args = cli.parse_args()
    if args.command == "review-pack":
        units = generate_review_materials(
            args.game_root, args.source_prefix, args.out_dir, min_dialogue_turns=args.min_dialogue_turns
        )
        must_review_total = sum(len(u["must_review_candidates"]) for u in units.values())
        oversized_total = sum(len(u["oversized_scenes"]) for u in units.values())
        oversized_low_total = sum(u["oversized_low_candidate_count"] for u in units.values())
        remaining_low_total = sum(u["remaining_low_candidate_count"] for u in units.values())
        total_scenes = sum(u["scenes"] for u in units.values())
        print(
            f"units={len(units)} must_review={must_review_total} oversized_scenes={oversized_total} "
            f"oversized_low={oversized_low_total} remaining_low_sampled={remaining_low_total} "
            f"preview_scenes={total_scenes}"
        )
        print(f"材料已写入 {args.out_dir}（scene_boundary_review.md / boundary_stats.json / overrides_template.json）")
    else:
        doc = json.loads(args.overrides.read_text(encoding="utf-8"))
        units = generate_oversized_review_pack(args.game_root, doc, args.out_dir)
        oversized_total = sum(len(u["oversized_scenes"]) for u in units.values())
        oversized_low_total = sum(u["oversized_low_candidate_count"] for u in units.values())
        total_scenes = sum(u["scenes"] for u in units.values())
        print(
            f"units={len(units)} decision_scenes={total_scenes} "
            f"oversized_scenes={oversized_total} oversized_low={oversized_low_total}"
        )
        print(f"材料已写入 {args.out_dir}（oversized_review.md / oversized_stats.json）")
