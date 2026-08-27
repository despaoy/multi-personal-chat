"""Structured review state for low-confidence scene-boundary candidates.

职责独立于 scene_segmenter：只负责 low 候选审核状态的创建、校验、原子保存、
oversized 外候选的确定性分层抽样，以及向 boundary_overrides.json adds 的安全
合并；不实现边界检测或场景重建（那些属于 scene_segmenter）。
"""

from __future__ import annotations

import json
import os
import re
from bisect import bisect_right
from copy import deepcopy
from pathlib import Path
from typing import Any

from knowledge.game_rag.scene_segmenter import (
    CONFIDENCE_LOW,
    OVERSIZED_DIALOGUE_TURNS,
    OVERSIZED_SPAN_LINES,
    BoundaryCandidate,
    _dialogue_turns_per_scene,
    _freeze_scenes_from_overrides,
    _load_source_lines,
    build_scene_documents,
    detect_scene_boundaries,
    plan_scene_boundaries_from_decisions,
    validate_boundary_overrides,
)
from knowledge.game_rag.story_units import SOURCE_PREFIX, STORY_UNITS, split_segments_by_unit

LOW_REVIEW_SCHEMA_VERSION = 1
DECISION_BOUNDARY = "boundary"
DECISION_NO_BOUNDARY = "no_boundary"
VALID_LOW_DECISIONS = {DECISION_BOUNDARY, DECISION_NO_BOUNDARY}

# 分层抽检规则参数（确定性，供外部复核重现）
SAMPLE_STRIDE = 5  # 每层按行号排序后每 5 条抽取 1 条（含首条）
SAMPLE_NEAR_BOUNDARY = 3  # 距既有生效边界锚点 ±3 行内的外部候选优先纳入


def _canonical_line(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isascii() and value.isdigit() and str(int(value)) == value:
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def low_candidate_lines(
    candidates_by_unit: dict[str, list[BoundaryCandidate]],
) -> dict[str, set[int]]:
    """Return every low-confidence candidate line, grouped by registered unit."""
    return {
        unit.unit_id: {
            candidate.line
            for candidate in candidates_by_unit.get(unit.unit_id, [])
            if candidate.confidence == CONFIDENCE_LOW
        }
        for unit in STORY_UNITS
    }


def stratified_external_sample(
    external_lines_by_unit: dict[str, set[int]],
    boundaries_by_unit: dict[str, list[int]],
) -> dict[str, list[int]]:
    """对 oversized 场景外的 low 候选做确定性分层抽检。

    规则（写入状态文件供复核）：
    - 层 = (单元, 场景)；场景序号由候选行号对生效边界锚点数组 bisect 得出；
    - 每层内按行号排序后每 SAMPLE_STRIDE 条取 1 条（含首条，保证每层覆盖）；
    - 距任一生效边界锚点 ±SAMPLE_NEAR_BOUNDARY 行内的外部候选优先纳入。
    """
    sampled: dict[str, list[int]] = {}
    for unit in STORY_UNITS:
        uid = unit.unit_id
        lines = sorted(external_lines_by_unit.get(uid, set()))
        boundaries = sorted(boundaries_by_unit.get(uid, []))
        selected: set[int] = set()
        strata: dict[int, list[int]] = {}
        for line in lines:
            strata.setdefault(bisect_right(boundaries, line), []).append(line)
            if any(abs(line - anchor) <= SAMPLE_NEAR_BOUNDARY for anchor in boundaries):
                selected.add(line)
        for scene_lines in strata.values():
            selected.update(scene_lines[::SAMPLE_STRIDE])
        sampled[uid] = sorted(selected)
    return sampled


def create_low_review_document(
    candidates_by_unit: dict[str, list[BoundaryCandidate]],
    overrides_doc: dict[str, Any],
    *,
    in_oversized_by_unit: dict[str, set[int]] | None = None,
    external_sample_by_unit: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Create a resumable review document, preserving low boundaries already in adds.

    - candidate_decisions 覆盖全部 low 候选（含 oversized 内外），初始 null；
      已存在于 overrides adds 的 low 候选预填 boundary；
    - oversized_scope 记录必审范围快照（579 条口径）；
    - external_sampling 记录抽样规则与抽中行号（快照，不随后续边界变化）；
    - replacement_adds 记录“候选本身 no_boundary 但附近存在准确锚点”的替代行。
    """
    candidate_lines = low_candidate_lines(candidates_by_unit)
    in_oversized = in_oversized_by_unit or {}
    sample = external_sample_by_unit or {}
    decisions: dict[str, dict[str, str | None]] = {}
    replacement_adds: dict[str, list[int]] = {}
    for unit in STORY_UNITS:
        uid = unit.unit_id
        existing_adds = {
            int(line) for line in overrides_doc.get("adds", {}).get(uid, []) if _canonical_line(line) is not None
        }
        decisions[uid] = {
            str(line): DECISION_BOUNDARY if line in existing_adds else None for line in sorted(candidate_lines[uid])
        }
        replacement_adds[uid] = []

    return {
        "schema_version": LOW_REVIEW_SCHEMA_VERSION,
        "review_status": "draft",
        "reviewer": str(overrides_doc.get("reviewer", "") or ""),
        "oversized_scope": {unit.unit_id: sorted(in_oversized.get(unit.unit_id, set())) for unit in STORY_UNITS},
        "external_sampling": {
            "rule": (
                f"stratified by (unit, scene); within each stratum sort by line and take every "
                f"{SAMPLE_STRIDE}-th candidate (including the first); additionally include external "
                f"candidates within +/-{SAMPLE_NEAR_BOUNDARY} lines of any effective boundary anchor"
            ),
            "sampled_lines": {unit.unit_id: sorted(sample.get(unit.unit_id, [])) for unit in STORY_UNITS},
        },
        "candidate_decisions": decisions,
        "replacement_adds": replacement_adds,
        "reasons": {},
        "replacement_reasons": {},
        "notes_by_unit": {},
        "notes": (
            "Low-confidence scene-boundary review state. boundary decisions and replacement_adds "
            "merge into boundary_overrides.json adds; no_boundary decisions remain here for "
            "resumability. oversized_scope and external_sampling are frozen snapshots of the "
            "79-scene decision caliber taken at document creation."
        ),
    }


def validate_low_review_document(
    review_doc: dict[str, Any] | None,
    candidates_by_unit: dict[str, list[BoundaryCandidate]],
    *,
    require_complete: bool = False,
    expected_oversized_by_unit: dict[str, set[str]] | None = None,
) -> list[str]:
    """Validate shape, candidate coverage, decisions, scopes, and replacement anchors.

    require_complete=True 时额外要求：oversized_scope 与 external_sampling 抽中行
    全部有明确决定（人工审核完成门禁）。
    """
    errors: list[str] = []
    if review_doc.get("schema_version") != LOW_REVIEW_SCHEMA_VERSION:
        errors.append(f"schema_version must be {LOW_REVIEW_SCHEMA_VERSION}, got {review_doc.get('schema_version')!r}")
    if review_doc.get("review_status") not in ("draft", "approved"):
        errors.append("review_status must be 'draft' or 'approved'")
    if not str(review_doc.get("reviewer", "") or "").strip():
        errors.append("reviewer must not be empty")

    decisions_map = review_doc.get("candidate_decisions")
    if not isinstance(decisions_map, dict):
        return [*errors, "candidate_decisions must be an object"]

    expected = low_candidate_lines(candidates_by_unit)
    expected_units = set(expected)
    actual_units = set(decisions_map)
    if missing := sorted(expected_units - actual_units):
        errors.append(f"candidate_decisions missing units: {missing}")
    if unknown := sorted(actual_units - expected_units):
        errors.append(f"candidate_decisions contains unknown units: {unknown}")

    parsed_by_unit: dict[str, dict[int, str | None]] = {}
    for unit_id, expected_lines in expected.items():
        raw_decisions = decisions_map.get(unit_id, {})
        if not isinstance(raw_decisions, dict):
            errors.append(f"{unit_id}: decisions must be an object")
            continue

        parsed: dict[int, str | None] = {}
        for key, value in raw_decisions.items():
            line = _canonical_line(key)
            if line is None:
                errors.append(f"{unit_id}: line {key!r} is not a canonical positive integer")
                continue
            if value is not None and value not in VALID_LOW_DECISIONS:
                errors.append(f"{unit_id} L{line}: invalid decision {value!r}")
                continue
            parsed[line] = value
        parsed_by_unit[unit_id] = parsed

        actual_lines = set(parsed)
        if missing_lines := sorted(expected_lines - actual_lines):
            errors.append(f"{unit_id}: missing low candidates {missing_lines}")
        if unknown_lines := sorted(actual_lines - expected_lines):
            errors.append(f"{unit_id}: unknown low candidates {unknown_lines}")

    # boundary 决定必须可追溯到非空理由。no_boundary 理由允许逐步补录。
    reasons_map = review_doc.get("reasons", {})
    if not isinstance(reasons_map, dict):
        errors.append("reasons must be an object")
        reasons_map = {}
    elif unknown := sorted(set(reasons_map) - expected_units):
        errors.append(f"reasons contains unknown units: {unknown}")
    for unit_id in expected_units:
        unit_reasons = reasons_map.get(unit_id, {})
        if not isinstance(unit_reasons, dict):
            errors.append(f"{unit_id}: reasons must be an object")
            continue
        for key, reason in unit_reasons.items():
            line = _canonical_line(key)
            if line is None:
                errors.append(f"{unit_id}: reason line {key!r} is not a canonical positive integer")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{unit_id} L{key}: reason must be a non-empty string")
        for line, decision in parsed_by_unit.get(unit_id, {}).items():
            if decision == DECISION_BOUNDARY and not str(unit_reasons.get(str(line), "") or "").strip():
                errors.append(f"{unit_id} L{line}: boundary decision requires a non-empty reason")

    # replacement_adds：结构、行号规范、与 no_boundary/未决候选不冲突
    replacement_map = review_doc.get("replacement_adds", {})
    if not isinstance(replacement_map, dict):
        errors.append("replacement_adds must be an object")
        replacement_map = {}
    elif set(replacement_map) - expected_units:
        errors.append(f"replacement_adds contains unknown units: {sorted(set(replacement_map) - expected_units)}")

    replacement_lines_by_unit: dict[str, set[int]] = {}
    for unit_id in expected_units:
        anchors_raw = replacement_map.get(unit_id, [])
        if not isinstance(anchors_raw, list):
            errors.append(f"{unit_id}: replacement_adds must be an array")
            continue
        anchors: list[int] = []
        for value in anchors_raw:
            line = _canonical_line(value)
            if line is None:
                errors.append(f"{unit_id}: replacement anchor {value!r} is not a canonical positive integer")
                continue
            anchors.append(line)
        if len(anchors) != len(set(anchors)):
            errors.append(f"{unit_id}: replacement_adds contains duplicates: {sorted(anchors)}")
        replacement_lines_by_unit[unit_id] = set(anchors)
        parsed = parsed_by_unit.get(unit_id, {})
        for anchor in sorted(set(anchors)):
            if anchor not in expected[unit_id]:
                # 替代锚点允许是非候选行（如对白先行上移到对白行）；
                # 锚点可切分性由 scene_segmenter.validate_boundary_overrides 在合并后校验
                continue
            decision = parsed.get(anchor)
            if decision == DECISION_NO_BOUNDARY:
                errors.append(f"{unit_id} L{anchor}: replacement anchor conflicts with no_boundary decision")
            elif decision is None:
                errors.append(f"{unit_id} L{anchor}: replacement anchor is an undecided low candidate")

    replacement_reasons = review_doc.get("replacement_reasons", {})
    if not isinstance(replacement_reasons, dict):
        errors.append("replacement_reasons must be an object")
        replacement_reasons = {}
    elif unknown := sorted(set(replacement_reasons) - expected_units):
        errors.append(f"replacement_reasons contains unknown units: {unknown}")
    for unit_id in expected_units:
        unit_reasons = replacement_reasons.get(unit_id, {})
        if not isinstance(unit_reasons, dict):
            errors.append(f"{unit_id}: replacement_reasons must be an object")
            continue
        parsed_reason_lines: set[int] = set()
        for key, reason in unit_reasons.items():
            line = _canonical_line(key)
            if line is None:
                errors.append(f"{unit_id}: replacement reason line {key!r} is not a canonical positive integer")
                continue
            parsed_reason_lines.add(line)
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{unit_id} L{line}: replacement reason must be a non-empty string")
        anchors = replacement_lines_by_unit.get(unit_id, set())
        if missing := sorted(anchors - parsed_reason_lines):
            errors.append(f"{unit_id}: replacement anchors missing reasons: {missing}")
        if orphaned := sorted(parsed_reason_lines - anchors):
            errors.append(f"{unit_id}: orphan replacement reasons: {orphaned}")

    notes_by_unit = review_doc.get("notes_by_unit", {})
    if not isinstance(notes_by_unit, dict):
        errors.append("notes_by_unit must be an object")
        notes_by_unit = {}
    elif unknown := sorted(set(notes_by_unit) - expected_units):
        errors.append(f"notes_by_unit contains unknown units: {unknown}")
    retained_by_unit: dict[str, set[str]] = {}
    for unit_id, unit_notes in notes_by_unit.items():
        if not isinstance(unit_notes, dict):
            errors.append(f"{unit_id}: notes_by_unit entry must be an object")
            continue
        retained = unit_notes.get("oversized_retained", [])
        if not isinstance(retained, list):
            errors.append(f"{unit_id}: oversized_retained must be an array")
            continue
        retained_ranges: set[str] = set()
        for index, item in enumerate(retained):
            if not isinstance(item, dict):
                errors.append(f"{unit_id}: oversized_retained[{index}] must be an object")
                continue
            lines = item.get("lines")
            if not isinstance(lines, str) or not re.fullmatch(r"L[1-9]\d*-[1-9]\d*", lines):
                errors.append(f"{unit_id}: oversized_retained[{index}].lines is invalid")
            else:
                retained_ranges.add(lines)
            for field in ("span", "turns"):
                if isinstance(item.get(field), bool) or not isinstance(item.get(field), int) or item[field] < 0:
                    errors.append(f"{unit_id}: oversized_retained[{index}].{field} must be a non-negative integer")
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                errors.append(f"{unit_id}: oversized_retained[{index}].reason must be a non-empty string")
        retained_by_unit[unit_id] = retained_ranges

    if expected_oversized_by_unit is not None:
        for unit_id in expected_units:
            expected_ranges = expected_oversized_by_unit.get(unit_id, set())
            actual_ranges = retained_by_unit.get(unit_id, set())
            if missing := sorted(expected_ranges - actual_ranges):
                errors.append(f"{unit_id}: oversized scenes missing retained reasons: {missing}")
            if stale := sorted(actual_ranges - expected_ranges):
                errors.append(f"{unit_id}: stale oversized retained reasons: {stale}")

    # oversized_scope / external_sampling：行号必须是真实 low 候选；二者不得重叠
    scope_map = review_doc.get("oversized_scope", {})
    if not isinstance(scope_map, dict):
        errors.append("oversized_scope must be an object")
        scope_map = {}
    elif set(scope_map) - expected_units:
        errors.append(f"oversized_scope contains unknown units: {sorted(set(scope_map) - expected_units)}")

    sampling = review_doc.get("external_sampling", {})
    sampled_map: dict[str, list[int]] = {}
    if not isinstance(sampling, dict):
        errors.append("external_sampling must be an object")
    else:
        if not isinstance(sampling.get("rule"), str) or not sampling.get("rule"):
            errors.append("external_sampling.rule must be a non-empty string")
        raw_sampled = sampling.get("sampled_lines", {})
        if not isinstance(raw_sampled, dict):
            errors.append("external_sampling.sampled_lines must be an object")
        else:
            if set(raw_sampled) - expected_units:
                errors.append(
                    f"external_sampling.sampled_lines contains unknown units: {sorted(set(raw_sampled) - expected_units)}"
                )
            for unit_id in expected_units:
                lines_raw = raw_sampled.get(unit_id, [])
                if not isinstance(lines_raw, list):
                    errors.append(f"{unit_id}: sampled_lines must be an array")
                    continue
                for value in lines_raw:
                    line = _canonical_line(value)
                    if line is None:
                        errors.append(f"{unit_id}: sampled line {value!r} is not a canonical positive integer")
                    elif line not in expected[unit_id]:
                        errors.append(f"{unit_id} L{line}: sampled line is not a low candidate")
                sampled_map[unit_id] = [int(v) for v in lines_raw if _canonical_line(v) is not None]

    for unit_id in expected_units:
        scope_lines = scope_map.get(unit_id, [])
        if not isinstance(scope_lines, list):
            errors.append(f"{unit_id}: oversized_scope must be an array")
            continue
        for value in scope_lines:
            line = _canonical_line(value)
            if line is None:
                errors.append(f"{unit_id}: oversized scope line {value!r} is not a canonical positive integer")
            elif line not in expected[unit_id]:
                errors.append(f"{unit_id} L{line}: oversized scope line is not a low candidate")
        scope_set = {int(v) for v in scope_lines if _canonical_line(v) is not None}
        if overlap := sorted(scope_set & set(sampled_map.get(unit_id, []))):
            errors.append(f"{unit_id}: lines both in oversized scope and external sample: {overlap}")

    if require_complete:
        for unit_id in expected_units:
            parsed = parsed_by_unit.get(unit_id, {})
            required = {int(v) for v in (scope_map.get(unit_id) or []) if _canonical_line(v) is not None} | set(
                sampled_map.get(unit_id, [])
            )
            for line in sorted(required):
                if parsed.get(line) not in VALID_LOW_DECISIONS:
                    errors.append(f"{unit_id} L{line}: required low candidate is undecided")
    return errors


def save_low_review_document(path: Path | str, review_doc: dict[str, Any]) -> None:
    """原子写出状态文件：先写 .tmp 再 os.replace，避免半写状态。"""
    path = Path(path)
    payload = json.dumps(review_doc, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load_low_review_document(path: Path | str) -> dict[str, Any]:
    """读取状态文件（JSON）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_low_boundaries_into_overrides(
    overrides_doc: dict[str, Any],
    review_doc: dict[str, Any],
    candidates_by_unit: dict[str, list[BoundaryCandidate]],
) -> dict[str, Any]:
    """Return a copy of overrides with reviewed low boundaries merged into adds.

    合并来源 = boundary 决定 ∪ replacement_adds；返回深拷贝，输入对象不被修改。
    """
    errors = validate_low_review_document(review_doc, candidates_by_unit)
    if errors:
        raise ValueError("low review document is invalid:\n- " + "\n- ".join(errors))

    merged = deepcopy(overrides_doc)
    adds_map = merged.setdefault("adds", {})
    decisions_map = review_doc["candidate_decisions"]
    replacement_map = review_doc.get("replacement_adds", {})
    for unit in STORY_UNITS:
        unit_id = unit.unit_id
        additions = {int(line) for line, decision in decisions_map[unit_id].items() if decision == DECISION_BOUNDARY}
        rejected = {int(line) for line, decision in decisions_map[unit_id].items() if decision == DECISION_NO_BOUNDARY}
        replacements = {int(line) for line in replacement_map.get(unit_id, [])}
        existing = {int(line) for line in adds_map.get(unit_id, [])}
        if conflicts := sorted(existing & rejected):
            raise ValueError(f"{unit_id}: adds conflict with no_boundary low decisions at {conflicts}")
        if conflicts := sorted(replacements & rejected):
            raise ValueError(f"{unit_id}: replacement_adds conflict with no_boundary decisions at {conflicts}")
        adds_map[unit_id] = sorted(existing | additions | replacements)
    return merged


def freeze_reviewed_scenes(
    game_root: Path,
    overrides_doc: dict[str, Any],
    review_doc: dict[str, Any],
    out_dir: Path,
    *,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    """Freeze scenes only after both boundary-review layers pass their gates.

    This is the sole public freeze entry. It proves that every required low
    candidate was reviewed, every reviewed boundary is present in overrides,
    and every currently retained oversized scene has an auditable reason.
    """
    from knowledge.game_rag.parser import parse_script_directory

    if not isinstance(review_doc, dict):
        raise ValueError("low review document is required")
    if review_doc.get("review_status") != "approved":
        raise ValueError(f"low review_status must be approved, got {review_doc.get('review_status')!r}")
    if str(review_doc.get("reviewer", "")).strip() != str(overrides_doc.get("reviewer", "")).strip():
        raise ValueError("boundary and low review documents must have the same reviewer")

    source_prefix = str(overrides_doc.get("source_prefix") or SOURCE_PREFIX)
    segments = parse_script_directory(game_root, source_prefix=source_prefix)
    grouped = split_segments_by_unit(segments)
    candidates_by_unit = {unit.unit_id: detect_scene_boundaries(unit, grouped[unit.unit_id]) for unit in STORY_UNITS}
    errors = validate_low_review_document(review_doc, candidates_by_unit, require_complete=True)
    if errors:
        raise ValueError("low review document is invalid:\n- " + "\n- ".join(errors))

    merged = merge_low_boundaries_into_overrides(overrides_doc, review_doc, candidates_by_unit)
    for unit in STORY_UNITS:
        unit_id = unit.unit_id
        supplied = sorted(int(line) for line in overrides_doc.get("adds", {}).get(unit_id, []))
        expected = merged.get("adds", {}).get(unit_id, [])
        if supplied != expected:
            raise ValueError(f"{unit_id}: overrides adds do not exactly include reviewed low boundaries")

    boundary_errors = validate_boundary_overrides(overrides_doc, grouped, candidates_by_unit)
    if boundary_errors:
        raise ValueError("boundary review document is invalid:\n- " + "\n- ".join(boundary_errors))

    expected_oversized: dict[str, set[str]] = {}
    for unit in STORY_UNITS:
        unit_id = unit.unit_id
        unit_segments = grouped[unit_id]
        plan = plan_scene_boundaries_from_decisions(unit, unit_segments, candidates_by_unit[unit_id], overrides_doc)
        source_lines = _load_source_lines(game_root, unit.source_path, source_prefix)
        scenes = build_scene_documents(unit, unit_segments, plan, source_lines=source_lines)
        turns = _dialogue_turns_per_scene(unit_segments, plan.boundaries)
        expected_oversized[unit_id] = {
            f"L{scene.source.line_start}-{scene.source.line_end}"
            for scene, dialogue_turns in zip(scenes, turns, strict=True)
            if scene.source.line_end - scene.source.line_start + 1 > OVERSIZED_SPAN_LINES
            or dialogue_turns > OVERSIZED_DIALOGUE_TURNS
        }

    errors = validate_low_review_document(
        review_doc,
        candidates_by_unit,
        require_complete=True,
        expected_oversized_by_unit=expected_oversized,
    )
    if errors:
        raise ValueError("low review document is invalid:\n- " + "\n- ".join(errors))

    return _freeze_scenes_from_overrides(game_root, overrides_doc, out_dir, review_dir=review_dir)
