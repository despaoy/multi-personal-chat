from __future__ import annotations

import json
from dataclasses import replace

import pytest

from knowledge.game_rag.low_review import (
    SAMPLE_NEAR_BOUNDARY,
    SAMPLE_STRIDE,
    create_low_review_document,
    load_low_review_document,
    low_candidate_lines,
    merge_low_boundaries_into_overrides,
    save_low_review_document,
    stratified_external_sample,
    validate_low_review_document,
)
from knowledge.game_rag.scene_segmenter import CONFIDENCE_LOW, BoundaryCandidate
from knowledge.game_rag.story_units import STORY_UNITS


def _candidates():
    return {
        unit.unit_id: [
            BoundaryCandidate(
                unit_id=unit.unit_id,
                source_path=unit.source_path,
                line=10 + index,
                signal="long_narration_gap",
                confidence=CONFIDENCE_LOW,
                effective_default=False,
                trigger_text="transition",
            )
        ]
        for index, unit in enumerate(STORY_UNITS)
    }


def _overrides():
    return {
        "reviewer": "reviewer-01",
        "adds": {unit.unit_id: [] for unit in STORY_UNITS},
    }


def _decide_boundary(doc, unit_id, line, reason="verified scene transition"):
    doc["candidate_decisions"][unit_id][str(line)] = "boundary"
    doc.setdefault("reasons", {}).setdefault(unit_id, {})[str(line)] = reason


def _add_replacement(doc, unit_id, line, reason="verified replacement anchor"):
    doc["replacement_adds"][unit_id] = [line]
    doc.setdefault("replacement_reasons", {}).setdefault(unit_id, {})[str(line)] = reason


def test_create_document_covers_all_low_candidates():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())

    assert validate_low_review_document(doc, candidates) == []
    assert sum(len(lines) for lines in low_candidate_lines(candidates).values()) == len(STORY_UNITS)


def test_existing_low_add_is_preserved_as_boundary():
    candidates = _candidates()
    overrides = _overrides()
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    overrides["adds"][unit.unit_id] = [line]

    doc = create_low_review_document(candidates, overrides)

    assert doc["candidate_decisions"][unit.unit_id][str(line)] == "boundary"


def test_unknown_and_missing_candidates_are_rejected():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    unit = STORY_UNITS[0]
    decisions = doc["candidate_decisions"][unit.unit_id]
    decisions.pop(next(iter(decisions)))
    decisions["999"] = None

    errors = validate_low_review_document(doc, candidates)

    assert any("missing low candidates" in error for error in errors)
    assert any("unknown low candidates" in error for error in errors)


def test_boundary_decision_merges_into_adds():
    candidates = _candidates()
    overrides = _overrides()
    doc = create_low_review_document(candidates, overrides)
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    _decide_boundary(doc, unit.unit_id, line)

    merged = merge_low_boundaries_into_overrides(overrides, doc, candidates)

    assert merged["adds"][unit.unit_id] == [line]
    assert overrides["adds"][unit.unit_id] == []


def test_no_boundary_conflicting_with_existing_add_is_rejected():
    candidates = _candidates()
    overrides = _overrides()
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    overrides["adds"][unit.unit_id] = [line]
    doc = create_low_review_document(candidates, overrides)
    doc["candidate_decisions"][unit.unit_id][str(line)] = "no_boundary"

    with pytest.raises(ValueError, match="conflict"):
        merge_low_boundaries_into_overrides(overrides, doc, candidates)


def test_non_low_candidates_are_not_part_of_review_state():
    candidates = _candidates()
    unit = STORY_UNITS[0]
    candidates[unit.unit_id].append(replace(candidates[unit.unit_id][0], line=999, confidence="medium"))

    lines = low_candidate_lines(candidates)

    assert 999 not in lines[unit.unit_id]


# ---------- 原子保存 / 读取 ----------


def test_save_and_load_roundtrip_atomic(tmp_path):
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    path = tmp_path / "low_candidate_review.json"

    save_low_review_document(path, doc)

    assert not (tmp_path / "low_candidate_review.json.tmp").exists()
    assert load_low_review_document(path) == doc


def test_save_replaces_existing_file_without_tmp_residue(tmp_path):
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    path = tmp_path / "low_candidate_review.json"
    path.write_text("{}", encoding="utf-8")

    save_low_review_document(path, doc)

    assert json.loads(path.read_text(encoding="utf-8")) == doc
    assert list(tmp_path.glob("*.tmp")) == []


# ---------- 替代锚点（replacement_adds） ----------


def test_replacement_anchor_merges_into_adds():
    candidates = _candidates()
    overrides = _overrides()
    doc = create_low_review_document(candidates, overrides)
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    doc["candidate_decisions"][unit.unit_id][str(line)] = "no_boundary"
    # 替代锚点不是候选行（对白先行上移到对白行的常见形态）
    _add_replacement(doc, unit.unit_id, line + 7)

    merged = merge_low_boundaries_into_overrides(overrides, doc, candidates)

    assert merged["adds"][unit.unit_id] == [line + 7]


def test_replacement_anchor_on_no_boundary_candidate_rejected():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    doc["candidate_decisions"][unit.unit_id][str(line)] = "no_boundary"
    _add_replacement(doc, unit.unit_id, line)

    errors = validate_low_review_document(doc, candidates)

    assert any("replacement anchor conflicts with no_boundary" in error for error in errors)


def test_replacement_anchor_on_undecided_candidate_rejected():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    _add_replacement(doc, unit.unit_id, line)

    errors = validate_low_review_document(doc, candidates)

    assert any("replacement anchor is an undecided low candidate" in error for error in errors)


def test_replacement_anchor_on_boundary_candidate_allowed():
    candidates = _candidates()
    overrides = _overrides()
    doc = create_low_review_document(candidates, overrides)
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    _decide_boundary(doc, unit.unit_id, line)
    doc["replacement_adds"][unit.unit_id] = [line, line + 3]
    doc["replacement_reasons"][unit.unit_id] = {
        str(line): "same candidate anchor",
        str(line + 3): "nearby replacement anchor",
    }

    merged = merge_low_boundaries_into_overrides(overrides, doc, candidates)

    assert merged["adds"][unit.unit_id] == [line, line + 3]


def test_duplicate_replacement_anchors_rejected():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    unit = STORY_UNITS[0]
    doc["replacement_adds"][unit.unit_id] = [42, 42]
    doc["replacement_reasons"][unit.unit_id] = {"42": "duplicate anchor"}

    errors = validate_low_review_document(doc, candidates)

    assert any("replacement_adds contains duplicates" in error for error in errors)


def test_boundary_decision_requires_reason():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    doc["candidate_decisions"][unit.unit_id][str(line)] = "boundary"

    errors = validate_low_review_document(doc, candidates)

    assert any("boundary decision requires a non-empty reason" in error for error in errors)


def test_replacement_reason_must_match_anchor_exactly():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    unit = STORY_UNITS[0]
    doc["replacement_adds"][unit.unit_id] = [42]
    doc["replacement_reasons"][unit.unit_id] = {"43": "orphan"}

    errors = validate_low_review_document(doc, candidates)

    assert any("replacement anchors missing reasons" in error for error in errors)
    assert any("orphan replacement reasons" in error for error in errors)


def test_notes_by_unit_entry_must_be_object():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    doc["notes_by_unit"][STORY_UNITS[0].unit_id] = "free-form note"

    errors = validate_low_review_document(doc, candidates)

    assert any("notes_by_unit entry must be an object" in error for error in errors)


def test_expected_oversized_ranges_require_exact_reasons():
    candidates = _candidates()
    doc = create_low_review_document(candidates, _overrides())
    uid = STORY_UNITS[0].unit_id
    doc["notes_by_unit"][uid] = {
        "oversized_retained": [{"lines": "L1-400", "span": 400, "turns": 160, "reason": "continuous event"}]
    }

    errors = validate_low_review_document(
        doc,
        candidates,
        expected_oversized_by_unit={uid: {"L2-401"}},
    )

    assert any("oversized scenes missing retained reasons" in error for error in errors)
    assert any("stale oversized retained reasons" in error for error in errors)


# ---------- 分层抽检（确定性） ----------


def test_stratified_sample_is_deterministic_and_strided():
    uid = STORY_UNITS[0].unit_id
    external = {uid: {101, 104, 107, 110, 113, 116, 119, 200, 203}}
    boundaries = {uid: [150]}

    sample = stratified_external_sample(external, boundaries)
    again = stratified_external_sample(external, boundaries)

    assert sample == again
    # 场景 0（<150）：排序后 101,104,107,110,113,116,119 → 每 5 条取 1 条（含首条）= 101,116
    # 场景 1（>=150）：200,203 → 取 200；200 距边界 150 超过 ±3，不因邻近纳入
    assert sample[uid] == [101, 116, 200]


def test_stratified_sample_includes_near_boundary_lines():
    uid = STORY_UNITS[0].unit_id
    external = {uid: {148, 151, 180}}
    boundaries = {uid: [150, 175]}

    sample = stratified_external_sample(external, boundaries)

    # 148/151 距边界 150 在 ±3 内 → 优先纳入；180 独立成层（>=175）取层首
    assert sample[uid] == [148, 151, 180]


def test_stratified_sample_skips_non_first_non_near_lines():
    uid = STORY_UNITS[0].unit_id
    external = {uid: {148, 151, 180}}
    boundaries = {uid: [150]}

    sample = stratified_external_sample(external, boundaries)

    # 180 与 151 同层且非层首、距边界超过 ±3 → 不入选
    assert sample[uid] == [148, 151]


def test_stratified_sample_covers_unregistered_units():
    sample = stratified_external_sample({}, {})

    assert set(sample) == {unit.unit_id for unit in STORY_UNITS}
    assert all(lines == [] for lines in sample.values())


# ---------- 完成度门禁（require_complete） ----------


def test_require_complete_flags_undecided_required_lines():
    candidates = _candidates()
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    doc = create_low_review_document(
        candidates,
        _overrides(),
        in_oversized_by_unit={unit.unit_id: {line}},
        external_sample_by_unit={unit.unit_id: []},
    )

    errors = validate_low_review_document(doc, candidates, require_complete=True)

    assert any("required low candidate is undecided" in error for error in errors)
    # 非必审范围（其他单元）的未决候选不触发门禁
    assert len([e for e in errors if "required low candidate" in e]) == 1


def test_require_complete_passes_when_scope_decided():
    candidates = _candidates()
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    doc = create_low_review_document(
        candidates,
        _overrides(),
        in_oversized_by_unit={unit.unit_id: {line}},
        external_sample_by_unit={unit.unit_id: []},
    )
    doc["candidate_decisions"][unit.unit_id][str(line)] = "no_boundary"

    assert validate_low_review_document(doc, candidates, require_complete=True) == []


def test_scope_and_sample_must_not_overlap():
    candidates = _candidates()
    unit = STORY_UNITS[0]
    line = candidates[unit.unit_id][0].line
    doc = create_low_review_document(
        candidates,
        _overrides(),
        in_oversized_by_unit={unit.unit_id: {line}},
        external_sample_by_unit={unit.unit_id: [line]},
    )

    errors = validate_low_review_document(doc, candidates)

    assert any("both in oversized scope and external sample" in error for error in errors)


def test_scope_lines_must_be_real_low_candidates():
    candidates = _candidates()
    unit = STORY_UNITS[0]
    doc = create_low_review_document(
        candidates,
        _overrides(),
        in_oversized_by_unit={unit.unit_id: {999}},
        external_sample_by_unit={unit.unit_id: []},
    )

    errors = validate_low_review_document(doc, candidates)

    assert any("oversized scope line is not a low candidate" in error for error in errors)


def test_sample_stride_constants_are_stable():
    assert SAMPLE_STRIDE == 5
    assert SAMPLE_NEAR_BOUNDARY == 3
