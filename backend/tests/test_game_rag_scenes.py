"""P3/P3.1 场景切分测试：合成样例 + 全语料回归 + 决定验证与冻结门禁。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from knowledge.game_rag.low_review import (
    freeze_reviewed_scenes,
    load_low_review_document,
    validate_low_review_document,
)
from knowledge.game_rag.models import ContentScope, SegmentType
from knowledge.game_rag.parser import _split_lines, parse_script_directory, parse_script_text
from knowledge.game_rag.scene_segmenter import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    DECISION_BOUNDARY,
    DECISION_NO_BOUNDARY,
    DEFAULT_MIN_DIALOGUE_TURNS,
    SIGNAL_BLANK_LINE,
    SIGNAL_LONG_NARRATION,
    SIGNAL_TIME_JUMP,
    SIGNAL_TRANSITION,
    _load_source_lines,
    build_scene_documents,
    detect_scene_boundaries,
    plan_scene_boundaries,
    plan_scene_boundaries_from_decisions,
    validate_boundary_overrides,
)
from knowledge.game_rag.scene_segmenter import (
    _freeze_scenes_from_overrides as freeze_scenes,
)
from knowledge.game_rag.story_units import (
    EPILOGUE_FIXED_BOUNDARY,
    STORY_UNITS,
    StoryUnit,
    split_segments_by_unit,
    unit_by_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAME_ROOT = PROJECT_ROOT / "gametext" / "纸上魔法使"
GAME_REL = "gametext/纸上魔法使"

needs_corpus = pytest.mark.skipif(not GAME_ROOT.exists(), reason="游戏语料不存在（CI 未携带 gametext）")

# 用真实登记单元 id（validate 按 STORY_UNITS 匹配单元键）
_SYNTH_UNIT_ID = "vol06_6芙蓉石的终焉轮回"


def _unit(unit_id: str = _SYNTH_UNIT_ID, *, line_start: int = 1) -> StoryUnit:
    return StoryUnit(
        unit_id=unit_id,
        source_path=f"{GAME_REL}/test.txt",
        story_title="测试卷",
        volume_number=1,
        content_scope=ContentScope.main_story,
        viewpoint=None,
        line_start=line_start,
        line_end=None,
    )


def _six_turn_text(prefix: str, suffix: str) -> str:
    """prefix + 6 轮对话 + suffix，保证主场景满足最小轮数。"""
    turns = "".join(f"[妃] 「第{i}句。」\n" for i in range(1, 7))
    return prefix + turns + suffix


def _parse(text: str):
    return parse_script_text(text, source_path=f"{GAME_REL}/test.txt")


def _build(unit, segments, plan, text: str):
    return build_scene_documents(unit, segments, plan, source_lines=_split_lines(text))


class TestDetectBoundaries:
    def test_time_jump_at_narration_start(self):
        text = _six_turn_text("", "次日，琉璃来到了学校。\n[琉璃] 「早。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        hits = [c for c in candidates if c.signal == SIGNAL_TIME_JUMP]
        assert len(hits) == 1
        assert hits[0].confidence == CONFIDENCE_HIGH
        assert hits[0].effective_default is True

    def test_time_jump_inside_merged_narration_block(self):
        """叙述块内部的时间词行也能检出（逐行扫描，不受块合并影响）。"""
        text = _six_turn_text("", "叙述一。\n次日，场景变了。\n[琉璃] 「早。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        hits = [c for c in candidates if c.signal == SIGNAL_TIME_JUMP]
        assert [c.line for c in hits] == [8]  # 6 轮对话后第 7 行叙述、第 8 行时间词

    def test_blank_line_gap(self):
        text = _six_turn_text("", "\n[琉璃] 「隔了一场。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        hits = [c for c in candidates if c.signal == SIGNAL_BLANK_LINE]
        assert len(hits) == 1
        assert hits[0].confidence == CONFIDENCE_MEDIUM

    def test_transition_marker(self):
        text = _six_turn_text("", "※追加剧本\n[琉璃] 「新场景。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        hits = [c for c in candidates if c.signal == SIGNAL_TRANSITION]
        assert len(hits) == 1 and hits[0].confidence == CONFIDENCE_HIGH

    def test_full_line_separator_matches(self):
        text = _six_turn_text("", "————\n[琉璃] 「新场景。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        assert [c for c in candidates if c.signal == SIGNAL_TRANSITION]

    def test_prefix_dash_line_is_not_transition(self):
        """行首破折号续写行（全语料 322 条）不得误判为转场（P3 修正回归）。"""
        text = _six_turn_text("", "——就这样，时间过去了。\n[琉璃] 「还在原场景。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        assert [c for c in candidates if c.signal == SIGNAL_TRANSITION] == []

    def test_long_narration_gap_recorded_only(self):
        narration = "".join(f"叙述第{i}行。\n" for i in range(1, 6))
        text = _six_turn_text("", narration + "[琉璃] 「后面。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        hits = [c for c in candidates if c.signal == SIGNAL_LONG_NARRATION]
        assert len(hits) == 1
        assert hits[0].confidence == CONFIDENCE_LOW
        assert hits[0].effective_default is False

    def test_no_candidate_inside_dialogue(self):
        """时间词出现在 dialogue 段（含未闭合吞并行）内不得产生候选。"""
        text = _six_turn_text("", "[夜子] 「未闭合\n次日的延续\n[琉璃] 「新台词。」\n")
        segments = _parse(text)
        candidates = detect_scene_boundaries(_unit(), segments)
        dialogue_spans = [
            (s.source.line_start, s.source.line_end) for s in segments if s.segment_type is SegmentType.dialogue
        ]
        for cand in candidates:
            for start, end in dialogue_spans:
                assert not (start < cand.line <= end), f"候选 L{cand.line} 锚进 dialogue L{start}-{end}"

    def test_candidates_sorted_and_deterministic(self):
        text = _six_turn_text("", "次日，变了。\n[琉璃] 「早。」\n\n又一场。\n")
        segments = _parse(text)
        first = detect_scene_boundaries(_unit(), segments)
        second = detect_scene_boundaries(_unit(), segments)
        assert first == second
        assert [c.line for c in first] == sorted(c.line for c in first)


class TestPlanAndBuild:
    def test_narration_split_at_interior_boundary(self):
        """边界落在叙述块内部时，叙述被切成两片，各归前后场景。"""
        text = _six_turn_text("", "叙述一。\n次日，场景变了。\n[琉璃] 「早。」\n")
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        plan = plan_scene_boundaries(unit, segments, candidates, min_dialogue_turns=0)
        scenes = _build(unit, segments, plan, text)
        assert len(scenes) == 2
        assert (scenes[0].source.line_end, scenes[1].source.line_start) == (7, 8)

    def test_min_turns_merges_small_scene(self):
        text = _six_turn_text("", "次日，新场景。\n[琉璃] 「只有一句。」\n")
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        plan = plan_scene_boundaries(unit, segments, candidates, min_dialogue_turns=DEFAULT_MIN_DIALOGUE_TURNS)
        assert plan.auto_merged  # 小场景边界被合并
        assert plan.boundaries == []
        scenes = _build(unit, segments, plan, text)
        assert len(scenes) == 1

    def test_first_small_scene_merges_forward(self):
        text = "开场一句。\n[琉璃] 「序幕。」\n次日，正文。\n" + "".join(f"[妃] 「第{i}句。」\n" for i in range(1, 7))
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        plan = plan_scene_boundaries(unit, segments, candidates, min_dialogue_turns=6)
        assert plan.auto_merged
        scenes = _build(unit, segments, plan, text)
        assert len(scenes) == 1

    def test_overrides_remove_and_add_preview_mode(self):
        text = _six_turn_text("", "次日，新场景。\n" + "".join(f"[琉璃] 「第{i}句。」\n" for i in range(1, 7)))
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        boundary_line = next(c.line for c in candidates if c.signal == SIGNAL_TIME_JUMP)
        overrides = {"unit_decisions": {unit.unit_id: {"remove": [boundary_line], "add": []}}}
        plan = plan_scene_boundaries(unit, segments, candidates, overrides=overrides, min_dialogue_turns=0)
        assert plan.boundaries == []
        assert plan.overridden_removed == [boundary_line]

    def test_overrides_add_protected_from_merge(self):
        text = _six_turn_text("", "普通叙述。\n[琉璃] 「一句。」\n")
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)  # 无自动候选
        add_line = segments[-1].source.line_start  # 在最后一句 dialogue 前加边界
        overrides = {"unit_decisions": {unit.unit_id: {"remove": [], "add": [add_line - 1]}}}
        plan = plan_scene_boundaries(unit, segments, candidates, overrides=overrides, min_dialogue_turns=6)
        assert plan.boundaries == [add_line - 1]  # 人工边界不被合并撤销
        scenes = _build(unit, segments, plan, text)
        assert len(scenes) == 2

    def test_overrides_add_inside_dialogue_rejected(self):
        text = _six_turn_text("", "[夜子] 「未闭合\n被吞的叙述行\n[琉璃] 「新台词。」\n")
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        dialogue = next(
            s for s in segments if s.segment_type is SegmentType.dialogue and s.source.line_end > s.source.line_start
        )
        bad_anchor = dialogue.source.line_end  # 未闭合 dialogue 内部行
        overrides = {"unit_decisions": {unit.unit_id: {"remove": [], "add": [bad_anchor]}}}
        with pytest.raises(ValueError, match="dialogue 段内部"):
            plan_scene_boundaries(unit, segments, candidates, overrides=overrides, min_dialogue_turns=0)

    def test_scene_document_fields(self):
        text = _six_turn_text("", "次日，新场景。\n" + "".join(f"[琉璃] 「新{i}。」\n" for i in range(1, 7)))
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        plan = plan_scene_boundaries(unit, segments, candidates, min_dialogue_turns=0)
        scenes = _build(unit, segments, plan, text)
        assert len(scenes) == 2
        for scene in scenes:
            assert scene.story.story_unit_id == unit.unit_id
            assert scene.story.route is None and scene.story.continuity_id is None
            assert scene.story.content_scope is ContentScope.main_story
            assert scene.story.temporal_scope is None  # P3.1：尚未审核为 None
            assert scene.mentioned_characters == [] and scene.present_characters == []
            assert scene.review_status.value == "draft"
            assert scene.id.startswith("scene_")
        assert scenes[0].speakers == ["妃"]
        assert scenes[1].speakers == ["琉璃"]

    def test_scene_ids_deterministic_and_stable(self):
        text = _six_turn_text("", "次日，新场景。\n" + "".join(f"[琉璃] 「新{i}。」\n" for i in range(1, 7)))
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        plan = plan_scene_boundaries(unit, segments, candidates, min_dialogue_turns=0)
        scenes_a = _build(unit, segments, plan, text)
        scenes_b = _build(unit, segments, plan, text)
        assert [s.model_dump() for s in scenes_a] == [s.model_dump() for s in scenes_b]


class TestSceneTextVerbatim:
    """P3.1：text 为原文物理行逐字拼接（含标签/空行/未闭合/重复闭引号）。"""

    def test_dialogue_keeps_speaker_tag(self):
        text = _six_turn_text("", "")
        segments = _parse(text)
        plan = plan_scene_boundaries(_unit(), segments, [], min_dialogue_turns=0)
        (scene,) = _build(_unit(), segments, plan, text)
        assert "[妃] 「第1句。」" in scene.text
        assert "[妃] 「第6句。」" in scene.text

    def test_multiline_dialogue_tag_only_on_first_line(self):
        text = "[妃] 「我讨厌大海，\n受不了海风吹乱头发。」\n" + "".join(f"[琉璃] 「第{i}句。」\n" for i in range(1, 7))
        segments = _parse(text)
        plan = plan_scene_boundaries(_unit(), segments, [], min_dialogue_turns=0)
        (scene,) = _build(_unit(), segments, plan, text)
        assert "[妃] 「我讨厌大海，\n受不了海风吹乱头发。」" in scene.text
        lines = scene.text.split("\n")
        assert lines[0] == "[妃] 「我讨厌大海，"
        assert lines[1] == "受不了海风吹乱头发。」"  # 续行无标签

    def test_merged_blank_line_restored_in_text(self):
        """被合并的空行边界：场景内部空行必须恢复（P3.1 Critical）。"""
        text = _six_turn_text("", "\n[琉璃] 「隔了一场。」\n")
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        plan = plan_scene_boundaries(unit, segments, candidates, min_dialogue_turns=6)
        assert plan.auto_merged  # 空行边界因 <6 轮被合并
        (scene,) = _build(unit, segments, plan, text)
        # 逐字一致：text 即原文 L1-8（含第 7 行空行）
        assert scene.text == "\n".join(_split_lines(text)[scene.source.line_start - 1 : scene.source.line_end])
        assert "\n\n" in scene.text  # 空行（第 7 行）恢复在合并场景内部

    def test_scene_text_equals_source_lines_by_span(self):
        text = "叙述一。\n\n叙述二。\n[妃] 「台词。」\n结尾。\n"
        segments = _parse(text)
        plan = plan_scene_boundaries(_unit(), segments, [], min_dialogue_turns=0)
        (scene,) = _build(_unit(), segments, plan, text)
        assert scene.text == "\n".join(_split_lines(text)[0:5])

    def test_unclosed_and_duplicate_quote_lines_preserved(self):
        """未闭合台词与重复闭引号所在行原样进入场景文本。"""
        text = _six_turn_text("", "[夜子] 「未闭合\n[琉璃] 「新台词。」\n[克] 「不要！」」\n")
        segments = _parse(text)
        plan = plan_scene_boundaries(_unit(), segments, [], min_dialogue_turns=0)
        (scene,) = _build(_unit(), segments, plan, text)
        lines = _split_lines(text)
        for line in lines[1:]:
            assert line in scene.text
        assert "[夜子] 「未闭合" in scene.text  # 未闭合原文行（含标签）
        assert "[克] 「不要！」」" in scene.text  # 重复闭引号原样


class TestDecisionsMode:
    """P3.1：approved 决定模式（无自动合并，边界完全由决定生成）。"""

    def _context(self):
        text = _six_turn_text("", "次日，新场景。\n" + "".join(f"[琉璃] 「新{i}。」\n" for i in range(1, 7)))
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        return unit, segments, candidates, text

    def _doc(self, decisions: dict, adds: list[int] | None = None) -> dict:
        return {
            "boundary_review_status": "approved",
            "reviewer": "tester",
            "min_dialogue_turns": 6,
            "candidate_decisions": {_SYNTH_UNIT_ID: decisions},
            "adds": {_SYNTH_UNIT_ID: adds or []},
        }

    def test_boundary_decision_keeps_boundary(self):
        unit, segments, candidates, text = self._context()
        line = next(c.line for c in candidates if c.confidence == CONFIDENCE_HIGH)
        doc = self._doc({str(line): DECISION_BOUNDARY})
        plan = plan_scene_boundaries_from_decisions(unit, segments, candidates, doc)
        assert plan.boundaries == [line]
        scenes = _build(unit, segments, plan, text)
        assert len(scenes) == 2

    def test_no_boundary_decision_removes_boundary(self):
        unit, segments, candidates, text = self._context()
        line = next(c.line for c in candidates if c.confidence == CONFIDENCE_HIGH)
        doc = self._doc({str(line): DECISION_NO_BOUNDARY})
        plan = plan_scene_boundaries_from_decisions(unit, segments, candidates, doc)
        assert plan.boundaries == []
        (scene,) = _build(unit, segments, plan, text)
        assert scene.source.line_start == 1

    def test_adds_promote_low_candidate_or_new_boundary(self):
        unit, segments, candidates, text = self._context()
        narration = "".join(f"叙述第{i}行。\n" for i in range(1, 6))
        text2 = _six_turn_text("", narration + "[琉璃] 「后面。」\n")
        segments2 = _parse(text2)
        candidates2 = detect_scene_boundaries(unit, segments2)
        low_line = next(c.line for c in candidates2 if c.confidence == CONFIDENCE_LOW)
        doc = self._doc({}, adds=[low_line])
        plan = plan_scene_boundaries_from_decisions(unit, segments2, candidates2, doc)
        assert plan.boundaries == [low_line]
        scenes = _build(unit, segments2, plan, text2)
        assert len(scenes) == 2

    def test_decisions_do_not_trigger_auto_merge(self):
        """决定模式不做最小轮数合并：小场景边界若被决定为 boundary 则保留。"""
        text = _six_turn_text("", "次日，小场景。\n[琉璃] 「一句。」\n")
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        line = next(c.line for c in candidates if c.confidence == CONFIDENCE_HIGH)
        doc = self._doc({str(line): DECISION_BOUNDARY})
        plan = plan_scene_boundaries_from_decisions(unit, segments, candidates, doc)
        assert plan.boundaries == [line]  # 即使 <6 轮也保留（人工决定优先）
        scenes = _build(unit, segments, plan, text)
        assert len(scenes) == 2


class TestValidateBoundaryOverrides:
    """P3.1：approved 决定验证（错误注入，合成数据）。"""

    def _context(self):
        text = _six_turn_text("", "次日，新场景。\n" + "".join(f"[琉璃] 「新{i}。」\n" for i in range(1, 7)))
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        grouped = {u.unit_id: [] for u in STORY_UNITS}
        grouped[unit.unit_id] = segments
        candidates_by_unit = {u.unit_id: [] for u in STORY_UNITS}
        candidates_by_unit[unit.unit_id] = candidates
        return unit, segments, candidates, grouped, candidates_by_unit

    def _full_doc(self, grouped, candidates_by_unit, decision=DECISION_NO_BOUNDARY):
        decisions = {}
        for uid, cands in candidates_by_unit.items():
            must = {str(c.line) for c in cands if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM)}
            decisions[uid] = {line: decision for line in must}
        return {
            "schema_version": 2,
            "boundary_review_status": "approved",
            "reviewer": "tester",
            "min_dialogue_turns": 6,
            "candidate_decisions": decisions,
            "adds": {uid: [] for uid in grouped},
        }

    def test_valid_full_decisions_pass(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        assert (
            validate_boundary_overrides(self._full_doc(grouped, candidates_by_unit), grouped, candidates_by_unit) == []
        )

    def test_boundary_decisions_pass(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        assert (
            validate_boundary_overrides(
                self._full_doc(grouped, candidates_by_unit, DECISION_BOUNDARY), grouped, candidates_by_unit
            )
            == []
        )

    def test_draft_status_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["boundary_review_status"] = "draft"
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("boundary_review_status" in e for e in errors)

    def test_missing_reviewer_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["reviewer"] = "   "
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("reviewer" in e for e in errors)

    def test_missing_reviewer_does_not_hide_missing_decision(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["reviewer"] = ""
        must_line = next(str(c.line) for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM))
        doc["candidate_decisions"][unit.unit_id][must_line] = None

        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)

        assert any("reviewer" in error for error in errors)
        assert any("缺少明确决定" in error for error in errors)

    def test_missing_decision_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        must_line = next(str(c.line) for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM))
        doc["candidate_decisions"][unit.unit_id][must_line] = None  # null = 未决定
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("缺少明确决定" in e for e in errors)

    def test_removed_decision_key_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        must_line = next(str(c.line) for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM))
        del doc["candidate_decisions"][unit.unit_id][must_line]
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("缺少明确决定" in e for e in errors)

    def test_unknown_unit_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["candidate_decisions"]["vol99_不存在"] = {}
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("未知单元" in e for e in errors)

    def test_incomplete_unit_coverage_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        del doc["candidate_decisions"]["vol01_1翡翠的排挤原理"]
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("缺少单元" in e for e in errors)

    def test_decision_on_non_candidate_line_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["candidate_decisions"][unit.unit_id]["9999"] = DECISION_BOUNDARY
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("不属于该单元" in e for e in errors)

    def test_invalid_decision_value_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        must_line = next(str(c.line) for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM))
        doc["candidate_decisions"][unit.unit_id][must_line] = "maybe"
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("非法" in e for e in errors)

    def test_add_conflict_with_decision_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        must_line = next(c.line for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM))
        doc["adds"][unit.unit_id] = [must_line]  # 与决定冲突
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("冲突" in e for e in errors)

    def test_add_inside_dialogue_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        dialogue = next(s for s in segments if s.segment_type is SegmentType.dialogue)
        doc["adds"][unit.unit_id] = [dialogue.source.line_end]  # dialogue 末行
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("dialogue 段内部" in e or "不合法" in e for e in errors)

    def test_add_on_blank_line_rejected(self):
        text = _six_turn_text("", "\n[琉璃] 「隔了一场。」\n")
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        grouped = {u.unit_id: [] for u in STORY_UNITS}
        grouped[unit.unit_id] = segments
        candidates_by_unit = {u.unit_id: [] for u in STORY_UNITS}
        candidates_by_unit[unit.unit_id] = candidates
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["adds"][unit.unit_id] = [7]  # 第 7 行是空行（不属于任何段）
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("不在任何已解析段的覆盖范围内" in e for e in errors)

    def test_min_dialogue_turns_mismatch_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit, expected_min_dialogue_turns=5)
        assert any("不一致" in e for e in errors)


class TestStoryUnits:
    def test_registry_shape(self):
        assert len(STORY_UNITS) == 18
        scopes = [u.content_scope for u in STORY_UNITS]
        assert sum(1 for s in scopes if s is ContentScope.main_story) == 16
        assert sum(1 for s in scopes if s is ContentScope.promotional_meta) == 1
        assert sum(1 for s in scopes if s is ContentScope.bonus_story) == 1
        assert all(u.viewpoint is None for u in STORY_UNITS)  # 未核实不臆填
        assert all(u.volume_number is not None for u in STORY_UNITS if u.content_scope is ContentScope.main_story)

    def test_epilogue_fixed_boundary(self):
        meta = unit_by_id("epilogue_meta")
        bonus = unit_by_id("epilogue_bonus")
        assert (meta.line_start, meta.line_end) == (1, EPILOGUE_FIXED_BOUNDARY - 1)
        assert (bonus.line_start, bonus.line_end) == (EPILOGUE_FIXED_BOUNDARY, None)

    def test_split_rejects_unknown_source_path(self):
        segments = parse_script_text("[妃] 「喵。」\n", source_path="wrong/path.txt")
        with pytest.raises(ValueError, match="未在登记表"):
            split_segments_by_unit(segments)


# ============================================================
# 全语料回归
# ============================================================


@pytest.fixture(scope="module")
def corpus_plan():
    segments = parse_script_directory(GAME_ROOT, source_prefix=GAME_REL)
    grouped = split_segments_by_unit(segments)
    lines_cache: dict[str, list[str]] = {}
    result = {}
    for unit in STORY_UNITS:
        unit_segments = grouped[unit.unit_id]
        candidates = detect_scene_boundaries(unit, unit_segments)
        plan = plan_scene_boundaries(unit, unit_segments, candidates)
        if unit.source_path not in lines_cache:
            lines_cache[unit.source_path] = _load_source_lines(GAME_ROOT, unit.source_path, GAME_REL)
        scenes = build_scene_documents(unit, unit_segments, plan, source_lines=lines_cache[unit.source_path])
        result[unit.unit_id] = (unit, unit_segments, candidates, plan, scenes)
    return segments, grouped, result, lines_cache


@needs_corpus
class TestCorpusScenes:
    def test_all_segments_assigned_once(self, corpus_plan):
        segments, grouped, result, _ = corpus_plan
        assert sum(len(v) for v in grouped.values()) == len(segments) == 28506
        assert set(grouped) == {u.unit_id for u in STORY_UNITS}

    def test_preview_counts_unchanged(self, corpus_plan):
        """P3.1 不改变未应用人工决定前的预览口径。

        日后谈末行 DOS EOF 标记（\x1a）被忽略后：候选 758→757、自动合并 10→9
        （原 L879 blank_line 候选消失）；生效边界与场景数不变。
        """
        _, _, result, _ = corpus_plan
        total_candidates = sum(len(c) for _, _, c, _, _ in result.values())
        total_effective = sum(len(p.boundaries) for _, _, _, p, _ in result.values())
        total_merged = sum(len(p.auto_merged) for _, _, _, p, _ in result.values())
        total_scenes = sum(len(s) for _, _, _, _, s in result.values())
        assert total_candidates == 757
        assert total_effective == 76
        assert total_merged == 9
        assert total_scenes == 94

    def test_dialogue_and_kisaki_counts_unchanged(self, corpus_plan):
        _, grouped, _, _ = corpus_plan
        dialogues = [s for segs in grouped.values() for s in segs if s.segment_type is SegmentType.dialogue]
        assert len(dialogues) == 17530
        assert sum(1 for s in dialogues if s.speaker == "妃") == 1598

    def test_epilogue_units_respect_fixed_boundary(self, corpus_plan):
        _, grouped, _, _ = corpus_plan
        meta = grouped["epilogue_meta"]
        bonus = grouped["epilogue_bonus"]
        assert max(s.source.line_end for s in meta) <= EPILOGUE_FIXED_BOUNDARY - 1
        assert min(s.source.line_start for s in bonus) >= EPILOGUE_FIXED_BOUNDARY

    def test_bonus_unit_first_scene_starts_at_content_line(self, corpus_plan):
        _, _, result, _ = corpus_plan
        scenes = result["epilogue_bonus"][4]
        assert scenes[0].source.line_start == EPILOGUE_FIXED_BOUNDARY + 1  # 第 39 行

    def test_no_candidate_inside_any_dialogue(self, corpus_plan):
        _, grouped, result, _ = corpus_plan
        for unit_id, (_, unit_segments, candidates, _, _) in result.items():
            spans = [
                (s.source.line_start, s.source.line_end)
                for s in unit_segments
                if s.segment_type is SegmentType.dialogue
            ]
            for cand in candidates:
                for start, end in spans:
                    assert not (start < cand.line <= end), f"{unit_id} 候选 L{cand.line} 锚进 dialogue L{start}-{end}"

    def test_dialogue_never_split_across_scenes(self, corpus_plan):
        _, _, result, _ = corpus_plan
        for unit_id, (_, unit_segments, _, _plan, scenes) in result.items():
            for seg in unit_segments:
                if seg.segment_type is not SegmentType.dialogue:
                    continue
                owners = [s for s in scenes if s.source.line_start <= seg.source.line_start <= s.source.line_end]
                assert len(owners) == 1, f"{unit_id} dialogue L{seg.source.line_start} 场景归属数 {len(owners)}"

    def test_scene_spans_ordered_and_covering(self, corpus_plan):
        _, _, result, _ = corpus_plan
        for _unit_id, (_unit, unit_segments, _, _plan, scenes) in result.items():
            prev_end = unit_segments[0].source.line_start - 1
            for scene in scenes:
                assert scene.source.line_start > prev_end
                assert scene.source.line_start <= scene.source.line_end
                prev_end = scene.source.line_end
            assert prev_end == unit_segments[-1].source.line_end

    def test_min_turns_satisfied_or_single_scene(self, corpus_plan):
        from knowledge.game_rag.scene_segmenter import _dialogue_turns_per_scene

        _, _, result, _ = corpus_plan
        for unit_id, (_, unit_segments, _, plan, scenes) in result.items():
            if len(scenes) <= 1:
                continue
            turns = _dialogue_turns_per_scene(unit_segments, plan.boundaries)
            small = [i for i, t in enumerate(turns) if t < DEFAULT_MIN_DIALOGUE_TURNS]
            assert small == [], f"{unit_id} 存在低于最小轮数的场景: {small}"

    def test_scene_story_context_defaults(self, corpus_plan):
        _, _, result, _ = corpus_plan
        for _unit_id, (unit, _, _, _, scenes) in result.items():
            for scene in scenes:
                assert scene.story.story_unit_id == unit.unit_id
                assert scene.story.content_scope is unit.content_scope
                assert scene.story.route is None
                assert scene.story.continuity_id is None
                assert scene.story.temporal_scope is None  # P3.1
                assert scene.review_status.value == "draft"

    def test_full_pipeline_deterministic(self, corpus_plan):
        _, _, result, lines_cache = corpus_plan
        for unit in STORY_UNITS:
            unit_segments = split_segments_by_unit(parse_script_directory(GAME_ROOT, source_prefix=GAME_REL))[
                unit.unit_id
            ]
            candidates = detect_scene_boundaries(unit, unit_segments)
            plan = plan_scene_boundaries(unit, unit_segments, candidates)
            scenes = build_scene_documents(unit, unit_segments, plan, source_lines=lines_cache[unit.source_path])
            _, _, _, first_plan, first_scenes = result[unit.unit_id]
            assert plan == first_plan
            assert [s.model_dump() for s in scenes] == [s.model_dump() for s in first_scenes]

    def test_scene_text_verbatim_against_source_lines(self, corpus_plan):
        """P3.1 核心：每个 scene.text 与原文行范围 LF 规范化后逐字一致。"""
        _, _, result, lines_cache = corpus_plan
        checked = 0
        for unit_id, (unit, _, _, _, scenes) in result.items():
            lines = lines_cache[unit.source_path]
            for scene in scenes:
                expected = "\n".join(lines[scene.source.line_start - 1 : scene.source.line_end])
                assert scene.text == expected, f"{unit_id} L{scene.source.line_start} 场景文本与原文不一致"
                checked += 1
        assert checked == 94

    def test_verbatim_covers_unclosed_and_duplicate_quote(self, corpus_plan):
        """7 条未闭合台词与重复闭引号所在行的原文完整进入场景文本。"""
        _, _, result, lines_cache = corpus_plan
        targets = [
            ("vol12_12青金石的幻想图书馆", 2485, 2487),  # 未闭合（夜子）
            ("vol01_1翡翠的排挤原理", 3892, 3893),  # 未闭合（岬）
            ("vol02_2红宝石的天作之合", 70, 71),  # 未闭合（暗子）
            ("vol04_4紫水晶的怪异传说", 326, 327),  # 未闭合（彼方）
            ("vol06_6芙蓉石的终焉轮回", 228, 229),  # 未闭合（理央）
            ("vol09_9白珍珠的泡沫爱慕", 1641, 1644),  # 未闭合（琉璃）
            ("epilogue_bonus", 183, 185),  # 未闭合（彼方，日后谈行号）
            ("vol12_12青金石的幻想图书馆", 2540, 2540),  # 重复闭引号
        ]
        for unit_id, start, end in targets:
            unit, _, _, _, scenes = result[unit_id]
            lines = lines_cache[unit.source_path]
            owner = [s for s in scenes if s.source.line_start <= start and end <= s.source.line_end]
            assert len(owner) == 1, f"{unit_id} L{start}-{end} 场景归属数 {len(owner)}"
            text_lines = owner[0].text.split("\n")
            offset = owner[0].source.line_start
            for line_no in range(start, end + 1):
                assert text_lines[line_no - offset] == lines[line_no - 1], f"{unit_id} L{line_no} 与原文不一致"

    def test_merged_blank_line_boundary_restored(self, corpus_plan):
        """被自动合并的空行边界：空行行保留在合并场景文本中。"""
        _, _, result, _ = corpus_plan
        found = 0
        for unit_id, (_unit, _, candidates, plan, scenes) in result.items():
            merged_blank = {c.line for c in candidates if c.line in plan.auto_merged and c.signal == SIGNAL_BLANK_LINE}
            for anchor in merged_blank:
                blank_line_no = anchor - 1  # blank_line 候选锚点 = 空行后首段起始
                owners = [s for s in scenes if s.source.line_start <= blank_line_no <= s.source.line_end]
                assert len(owners) == 1, f"{unit_id} 空行 L{blank_line_no} 归属数 {len(owners)}"
                scene = owners[0]
                assert scene.text.split("\n")[blank_line_no - scene.source.line_start] == ""
                found += 1
        assert found > 0  # 语料中确实存在被合并的空行边界

    def test_kisaki_multiline_verbatim(self, corpus_plan):
        """妃的 3 条合法跨行台词在场景文本中保持原文两行。"""
        _, _, result, lines_cache = corpus_plan
        cases = [
            ("vol01_1翡翠的排挤原理", 46, 47),
            ("vol01_1翡翠的排挤原理", 59, 60),
            ("vol03_3蓝宝石的存在证明", 2536, 2537),
        ]
        for unit_id, start, end in cases:
            unit, _, _, _, scenes = result[unit_id]
            lines = lines_cache[unit.source_path]
            owner = [s for s in scenes if s.source.line_start <= start and end <= s.source.line_end]
            assert len(owner) == 1
            text_lines = owner[0].text.split("\n")
            offset = owner[0].source.line_start
            assert text_lines[start - offset] == lines[start - 1]
            assert text_lines[end - offset] == lines[end - 1]


# ============================================================
# 决定模式全语料回归（P3.2 low 审核合并后的真实决定文件）
# ============================================================

REVIEW_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "knowledge" / "tsukiyashiro_kisaki" / "scene_boundary_review"
)

needs_review_files = pytest.mark.skipif(
    not (REVIEW_DIR / "boundary_overrides.json").exists() or not (REVIEW_DIR / "low_candidate_review.json").exists(),
    reason="审核决定文件不存在（data/knowledge 未随测试环境分发）",
)


@pytest.fixture(scope="module")
def decision_corpus(corpus_plan):
    """决定模式全语料重建：加载真实 overrides/low 审核决定并重建场景。"""
    _, grouped, result, lines_cache = corpus_plan
    overrides_doc = json.loads((REVIEW_DIR / "boundary_overrides.json").read_text(encoding="utf-8"))
    review_doc = load_low_review_document(REVIEW_DIR / "low_candidate_review.json")
    plans = {}
    scenes_by_unit = {}
    for unit in STORY_UNITS:
        _u, unit_segments, candidates, _, _ = result[unit.unit_id]
        plan = plan_scene_boundaries_from_decisions(unit, unit_segments, candidates, overrides_doc)
        plans[unit.unit_id] = plan
        scenes_by_unit[unit.unit_id] = build_scene_documents(
            unit, unit_segments, plan, source_lines=lines_cache[unit.source_path]
        )
    return overrides_doc, review_doc, grouped, result, lines_cache, plans, scenes_by_unit


@needs_corpus
@needs_review_files
class TestCorpusDecisionMode:
    def test_low_review_state_complete(self, decision_corpus):
        """672 个 low 候选全覆盖：必审 579 + 抽检 36 无未决；53 个范围外候选显式 null。"""
        _, review_doc, _, result, _, _, _ = decision_corpus
        assert review_doc["review_status"] == "approved"  # 已获项目负责人正式冻结授权
        candidates_by_unit = {uid: c for uid, (_, _, c, _, _) in result.items()}
        assert validate_low_review_document(review_doc, candidates_by_unit, require_complete=True) == []

        decisions = review_doc["candidate_decisions"]
        counts = {"boundary": 0, "no_boundary": 0, None: 0}
        for v in decisions.values():
            for x in v.values():
                counts[x] += 1
        scope = sum(len(v) for v in review_doc["oversized_scope"].values())
        sampled = sum(len(v) for v in review_doc["external_sampling"]["sampled_lines"].values())
        assert sum(counts.values()) == 672
        assert counts["boundary"] == 13
        assert counts["no_boundary"] == 606
        assert counts[None] == 53  # 审核范围外候选：显式保留 null 而非缺失
        assert scope == 579  # oversized 必审范围
        assert sampled == 36  # 外部分层抽检
        # 范围外另有 4 条顺带决定的候选（vol01 L2545/L2594、vol02 L1101、vol12 L670，均 no_boundary）
        assert scope + sampled + counts[None] + 4 == 672

    def test_overrides_valid_draft_and_merged(self, decision_corpus):
        """决定模式校验通过、状态保持 draft；low 决定与人工锚点全部并入 adds（共 197）。"""
        overrides_doc, review_doc, grouped, result, _, _, _ = decision_corpus
        candidates_by_unit = {uid: c for uid, (_, _, c, _, _) in result.items()}
        errors = validate_boundary_overrides(overrides_doc, grouped, candidates_by_unit, allow_draft=True)
        assert errors == []
        assert overrides_doc["boundary_review_status"] == "approved"  # 已获项目负责人正式冻结授权
        adds_map = overrides_doc.get("adds", {})
        assert sum(len(v) for v in adds_map.values()) == 197
        for uid, decisions in review_doc["candidate_decisions"].items():
            expected = {int(k) for k, v in decisions.items() if v == "boundary"} | set(
                review_doc.get("replacement_adds", {}).get(uid, [])
            )
            actual = {int(x) for x in adds_map.get(uid, [])}
            assert expected <= actual, f"{uid}: low boundary/replacement 未全部进入 adds"

    def test_decision_mode_scene_counts_and_conservation(self, decision_corpus):
        """合并后 262 场景；span 有序覆盖无重叠；每个解析段归属唯一；0x1A 不进场景文本。"""
        _, _, grouped, result, _, _, scenes_by_unit = decision_corpus
        assert sum(len(s) for s in scenes_by_unit.values()) == 262
        for uid, scenes in scenes_by_unit.items():
            unit_segments = grouped[uid]
            prev_end = unit_segments[0].source.line_start - 1
            for scene in scenes:
                assert scene.source.line_start > prev_end
                assert scene.source.line_start <= scene.source.line_end
                assert "\x1a" not in scene.text
                prev_end = scene.source.line_end
            assert prev_end == unit_segments[-1].source.line_end
            for seg in unit_segments:
                for line in range(seg.source.line_start, seg.source.line_end + 1):
                    owners = [s for s in scenes if s.source.line_start <= line <= s.source.line_end]
                    assert len(owners) == 1, f"{uid} parsed line L{line} 归属数 {len(owners)}"

    def test_oversized_stats_file_consistent(self, decision_corpus):
        """oversized_stats.json 与决定模式重算一致：262 场景、24 个 oversized、133 个残留 low 候选。"""
        from knowledge.game_rag.scene_segmenter import _dialogue_turns_per_scene

        _, _, _, result, _, plans, scenes_by_unit = decision_corpus
        stats = json.loads((REVIEW_DIR / "oversized_stats.json").read_text(encoding="utf-8"))
        oversized = 0
        residual_low = 0
        for uid, scenes in scenes_by_unit.items():
            _, unit_segments, candidates, _, _ = result[uid]
            low_lines = {c.line for c in candidates if c.confidence == CONFIDENCE_LOW}
            turns = _dialogue_turns_per_scene(unit_segments, plans[uid].boundaries)
            for i, scene in enumerate(scenes):
                if (scene.source.line_end - scene.source.line_start + 1) > 300 or turns[i] > 150:
                    oversized += 1
                    residual_low += sum(1 for ln in low_lines if scene.source.line_start <= ln <= scene.source.line_end)
        assert stats["total_scenes"] == 262
        assert stats["total_oversized_scenes"] == oversized == 24
        assert stats["total_oversized_low_candidates"] == residual_low == 133

    def test_vol12_perspective_switch_scene_split(self, decision_corpus):
        """P3.5 回归：琉璃→克丽索贝莉露第一人称视角切换（L672）必须独立成景。

        抽查发现场景 L459-754（296 行，恰低于 oversized 阈值）内部隐藏视角切换：
        L672'我无数次重复低语'起为克丽索贝莉露叙述（与 L691 同源），L676「笨女孩。」
        视角明确。候选 L670 本身非切换点，准确锚点以 replacement 记录，防止同类漏切复发。
        """
        overrides_doc, review_doc, _, _, _, _, scenes_by_unit = decision_corpus
        uid = "vol12_12青金石的幻想图书馆"
        assert 672 in overrides_doc["adds"][uid]
        assert 672 in review_doc["replacement_adds"][uid]
        assert review_doc["replacement_reasons"][uid]["672"].strip()
        assert review_doc["candidate_decisions"][uid]["670"] == "no_boundary"
        ranges = [(s.source.line_start, s.source.line_end) for s in scenes_by_unit[uid]]
        assert (459, 671) in ranges  # 琉璃/彼方现实侧：拒绝、探询与决意
        assert (672, 754) in ranges  # 克丽索贝莉露书房视角：旁观夜子自我封闭并决意实现其欲望

    def test_known_non_candidate_transitions_are_split(self, decision_corpus):
        """外部复核：已明确记录的非候选切换必须落实为人工锚点。"""
        overrides_doc, review_doc, _, _, _, _, scenes_by_unit = decision_corpus
        expected = {
            "vol04_4紫水晶的怪异传说": (3315, (2756, 3314), (3315, 3413)),
            "epilogue_bonus": (758, (632, 757), (758, 877)),
        }
        for uid, (anchor, before, after) in expected.items():
            assert anchor in overrides_doc["adds"][uid]
            assert anchor in review_doc["replacement_adds"][uid]
            assert review_doc["replacement_reasons"][uid][str(anchor)].strip()
            ranges = {(scene.source.line_start, scene.source.line_end) for scene in scenes_by_unit[uid]}
            assert before in ranges
            assert after in ranges


@needs_corpus
@needs_review_files
class TestControlledFreezeGate:
    """公开冻结入口必须同时验证 high/medium 与 low 两层审核。"""

    @staticmethod
    def _approved_docs(decision_corpus):
        overrides_doc, review_doc, *_ = decision_corpus
        overrides = json.loads(json.dumps(overrides_doc, ensure_ascii=False))
        review = json.loads(json.dumps(review_doc, ensure_ascii=False))
        overrides["boundary_review_status"] = "approved"
        review["review_status"] = "approved"
        return overrides, review

    def test_missing_low_document_rejected_without_writes(self, decision_corpus, tmp_path):
        overrides, _ = self._approved_docs(decision_corpus)

        with pytest.raises(ValueError, match="low review document is required"):
            freeze_reviewed_scenes(GAME_ROOT, overrides, None, tmp_path)

        assert not list(tmp_path.iterdir())

    def test_low_draft_rejected_without_writes(self, decision_corpus, tmp_path):
        overrides, review = self._approved_docs(decision_corpus)
        review["review_status"] = "draft"

        with pytest.raises(ValueError, match="low review_status must be approved"):
            freeze_reviewed_scenes(GAME_ROOT, overrides, review, tmp_path)

        assert not list(tmp_path.iterdir())

    def test_incomplete_required_scope_rejected_without_writes(self, decision_corpus, tmp_path):
        overrides, review = self._approved_docs(decision_corpus)
        uid = next(uid for uid, lines in review["oversized_scope"].items() if lines)
        line = review["oversized_scope"][uid][0]
        review["candidate_decisions"][uid][str(line)] = None

        with pytest.raises(ValueError, match="required low candidate is undecided"):
            freeze_reviewed_scenes(GAME_ROOT, overrides, review, tmp_path)

        assert not list(tmp_path.iterdir())

    def test_merged_adds_mismatch_rejected_without_writes(self, decision_corpus, tmp_path):
        overrides, review = self._approved_docs(decision_corpus)
        uid = next(
            uid
            for uid, decisions in review["candidate_decisions"].items()
            if any(value == "boundary" for value in decisions.values())
        )
        line = next(int(key) for key, value in review["candidate_decisions"][uid].items() if value == "boundary")
        overrides["adds"][uid].remove(line)

        with pytest.raises(ValueError, match="exactly include reviewed low boundaries"):
            freeze_reviewed_scenes(GAME_ROOT, overrides, review, tmp_path)

        assert not list(tmp_path.iterdir())

    def test_approved_exact_reviews_reach_atomic_freeze(self, decision_corpus, tmp_path):
        overrides, review = self._approved_docs(decision_corpus)

        manifest = freeze_reviewed_scenes(GAME_ROOT, overrides, review, tmp_path)

        assert manifest["total_scenes"] == 262
        assert (tmp_path / "scenes.jsonl").exists()
        assert (tmp_path / "boundary_manifest.json").exists()


@needs_corpus
class TestFreezeScenes:
    def _full_overrides(self, corpus_plan, decision=DECISION_NO_BOUNDARY):
        _, _, result, _ = corpus_plan
        decisions = {}
        for uid, (_, _, candidates, _, _) in result.items():
            must = {str(c.line) for c in candidates if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM)}
            decisions[uid] = {line: decision for line in must}
        return {
            "schema_version": 2,
            "boundary_review_status": "approved",
            "reviewer": "audit-tester",
            "min_dialogue_turns": DEFAULT_MIN_DIALOGUE_TURNS,
            "source_prefix": GAME_REL,
            "candidate_decisions": decisions,
            "adds": {uid: [] for uid in result},
            "notes": "test",
        }

    def test_freeze_no_boundary_decisions(self, corpus_plan, tmp_path):
        """全部 no_boundary：每单元单场景（18 个），review_status 保持 draft。"""
        doc = self._full_overrides(corpus_plan, DECISION_NO_BOUNDARY)
        manifest = freeze_scenes(GAME_ROOT, doc, tmp_path)
        assert manifest["total_scenes"] == 18
        assert manifest["boundary_review_status"] == "approved"
        assert manifest["reviewer"] == "audit-tester"
        assert manifest["scene_review_status"] == "draft"
        lines = (tmp_path / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 18
        for line in lines:
            record = json.loads(line)
            assert record["document_type"] == "scene"
            assert record["review_status"] == "draft"  # 边界冻结≠内容审核
            assert record["story"]["temporal_scope"] is None
        assert (tmp_path / "boundary_manifest.json").exists()

    def test_freeze_all_boundary_decisions(self, corpus_plan, tmp_path):
        """全部 boundary：85 个 high/medium 候选全保留（决定模式不做自动合并）→ 103 场景。

        预览口径的 76 生效 = 85 候选 − 9 自动合并；决定模式不做合并，故场景数更多。
        （日后谈末行 DOS EOF 标记 \x1a 被解析器忽略后，原 L879 blank_line 候选消失，
        必审候选 86 → 85、auto_merged 10 → 9。）
        """
        doc = self._full_overrides(corpus_plan, DECISION_BOUNDARY)
        manifest = freeze_scenes(GAME_ROOT, doc, tmp_path)
        assert manifest["total_scenes"] == 103

    def test_freeze_draft_rejected(self, corpus_plan, tmp_path):
        doc = self._full_overrides(corpus_plan)
        doc["boundary_review_status"] = "draft"
        with pytest.raises(ValueError, match="boundary_review_status"):
            freeze_scenes(GAME_ROOT, doc, tmp_path)
        assert not (tmp_path / "scenes.jsonl").exists()

    def test_freeze_missing_decision_rejected(self, corpus_plan, tmp_path):
        doc = self._full_overrides(corpus_plan)
        uid, decisions = next(iter(doc["candidate_decisions"].items()))
        decisions[next(iter(decisions))] = None
        with pytest.raises(ValueError, match="缺少明确决定"):
            freeze_scenes(GAME_ROOT, doc, tmp_path)
        assert not (tmp_path / "scenes.jsonl").exists()

    def test_freeze_min_turns_mismatch_rejected(self, corpus_plan, tmp_path):
        """min_dialogue_turns 与审核包不一致时拒绝冻结。"""
        doc = self._full_overrides(corpus_plan)
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (review_dir / "boundary_stats.json").write_text(json.dumps({"min_dialogue_turns": 5}), encoding="utf-8")
        with pytest.raises(ValueError, match="不一致"):
            freeze_scenes(GAME_ROOT, doc, tmp_path / "out", review_dir=review_dir)

    def test_freeze_scenes_verbatim(self, corpus_plan, tmp_path):
        """冻结产物逐字一致：scenes.jsonl 文本与源文件行完全对应。"""
        doc = self._full_overrides(corpus_plan, DECISION_BOUNDARY)
        freeze_scenes(GAME_ROOT, doc, tmp_path)
        lines_cache: dict[str, list[str]] = {}
        for line in (tmp_path / "scenes.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            path = record["source"]["source_path"]
            if path not in lines_cache:
                lines_cache[path] = _load_source_lines(GAME_ROOT, path, GAME_REL)
            src = lines_cache[path]
            expected = "\n".join(src[record["source"]["line_start"] - 1 : record["source"]["line_end"]])
            assert record["text"] == expected


class TestFreezePairAtomic:
    """P3.2：两份冻结文件的整体原子提交（故障注入）。"""

    def _write_pair(self, out_dir: Path, scenes_text: str, manifest_text: str) -> None:
        from knowledge.game_rag.scene_segmenter import _freeze_pair_atomic

        _freeze_pair_atomic(out_dir, scenes_payload=scenes_text, manifest_payload=manifest_text)

    def test_both_files_written_on_success(self, tmp_path):
        self._write_pair(tmp_path, "s1\n", "m1\n")
        assert (tmp_path / "scenes.jsonl").read_text(encoding="utf-8") == "s1\n"
        assert (tmp_path / "boundary_manifest.json").read_text(encoding="utf-8") == "m1\n"

    def test_no_temp_files_left_after_success(self, tmp_path):
        self._write_pair(tmp_path, "s1\n", "m1\n")
        assert not list(tmp_path.glob("*.tmp"))
        assert not list(tmp_path.glob("*.tmp.old"))

    def test_second_success_replaces_pair_without_leaving_backup(self, tmp_path):
        self._write_pair(tmp_path, "s1\n", "m1\n")
        self._write_pair(tmp_path, "s2\n", "m2\n")
        assert (tmp_path / "scenes.jsonl").read_text(encoding="utf-8") == "s2\n"
        assert (tmp_path / "boundary_manifest.json").read_text(encoding="utf-8") == "m2\n"
        assert not list(tmp_path.glob("*.tmp"))
        assert not list(tmp_path.glob("*.tmp.old"))

    def test_existing_recovery_backup_is_not_overwritten(self, tmp_path):
        backup = tmp_path / "scenes.jsonl.tmp.old"
        backup.write_text("RECOVERY COPY\n", encoding="utf-8")
        with pytest.raises(ValueError, match="未恢复的旧版备份"):
            self._write_pair(tmp_path, "NEW\n", "NEW_M\n")
        assert backup.read_text(encoding="utf-8") == "RECOVERY COPY\n"
        assert not list(tmp_path.glob("*.tmp"))

    def test_second_replace_failure_rolls_back_scenes(self, tmp_path, monkeypatch):
        """manifest 替换失败：新 scenes 不得生效，旧版本保留（不留混合版本）。"""
        self._write_pair(tmp_path, "OLD_SCENES\n", "OLD_MANIFEST\n")  # 先冻结一版
        import knowledge.game_rag.scene_segmenter as seg

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if str(dst).endswith("boundary_manifest.json"):
                raise OSError("injected: manifest replace failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(seg.os, "replace", failing_replace)
        with pytest.raises(ValueError, match="已回滚"):
            self._write_pair(tmp_path, "NEW_SCENES\n", "NEW_MANIFEST\n")
        monkeypatch.undo()
        # 旧组合完整保留：新 scenes + 旧 manifest 的混合版本不存在
        assert (tmp_path / "scenes.jsonl").read_text(encoding="utf-8") == "OLD_SCENES\n"
        assert (tmp_path / "boundary_manifest.json").read_text(encoding="utf-8") == "OLD_MANIFEST\n"
        assert not list(tmp_path.glob("*.tmp"))
        assert not list(tmp_path.glob("*.tmp.old"))

    def test_second_replace_failure_without_old_scenes_removes_new(self, tmp_path, monkeypatch):
        """首次冻结时 manifest 替换失败：不留孤儿 scenes.jsonl。"""
        import knowledge.game_rag.scene_segmenter as seg

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if str(dst).endswith("boundary_manifest.json"):
                raise OSError("injected: manifest replace failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(seg.os, "replace", failing_replace)
        with pytest.raises(ValueError, match="已回滚"):
            self._write_pair(tmp_path, "NEW_SCENES\n", "NEW_MANIFEST\n")
        monkeypatch.undo()
        assert not (tmp_path / "scenes.jsonl").exists()
        assert not (tmp_path / "boundary_manifest.json").exists()

    def test_first_stage_write_failure_keeps_old(self, tmp_path, monkeypatch):
        """阶段 1（写 tmp）失败：旧文件完全未动。"""
        self._write_pair(tmp_path, "OLD\n", "OLD_M\n")
        real_write_text = Path.write_text

        def failing_write_text(self_path, data, *args, **kwargs):
            if str(self_path).endswith("boundary_manifest.json.tmp"):
                raise OSError("injected: tmp write failure")
            return real_write_text(self_path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)
        with pytest.raises(OSError):
            self._write_pair(tmp_path, "NEW\n", "NEW_M\n")
        monkeypatch.undo()
        assert (tmp_path / "scenes.jsonl").read_text(encoding="utf-8") == "OLD\n"
        assert (tmp_path / "boundary_manifest.json").read_text(encoding="utf-8") == "OLD_M\n"
        assert not list(tmp_path.glob("*.tmp"))

    def test_rollback_failure_keeps_backup_and_reports_path(self, tmp_path, monkeypatch):
        """manifest 替换失败且 scenes 回滚也失败：备份必须保留（唯一可恢复副本），
        错误信息报告备份路径；不得静默删掉 .tmp.old 造成旧版本永久丢失。"""
        self._write_pair(tmp_path, "OLD_SCENES\n", "OLD_MANIFEST\n")  # 先冻结一版
        import knowledge.game_rag.scene_segmenter as seg

        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            # manifest 替换失败 + 备份恢复（回滚）失败：双重故障注入
            if str(dst).endswith("boundary_manifest.json"):
                raise OSError("injected: manifest replace failure")
            if str(src).endswith("scenes.jsonl.tmp.old"):
                raise OSError("injected: rollback failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(seg.os, "replace", failing_replace)
        with pytest.raises(ValueError, match="回滚失败"):
            self._write_pair(tmp_path, "NEW_SCENES\n", "NEW_MANIFEST\n")
        monkeypatch.undo()
        backup = tmp_path / "scenes.jsonl.tmp.old"
        # 备份保留且内容为旧版本：唯一可恢复副本未被 finally 误删
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "OLD_SCENES\n"
        # 现状为未回滚的混合版本（新 scenes + 旧 manifest），备份是唯一恢复途径
        assert (tmp_path / "scenes.jsonl").read_text(encoding="utf-8") == "NEW_SCENES\n"
        assert (tmp_path / "boundary_manifest.json").read_text(encoding="utf-8") == "OLD_MANIFEST\n"

    def test_rollback_retry_success_removes_backup(self, tmp_path, monkeypatch):
        """内层回滚失败但外层重试成功（瞬时故障）：旧 scenes 恢复原位，备份清理。"""
        self._write_pair(tmp_path, "OLD_SCENES\n", "OLD_MANIFEST\n")
        import knowledge.game_rag.scene_segmenter as seg

        real_replace = os.replace
        rollback_calls = {"n": 0}

        def failing_replace(src, dst, *args, **kwargs):
            if str(dst).endswith("boundary_manifest.json"):
                raise OSError("injected: manifest replace failure")
            if str(src).endswith("scenes.jsonl.tmp.old"):
                rollback_calls["n"] += 1
                if rollback_calls["n"] == 1:  # 内层首次回滚失败，外层重试成功
                    raise OSError("injected: transient rollback failure")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(seg.os, "replace", failing_replace)
        with pytest.raises(ValueError):
            self._write_pair(tmp_path, "NEW_SCENES\n", "NEW_MANIFEST\n")
        monkeypatch.undo()
        assert rollback_calls["n"] == 2
        # 旧组合完整恢复，无 tmp / 备份残留
        assert (tmp_path / "scenes.jsonl").read_text(encoding="utf-8") == "OLD_SCENES\n"
        assert (tmp_path / "boundary_manifest.json").read_text(encoding="utf-8") == "OLD_MANIFEST\n"
        assert not list(tmp_path.glob("*.tmp"))
        assert not list(tmp_path.glob("*.tmp.old"))


class TestSchemaVersionGate:
    """P3.2：schema_version==2 门禁与结构/行号规范化校验。"""

    def _context(self):
        text = _six_turn_text("", "次日，新场景。\n" + "".join(f"[琉璃] 「新{i}。」\n" for i in range(1, 7)))
        segments = _parse(text)
        unit = _unit()
        candidates = detect_scene_boundaries(unit, segments)
        grouped = {u.unit_id: [] for u in STORY_UNITS}
        grouped[unit.unit_id] = segments
        candidates_by_unit = {u.unit_id: [] for u in STORY_UNITS}
        candidates_by_unit[unit.unit_id] = candidates
        return unit, segments, candidates, grouped, candidates_by_unit

    def _full_doc(self, grouped, candidates_by_unit, decision=DECISION_NO_BOUNDARY):
        decisions = {}
        for uid, cands in candidates_by_unit.items():
            must = {str(c.line) for c in cands if c.confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM)}
            decisions[uid] = {line: decision for line in must}
        return {
            "schema_version": 2,
            "boundary_review_status": "approved",
            "reviewer": "tester",
            "min_dialogue_turns": 6,
            "candidate_decisions": decisions,
            "adds": {uid: [] for uid in grouped},
        }

    def test_v2_passes(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        assert (
            validate_boundary_overrides(self._full_doc(grouped, candidates_by_unit), grouped, candidates_by_unit) == []
        )

    def test_missing_schema_version_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        del doc["schema_version"]
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("schema_version" in e for e in errors)

    def test_wrong_schema_version_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["schema_version"] = 1
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("schema_version" in e for e in errors)

    def test_non_dict_candidate_decisions_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["candidate_decisions"] = ["not", "a", "dict"]
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("candidate_decisions 必须是对象" in e for e in errors)

    def test_non_dict_unit_decisions_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["candidate_decisions"][unit.unit_id] = ["boundary"]
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("单元决定必须是对象" in e for e in errors)

    def test_non_list_adds_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["adds"][unit.unit_id] = "9999"
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("adds 必须是数组" in e for e in errors)

    def test_non_canonical_line_key_rejected(self):
        """'01' 与 '1' 视为同一行的两种写法必须被拒绝（防键冲突绕过）。"""
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        # '01' 若被静默规范化为 1，会绕过 JSON 对象键唯一性
        doc["candidate_decisions"][unit.unit_id]["01"] = DECISION_BOUNDARY
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("不是规范十进制整数" in e for e in errors)

    def test_padded_add_line_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["adds"][unit.unit_id] = ["007"]
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("不是规范十进制整数" in e for e in errors)

    def test_duplicate_add_line_rejected(self):
        unit, segments, candidates, grouped, candidates_by_unit = self._context()
        doc = self._full_doc(grouped, candidates_by_unit)
        doc["adds"][unit.unit_id] = [999, 999]
        errors = validate_boundary_overrides(doc, grouped, candidates_by_unit)
        assert any("adds 含重复行号" in e for e in errors)

    def test_freeze_rejects_v1_doc(self, corpus_plan, tmp_path):
        """v1 决定文件即使字段齐全也进不了冻结。"""
        doc = TestFreezeScenes()._full_overrides(corpus_plan)
        doc["schema_version"] = 1
        with pytest.raises(ValueError, match="schema_version"):
            freeze_scenes(GAME_ROOT, doc, tmp_path)
        assert not (tmp_path / "scenes.jsonl").exists()


@needs_corpus
class TestReviewPackMutualExclusion:
    """P3.2：审核包 low 候选两组互斥 + oversized 内逐条展示。"""

    def _generate(self, tmp_path):
        from knowledge.game_rag.scene_segmenter import generate_review_materials

        generate_review_materials(GAME_ROOT, GAME_REL, tmp_path)
        stats = json.loads((tmp_path / "boundary_stats.json").read_text(encoding="utf-8"))
        md = (tmp_path / "scene_boundary_review.md").read_text(encoding="utf-8")
        return stats["units"], md

    def test_oversized_and_sampled_low_disjoint(self, tmp_path, corpus_plan):
        """oversized 展示的 low 候选与抽样建议的候选互斥，数量守恒。"""
        units, _md = self._generate(tmp_path)
        _, _, result, _ = corpus_plan
        oversized_total = 0
        remaining_total = 0
        for unit_id, unit_stats in units.items():
            _, _, candidates, _, _ = result[unit_id]
            low_lines = {c.line for c in candidates if c.confidence == CONFIDENCE_LOW}
            oversized_lines: set[int] = set()
            for scene in unit_stats["oversized_scenes"]:
                oversized_lines.update(scene["low_candidates"])
            assert oversized_lines <= low_lines, f"{unit_id} oversized 含非 low 候选"
            assert (
                unit_stats["oversized_low_candidate_count"]
                == len(oversized_lines)
                == sum(len(s["low_candidates"]) for s in unit_stats["oversized_scenes"])
            )
            assert unit_stats["remaining_low_candidate_count"] == len(low_lines) - len(oversized_lines)
            oversized_total += len(oversized_lines)
            remaining_total += unit_stats["remaining_low_candidate_count"]
        # 全语料口径：672 = oversized 内 + 其余
        assert oversized_total + remaining_total == 672
        assert oversized_total == 567  # 复核给出的实测值
        assert remaining_total == 105

    def test_oversized_low_items_have_context_and_decision_slot(self, tmp_path):
        """oversized 内每个 low 候选都有 ±2 行摘录与提升决定栏。"""
        _units, md = self._generate(tmp_path)
        lines = md.splitlines()
        items = [ln for ln in lines if ln.startswith("- L") and "｜low｜" in ln]
        assert len(items) == 567
        for idx, line in enumerate(lines):
            if line.startswith("- L") and "｜low｜" in line:
                assert "提升为边界（写入 adds）: ______" in line
                context = lines[idx + 1 : idx + 6]
                assert any(">>L" in c for c in context), f"缺少锚点摘录: {line}"

    def test_sample_lines_exclude_oversized(self, tmp_path, corpus_plan):
        """抽样建议行不包含 oversized 内的候选行号。"""
        units, md = self._generate(tmp_path)
        _, _, result, _ = corpus_plan
        oversized_all: dict[str, set[int]] = {}
        for unit_id, unit_stats in units.items():
            lines_set: set[int] = set()
            for scene in unit_stats["oversized_scenes"]:
                lines_set.update(scene["low_candidates"])
            oversized_all[unit_id] = lines_set
        sample_section = [s for s in md.split("## ") if s.startswith("low 候选分层抽样建议")][0]
        for row in sample_section.splitlines():
            if not row.startswith("- "):
                continue
            unit_id = row.split(" / ")[0][2:]
            import re as _re

            sampled = {int(x) for x in _re.findall(r"L(\d+)", row.split("建议抽检")[-1])}
            assert sampled <= set(range(1, 10000))
            if sampled:
                assert not (sampled & oversized_all[unit_id]), (
                    f"{unit_id} 抽样建议包含 oversized 内候选: {sorted(sampled & oversized_all[unit_id])}"
                )


@needs_corpus
class TestOversizedReviewPack:
    """P3.3：按最终决定（决定模式）重建场景并生成新口径 oversized 审核包。"""

    def _draft_doc(self, corpus_plan, decision=DECISION_NO_BOUNDARY):
        from knowledge.game_rag.scene_segmenter import generate_oversized_review_pack  # noqa: F401

        doc = TestFreezeScenes()._full_overrides(corpus_plan, decision)
        doc["boundary_review_status"] = "draft"  # 审核包重建先于 approved 冻结
        return doc

    def test_draft_complete_decisions_generate_pack(self, tmp_path, corpus_plan):
        """draft + 完整决定：生成 oversized_review.md / oversized_stats.json。"""
        from knowledge.game_rag.scene_segmenter import generate_oversized_review_pack

        doc = self._draft_doc(corpus_plan)
        units = generate_oversized_review_pack(GAME_ROOT, doc, tmp_path)
        stats = json.loads((tmp_path / "oversized_stats.json").read_text(encoding="utf-8"))
        assert stats["mode"] == "decisions"
        # 全部 no_boundary：决定模式无自动合并 → 每单元单场景（18 个）
        assert stats["total_scenes"] == 18
        assert stats["total_oversized_scenes"] == sum(len(u["oversized_scenes"]) for u in units.values())
        md = (tmp_path / "oversized_review.md").read_text(encoding="utf-8")
        assert "最终决定口径" in md
        assert "提升为边界（写入 adds）" in md
        # 决定口径统计行写明场景/oversized/low 候选总数
        assert f"场景 {stats['total_scenes']}" in md
        assert f"oversized {stats['total_oversized_scenes']}" in md

    def test_incomplete_decisions_rejected(self, tmp_path, corpus_plan):
        """必审候选决定缺失（null）：拒绝重建且不写出任何文件。"""
        from knowledge.game_rag.scene_segmenter import generate_oversized_review_pack

        doc = self._draft_doc(corpus_plan)
        first_uid = next(iter(doc["candidate_decisions"]))
        first_line = next(k for k, v in doc["candidate_decisions"][first_uid].items() if v is not None)
        doc["candidate_decisions"][first_uid][first_line] = None
        with pytest.raises(ValueError, match="缺少明确决定"):
            generate_oversized_review_pack(GAME_ROOT, doc, tmp_path)
        assert not (tmp_path / "oversized_review.md").exists()
        assert not (tmp_path / "oversized_stats.json").exists()

    def test_invalid_status_rejected(self, tmp_path, corpus_plan):
        """状态既非 approved 也非 draft：拒绝。"""
        from knowledge.game_rag.scene_segmenter import generate_oversized_review_pack

        doc = self._draft_doc(corpus_plan)
        doc["boundary_review_status"] = "pending"
        with pytest.raises(ValueError, match="boundary_review_status"):
            generate_oversized_review_pack(GAME_ROOT, doc, tmp_path)

    def test_decision_mode_recounts_oversized(self, tmp_path, corpus_plan):
        """全 boundary 决定（85 边界 + 18 单元起点 = 103 场景）：
        新口径 oversized 显著少于预览口径的 51，证明统计确按决定模式重建。"""
        from knowledge.game_rag.scene_segmenter import generate_oversized_review_pack

        doc = self._draft_doc(corpus_plan, DECISION_BOUNDARY)
        generate_oversized_review_pack(GAME_ROOT, doc, tmp_path)
        stats = json.loads((tmp_path / "oversized_stats.json").read_text(encoding="utf-8"))
        assert stats["total_scenes"] == 103
        assert stats["total_oversized_scenes"] < 51

    def test_adds_take_effect_in_pack(self, tmp_path, corpus_plan):
        """adds 参与决定模式重建：add 行成为边界并改变场景统计。"""
        from knowledge.game_rag.scene_segmenter import generate_oversized_review_pack

        doc = self._draft_doc(corpus_plan)  # 全 no_boundary：每单元 1 场景
        # vol01 L3-6 narration 块内部的 L4：非候选、非 dialogue 内部的合法 add 锚点
        doc["adds"]["vol01_1翡翠的排挤原理"] = [4]
        generate_oversized_review_pack(GAME_ROOT, doc, tmp_path)
        stats = json.loads((tmp_path / "oversized_stats.json").read_text(encoding="utf-8"))
        assert stats["total_scenes"] == 19  # 18 + 1 个 add 边界
        vol01 = stats["units"]["vol01_1翡翠的排挤原理"]
        assert 4 in vol01["boundaries"]
