"""P1 契约层单元测试：月社妃游戏 RAG 领域模型。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from knowledge.game_rag import (
    ChapterSummaryDocument,
    ContentScope,
    DocumentType,
    EventDocument,
    FactDocument,
    QuoteStyle,
    RealityStatus,
    RelationDocument,
    ReviewStatus,
    RouteType,
    SceneDocument,
    ScriptSegment,
    SegmentType,
    SourceSpan,
    StoryContext,
    TemporalScope,
)


@pytest.fixture()
def span() -> SourceSpan:
    return SourceSpan(source_path="gametext/纸上魔法使/1翡翠的排挤原理.txt", line_start=1, line_end=10)


@pytest.fixture()
def story() -> StoryContext:
    return StoryContext(
        volume_number=1,
        story_unit_id="unit_001",
        story_title="翡翠的排挤原理",
        content_scope=ContentScope.main_story,
        temporal_scope=TemporalScope.current,
    )


class TestEnums:
    def test_all_enum_members_create_and_serialize(self):
        """所有合法枚举成员可创建并序列化为字符串值。"""
        cases = [
            (SegmentType, ["dialogue", "narration"]),
            (QuoteStyle, ["corner", "double_corner", "curly", "none"]),
            (ContentScope, ["main_story", "bonus_story", "promotional_meta", "unknown"]),
            (TemporalScope, ["current", "flashback", "reconstruction", "hypothetical", "unknown"]),
            (RouteType, ["main", "branch", "parallel", "unknown"]),
            (RealityStatus, ["objective", "character_claim", "inferred", "fictional", "conflicted", "unknown"]),
            (ReviewStatus, ["draft", "needs_review", "approved", "rejected"]),
            (DocumentType, ["scene", "fact", "relation", "event", "chapter_summary"]),
        ]
        for enum_cls, expected_values in cases:
            members = list(enum_cls)
            assert [m.value for m in members] == expected_values
            for member in members:
                assert enum_cls(member.value) is member
                assert isinstance(member.value, str)

    def test_flashback_allowed_in_temporal_scope_but_rejected_by_route_type(self):
        """flashback 属于时间状态维度，不属于路线维度。"""
        assert TemporalScope("flashback") is TemporalScope.flashback
        with pytest.raises(ValueError):
            RouteType("flashback")

    def test_route_type_members_have_no_flashback(self):
        assert "flashback" not in {member.value for member in RouteType}


class TestStoryContext:
    def test_route_and_continuity_can_be_none(self, story):
        """route/continuity_id 为 None 表示尚未审核。"""
        assert story.route is None
        assert story.continuity_id is None

    def test_temporal_scope_defaults_to_none(self):
        """temporal_scope=None 表示尚未审核；unknown 表示已审核但无法判断（P3.1）。"""
        ctx = StoryContext(
            volume_number=1,
            story_unit_id="u1",
            story_title="测试",
            content_scope=ContentScope.main_story,
        )
        assert ctx.temporal_scope is None

    def test_temporal_scope_none_roundtrip(self, story):
        dumped = story.model_dump()
        assert dumped["temporal_scope"] is not None  # fixture 用显式枚举，保持兼容
        ctx = StoryContext.model_validate({**dumped, "temporal_scope": None})
        assert ctx.temporal_scope is None
        assert StoryContext.model_validate(ctx.model_dump()) == ctx

    def test_route_unknown_means_reviewed_but_undetermined(self, story):
        ctx = StoryContext.model_validate({**story.model_dump(), "route": RouteType.unknown})
        assert ctx.route is RouteType.unknown

    def test_volume_number_allows_none_for_bonus_story(self, story):
        ctx = StoryContext.model_validate(
            {**story.model_dump(), "volume_number": None, "content_scope": ContentScope.bonus_story}
        )
        assert ctx.volume_number is None

    def test_sequence_order_must_be_non_negative(self, story):
        with pytest.raises(ValidationError):
            StoryContext.model_validate({**story.model_dump(), "sequence_order": -1})

    def test_unknown_fields_rejected(self, story):
        with pytest.raises(ValidationError):
            StoryContext.model_validate({**story.model_dump(), "continuity": "主线"})


class TestSourceSpan:
    def test_line_start_zero_rejected(self):
        with pytest.raises(ValidationError):
            SourceSpan(source_path="a.txt", line_start=0, line_end=5)

    def test_line_end_before_start_rejected(self):
        with pytest.raises(ValidationError):
            SourceSpan(source_path="a.txt", line_start=10, line_end=9)

    def test_single_line_span_allowed(self):
        span = SourceSpan(source_path="a.txt", line_start=3, line_end=3)
        assert span.line_start == span.line_end == 3

    def test_empty_source_path_rejected(self):
        with pytest.raises(ValidationError):
            SourceSpan(source_path="  ", line_start=1, line_end=1)


class TestScriptSegment:
    def test_dialogue_with_speaker_passes(self, span):
        seg = ScriptSegment(
            id="seg_001",
            segment_type=SegmentType.dialogue,
            text="「我讨厌大海，受不了海风吹乱头发。」",
            source=span,
            speaker="妃",
            quote_style=QuoteStyle.corner,
        )
        assert seg.speaker == "妃"
        assert seg.warnings == []

    def test_dialogue_missing_speaker_rejected(self, span):
        with pytest.raises(ValidationError, match="speaker"):
            ScriptSegment(
                id="seg_001",
                segment_type=SegmentType.dialogue,
                text="「……」",
                source=span,
                quote_style=QuoteStyle.corner,
            )

    def test_narration_without_speaker_passes(self, span):
        seg = ScriptSegment(
            id="seg_002",
            segment_type=SegmentType.narration,
            text="她的名字叫月社妃。",
            source=span,
            quote_style=QuoteStyle.none,
        )
        assert seg.speaker is None

    def test_narration_with_speaker_rejected(self, span):
        """narration 携带 speaker 属于状态机分类错误，必须拒绝。"""
        with pytest.raises(ValidationError, match="speaker"):
            ScriptSegment(
                id="seg_005",
                segment_type=SegmentType.narration,
                text="她的名字叫月社妃。",
                source=span,
                speaker="妃",
                quote_style=QuoteStyle.none,
            )

    def test_narration_with_non_none_quote_style_rejected(self, span):
        with pytest.raises(ValidationError, match="quote_style"):
            ScriptSegment(
                id="seg_006",
                segment_type=SegmentType.narration,
                text="她的名字叫月社妃。",
                source=span,
                quote_style=QuoteStyle.corner,
            )

    def test_dialogue_with_none_quote_style_rejected(self, span):
        with pytest.raises(ValidationError, match="quote_style"):
            ScriptSegment(
                id="seg_007",
                segment_type=SegmentType.dialogue,
                text="「……」",
                source=span,
                speaker="妃",
                quote_style=QuoteStyle.none,
            )

    def test_unclosed_quote_dialogue_still_valid_with_warning(self, span):
        """未闭合引号的台词仍是 dialogue：保留原始文本，通过 warnings 记录。"""
        seg = ScriptSegment(
            id="seg_008",
            segment_type=SegmentType.dialogue,
            text="「就一点点，让我继续喜欢你一点点时间。",
            source=span,
            speaker="夜子",
            quote_style=QuoteStyle.corner,
            warnings=["unclosed_quote: truncated before next speaker tag"],
        )
        assert seg.text.startswith("「")
        assert seg.warnings == ["unclosed_quote: truncated before next speaker tag"]

    def test_empty_text_rejected(self, span):
        with pytest.raises(ValidationError):
            ScriptSegment(
                id="seg_003",
                segment_type=SegmentType.narration,
                text="   ",
                source=span,
                quote_style=QuoteStyle.none,
            )

    def test_extra_field_rejected(self, span):
        with pytest.raises(ValidationError):
            ScriptSegment(
                id="seg_004",
                segment_type=SegmentType.narration,
                text="正文",
                source=span,
                quote_style=QuoteStyle.none,
                parse_score=0.9,
            )


class TestContentGuards:
    """空关键内容必须失败的通用守卫。"""

    def test_scene_empty_text_rejected(self, story, span):
        with pytest.raises(ValidationError):
            SceneDocument(id="s1", title="场景", text=" ", story=story, source=span)

    def test_fact_empty_subject_rejected(self, story, span):
        with pytest.raises(ValidationError):
            FactDocument(
                id="f1",
                title="事实",
                subject="",
                predicate="讨厌",
                value="大海",
                summary="妃讨厌大海",
                evidence_text="原文",
                story=story,
                source=span,
                reality_status=RealityStatus.objective,
                review_status=ReviewStatus.draft,
            )

    def test_relation_empty_target_rejected(self, story, span):
        with pytest.raises(ValidationError):
            RelationDocument(
                id="r1",
                title="关系",
                subject="妃",
                relation="妹妹",
                target=" ",
                summary="妃是琉璃的妹妹",
                evidence_text="原文",
                story=story,
                source=span,
                reality_status=RealityStatus.objective,
                review_status=ReviewStatus.draft,
            )

    def test_event_empty_evidence_text_rejected(self, story, span):
        with pytest.raises(ValidationError):
            EventDocument(
                id="e1",
                title="事件",
                summary="概要",
                evidence_text="",
                story=story,
                source=span,
                reality_status=RealityStatus.objective,
                review_status=ReviewStatus.draft,
            )

    def test_chapter_summary_empty_summary_rejected(self, story, span):
        with pytest.raises(ValidationError):
            ChapterSummaryDocument(id="c1", title="第一章", summary="", story=story, source=span)


class TestDocumentRoundtrip:
    def _scene(self, story, span) -> SceneDocument:
        return SceneDocument(
            id="s1",
            title="妃与琉璃重逢",
            text="[妃] 「我讨厌大海，受不了海风吹乱头发。」",
            story=story,
            source=span,
            speakers=["妃"],
            mentioned_characters=["琉璃"],
            present_characters=["妃", "琉璃"],
            reality_status=RealityStatus.objective,
            review_status=ReviewStatus.approved,
        )

    def _fact(self, story, span) -> FactDocument:
        return FactDocument(
            id="f1",
            title="妃的头发",
            subject="妃",
            predicate="留有",
            value="长发",
            summary="妃留着一头让琉璃着迷的长发",
            evidence_text="[妃] 「不用你说我也知道。我的头发迷倒了琉璃。」",
            story=story,
            source=span,
            reality_status=RealityStatus.character_claim,
            review_status=ReviewStatus.needs_review,
        )

    def _relation(self, story, span) -> RelationDocument:
        return RelationDocument(
            id="r1",
            title="妃与琉璃",
            subject="妃",
            relation="亲妹妹",
            target="琉璃",
            summary="妃是琉璃的亲妹妹，判给分居双亲中的母亲",
            evidence_text="我的亲妹妹，判给分居双亲中母亲的妹妹。",
            story=story,
            source=span,
            reality_status=RealityStatus.objective,
            review_status=ReviewStatus.approved,
        )

    def _event(self, story, span) -> EventDocument:
        return EventDocument(
            id="e1",
            title="琉璃回岛",
            summary="琉璃时隔两年回到小岛，最先由妃迎接",
            participants=["琉璃", "妃"],
            causes=["分居"],
            outcomes=["兄妹重逢"],
            evidence_text="进岛后最先迎接我的，是我的亲人。",
            story=story,
            source=span,
            reality_status=RealityStatus.objective,
            review_status=ReviewStatus.approved,
        )

    def _chapter_summary(self, story, span) -> ChapterSummaryDocument:
        return ChapterSummaryDocument(
            id="c1",
            title="第一章概要",
            summary="琉璃回到小岛，与妹妹妃重逢，故事围绕排挤事件展开。",
            story=story,
            source=span,
            review_status=ReviewStatus.approved,
        )

    @pytest.mark.parametrize(
        ("builder", "expected_type"),
        [
            ("_scene", DocumentType.scene),
            ("_fact", DocumentType.fact),
            ("_relation", DocumentType.relation),
            ("_event", DocumentType.event),
            ("_chapter_summary", DocumentType.chapter_summary),
        ],
    )
    def test_roundtrip_and_defaults(self, request, story, span, builder, expected_type):
        """每种文档 model_dump → model_validate 往返，document_type 默认正确。"""
        doc = getattr(self, builder)(story, span)
        assert doc.document_type is expected_type
        dumped = doc.model_dump()
        restored = type(doc).model_validate(dumped)
        assert restored == doc
        assert restored.model_dump() == dumped

    @pytest.mark.parametrize(
        "builder",
        ["_scene", "_fact", "_relation", "_event", "_chapter_summary"],
    )
    def test_json_roundtrip(self, request, story, span, builder):
        """JSONL 真实口径：model_dump_json → model_validate_json 往返相等。"""
        doc = getattr(self, builder)(story, span)
        json_text = doc.model_dump_json()
        restored = type(doc).model_validate_json(json_text)
        assert restored == doc
        assert restored.model_dump_json() == json_text

    @pytest.mark.parametrize(
        ("builder", "wrong_type"),
        [
            ("_scene", DocumentType.fact),
            ("_fact", DocumentType.scene),
            ("_relation", DocumentType.event),
            ("_event", DocumentType.relation),
            ("_chapter_summary", DocumentType.scene),
        ],
    )
    def test_document_type_mismatch_rejected(self, request, story, span, builder, wrong_type):
        """document_type 与具体模型不匹配时必须失败。"""
        doc = getattr(self, builder)(story, span)
        with pytest.raises(ValidationError):
            type(doc).model_validate({**doc.model_dump(), "document_type": wrong_type.value})

    def test_scene_extra_field_rejected(self, story, span):
        with pytest.raises(ValidationError):
            SceneDocument(
                id="s1",
                title="场景",
                text="正文",
                story=story,
                source=span,
                embedding=[0.1, 0.2],
            )
