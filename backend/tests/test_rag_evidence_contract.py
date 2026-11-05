"""Approved literal evidence has its own field and must match its declared source."""

import json

import pytest

from knowledge.game_rag.models import (
    EventDocument,
    FactDocument,
    RelationDocument,
    SceneDocument,
    SourceSpan,
    StoryContext,
)
from knowledge.multiscale_rag.index_builder import CharacterKnowledgeIndexBuilder
from knowledge.retrieval_core.loaders import EVENTS_FILE, FACTS_FILE, RELATIONS_FILE, UnapprovedDataError

QUOTE = "甲与乙是朋友，两人约定只喝无糖茶，禁止加糖。"
FILES = {"fact": FACTS_FILE, "relation": RELATIONS_FILE, "event": EVENTS_FILE}


def complete_case(tmp_path, *, kind="fact", summary="两人的饮用约定", evidence=QUOTE):
    story = StoryContext(
        story_unit_id="source-contract", story_title="来源契约", content_scope="main_story", temporal_scope="current"
    )
    body = "前置边界。\n" + evidence + "\n后置边界。"
    (tmp_path / "source.txt").write_text(body, encoding="utf-8")
    source = SourceSpan(source_path="source.txt", line_start=1, line_end=len(body.splitlines()))
    common = dict(
        id=kind + "-card",
        title="饮用约定",
        summary=summary,
        evidence_text=evidence,
        source=source,
        story=story,
        reality_status="objective",
        review_status="approved",
    )
    if kind == "fact":
        card = FactDocument(**common, subject="甲", predicate="饮用约定", value="无糖茶")
    elif kind == "relation":
        card = RelationDocument(**common, subject="甲", relation="朋友", target="乙")
    else:
        card = EventDocument(**common, participants=["甲", "乙"], causes=["共同约定"], outcomes=["禁止加糖"])
    approved = tmp_path / "approved"
    approved.mkdir()
    for filename in FILES.values():
        (approved / filename).write_text("", encoding="utf-8")
    path = approved / FILES[kind]
    path.write_text(card.model_dump_json() + "\n", encoding="utf-8")
    scene = SceneDocument(
        id="source-scene",
        title="约定场景",
        text=body,
        source=source,
        story=story,
        present_characters=["甲", "乙"],
        reality_status="objective",
        review_status="approved",
    )
    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(scene.model_dump_json() + "\n", encoding="utf-8")
    builder = CharacterKnowledgeIndexBuilder(
        domain_id="unit", index_version="unit", aliases={"甲": "甲", "乙": "乙"}, corpus_root=tmp_path
    )
    return builder, approved, scenes, path


@pytest.mark.parametrize("kind", ["fact", "relation", "event"])
def test_summary_evidence_heading_cannot_replace_literal_source(kind, tmp_path):
    builder, approved, scenes, _ = complete_case(
        tmp_path, kind=kind, summary="两人约定的展示摘要。\n证据：这里仍是摘要。"
    )
    result = builder.build(approved, scenes)
    (evidence,) = [d for d in result.documents if d.document_type == "evidence"]
    assert evidence.content == QUOTE
    assert (evidence.source.line_start, evidence.source.line_end) == (2, 2)
    assert result.exact_evidence_matches == 1


def test_quote_outside_declared_source_range_is_an_error(tmp_path):
    builder, approved, scenes, path = complete_case(tmp_path)
    card = json.loads(path.read_text())
    card["source"]["line_end"] = 1
    path.write_text(json.dumps(card, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence|Evidence"):
        builder.build(approved, scenes)


def test_missing_approved_evidence_is_reported_instead_of_skipped(tmp_path):
    builder, approved, scenes, path = complete_case(tmp_path)
    card = json.loads(path.read_text())
    del card["evidence_text"]
    path.write_text(json.dumps(card, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(UnapprovedDataError, match="evidence_text"):
        builder.build(approved, scenes)


def test_source_indentation_is_not_trimmed_by_display_parsing(tmp_path):
    quote = "\t甲与乙是朋友。\n    两人约定只喝无糖茶，禁止加糖。"
    builder, approved, scenes, _ = complete_case(tmp_path, evidence=quote)
    result = builder.build(approved, scenes)
    (evidence,) = [d for d in result.documents if d.document_type == "evidence"]
    assert evidence.content == quote
    assert (evidence.source.line_start, evidence.source.line_end) == (2, 3)


def test_evidence_heading_inside_real_source_remains_literal(tmp_path):
    quote = "甲与乙约定只喝无糖茶。\n证据：若未确认，两人不会开始。"
    builder, approved, scenes, _ = complete_case(tmp_path, evidence=quote)
    result = builder.build(approved, scenes)
    (evidence,) = [d for d in result.documents if d.document_type == "evidence"]
    assert evidence.content == quote
    assert result.exact_evidence_matches == 1
