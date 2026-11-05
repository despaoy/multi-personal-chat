"""Explicit evidence intervals preserve the whole source between two anchors."""

import json

import pytest

from character.memory_llm import MemoryResponseError, build_memory_llm_messages, parse_llm_proposals

SOURCE = "饮料方面，我接受无糖热茶；我不喝含糖饮料。"


def parse(evidence, *, qualifiers=None):
    raw = dict(
        kind="other_user_fact",
        value="含糖饮料",
        content="",
        evidence=evidence,
        confidence=0.99,
        operation="ADD",
        attributed_to="user",
        scope_level="conversation",
        qualifiers=qualifiers or {},
    )
    return parse_llm_proposals(json.dumps(dict(memories=[raw]), ensure_ascii=False), source_message=SOURCE)[0]


def test_range_keeps_intervening_facts_and_original_context():
    result = parse(dict(start_unit="q0", end_unit="q2"), qualifiers={"context": {"source_unit": "q0"}})
    assert result.evidence == SOURCE
    assert "我接受无糖热茶；" in result.evidence and "我不喝含糖饮料。" in result.evidence
    assert dict(result.qualifiers)["context"] == "饮料方面"


def test_single_unit_range_preserves_the_negation_without_new_sentiment():
    result = parse(dict(start_unit="q2", end_unit="q2"))
    assert result.evidence == "我不喝含糖饮料。"
    assert "不喝" in result.memory.content and "不喜欢" not in result.memory.content


@pytest.mark.parametrize(
    "evidence",
    [
        dict(start_unit="unknown", end_unit="q2"),
        dict(start_unit="q2", end_unit="q0"),
        dict(start_unit=True, end_unit="q2"),
        dict(start_unit="q0", end_unit="q2", source_units=["q0", "q1", "q2"]),
        dict(start_unit="q0"),
    ],
)
def test_unknown_reversed_mixed_or_incomplete_ranges_are_errors(evidence):
    with pytest.raises(MemoryResponseError) as caught:
        parse(evidence)
    assert caught.value.field == "evidence"


def test_existing_contiguous_unit_list_retains_its_contract():
    assert parse(dict(source_units=["q0", "q1", "q2"])).evidence == SOURCE


def test_noncontiguous_legacy_list_is_not_guessed_into_a_range():
    with pytest.raises(MemoryResponseError) as caught:
        parse(dict(source_units=["q0", "q2"]))
    assert caught.value.field == "evidence"


def test_producer_advertises_one_explicit_range_format():
    messages = build_memory_llm_messages(SOURCE, (), (), (), 2000, 0.85)
    payload = json.loads(messages[-1]["content"])
    assert payload["proposal_constraints"]["evidence_formats"] == ["literal_string", "source_unit_range"]
    assert payload["current_user_message"] == SOURCE
    assert "start_unit" in messages[0]["content"] and "end_unit" in messages[0]["content"]


def test_explicit_qualifier_reference_is_part_of_the_same_grounded_span():
    result = parse(dict(start_unit="q2", end_unit="q2"), qualifiers={"context": {"source_unit": "q0"}})
    assert result.evidence == SOURCE
    assert dict(result.qualifiers)["context"] == "饮料方面"


def test_unknown_qualifier_reference_still_fails():
    with pytest.raises(MemoryResponseError) as caught:
        parse(dict(start_unit="q2", end_unit="q2"), qualifiers={"context": {"source_unit": "unknown"}})
    assert caught.value.field == "qualifiers"


def test_literal_qualifier_outside_the_explicit_span_is_not_located_by_guessing():
    with pytest.raises(MemoryResponseError) as caught:
        parse(dict(start_unit="q2", end_unit="q2"), qualifiers={"context": "饮料方面"})
    assert caught.value.field == "qualifiers"


def test_legacy_citation_is_not_silently_expanded_by_new_range_semantics():
    with pytest.raises(MemoryResponseError) as caught:
        parse(dict(source_units=["q2"]), qualifiers={"context": {"source_unit": "q0"}})
    assert caught.value.field == "qualifiers"
