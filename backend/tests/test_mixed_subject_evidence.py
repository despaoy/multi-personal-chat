"""Unrelated actors inside complete evidence cannot veto the user's own fact."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


@pytest.mark.parametrize(
    "kind,value,source,first,last,qualifiers",
    [
        (
            "like",
            "紫茶",
            "请记住：周末聚会时，我喜欢紫茶；朋友柏宁喜欢红茶；不过我喜欢的紫茶只限无糖版本。这些只适用于当前会话。",
            "q1",
            "q4",
            {"context": {"source_unit": "q1"}, "condition": {"source_unit": "q4"}},
        ),
        (
            "residence",
            "临沂",
            "请记住：工作期间，我住在临沂；我的朋友住在青州；我在临沂的居住安排只适用于工作期间。这些只适用于当前会话。",
            "q1",
            "q4",
            {"context": {"source_unit": "q1"}, "condition": {"source_unit": "q4"}},
        ),
    ],
)
def test_complete_self_fact_retains_intervening_friend_and_later_condition(
    kind, value, source, first, last, qualifiers
):
    raw = dict(
        kind=kind,
        value=value,
        content="",
        evidence={"start_unit": first, "end_unit": last},
        confidence=0.99,
        operation="ADD",
        attributed_to="user",
        scope_level="conversation",
        qualifiers=qualifiers,
    )
    result = parse_llm_proposals(json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source)
    assert len(result) == 1
    proposal = result[0]
    assert proposal.fact_kind == kind and value in proposal.memory.content
    assert "朋友" in proposal.evidence
    assert dict(proposal.qualifiers)["condition"] in proposal.evidence
    assert proposal.scope_level == "conversation"


@pytest.mark.parametrize("source,value", [("我的朋友喜欢红茶。", "红茶"), ("我妹妹住在青州。", "青州")])
def test_complete_third_party_fact_does_not_become_user_owned(source, value):
    raw = dict(
        kind="like" if "喜欢" in source else "residence",
        value=value,
        content="",
        evidence=source,
        confidence=0.99,
        operation="ADD",
        attributed_to="user",
        scope_level="conversation",
    )
    assert parse_llm_proposals(json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source) == []


@pytest.mark.parametrize(
    "source,expected",
    [
        ("我喜欢无糖版本的紫茶；朋友喜欢红茶。", 1),
        ("朋友喜欢无糖版本的紫茶。", 0),
    ],
)
def test_explicit_fragments_keep_the_same_actor_boundary(source, expected):
    raw = dict(
        kind="like",
        value=["无糖版本", "紫茶"],
        content="",
        evidence=source,
        confidence=0.99,
        operation="ADD",
        attributed_to="user",
        scope_level="conversation",
        qualifiers={},
    )
    result = parse_llm_proposals(json.dumps({"memories": [raw]}, ensure_ascii=False), source_message=source)
    assert len(result) == expected
    if result:
        assert result[0].value_fragments == ("无糖版本", "紫茶")
        assert result[0].evidence == source
