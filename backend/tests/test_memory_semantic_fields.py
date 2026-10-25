"""Semantic field identity is explicit; attribution and target boundaries still apply."""

import json

import pytest

from character.memory_llm import MemoryResponseError, parse_llm_proposals


def parse(kind, value, source, **extra):
    item = dict(kind=kind, value=value, evidence=source, confidence=.98, operation="ADD", attributed_to="user")
    item.update(extra)
    return parse_llm_proposals(json.dumps({"memories": [item]}, ensure_ascii=False), source_message=source)


@pytest.mark.parametrize("kind,value,source,key", [
    ("origin", "宣城", "我的籍贯是宣城", "user_origin"),
    ("residence", "丽水", "我目前住在丽水", "user_residence"),
])
def test_explicit_kind_keeps_its_own_field(kind, value, source, key):
    proposal, = parse(kind, value, source)
    assert proposal.memory.memory_key == key
    assert proposal.memory.memory_type == "user_fact"
    assert proposal.evidence == source


@pytest.mark.parametrize("kind", ["origin", "residence"])
@pytest.mark.parametrize("source", ["朋友的籍贯是宣城", "假如我住在宣城，我会每天散步。"])
def test_location_kinds_keep_existing_subject_and_hypothetical_boundaries(kind, source):
    assert parse(kind, "宣城", source) == []


@pytest.mark.parametrize("kind,key", [("origin", "user_residence"), ("residence", "user_origin")])
def test_origin_and_residence_cannot_overwrite_each_other(kind, key):
    source = "我的籍贯是宣城" if kind == "origin" else "我目前住在宣城"
    item = dict(kind=kind, value="宣城", evidence=source, confidence=.98, operation="SUPERSEDE",
                attributed_to="user", target_memory_id="1", target_memory_key=key)
    old = dict(id="1", memory_type="user_fact", memory_key=key, content="用户说自己居住在宣城",
               importance=.6, confidence=.98, status="active")
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(json.dumps({"memories": [item]}), source_message=source, existing_memories=(old,))
    assert caught.value.field == "kind"
