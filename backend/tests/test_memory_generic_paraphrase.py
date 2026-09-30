"""Grounded generic facts must not depend on summary word order."""

import json

import pytest

from character.memory_llm import parse_llm_proposals


def _parse(source, initial_value, content, **changes):
    proposal = dict(kind="other_user_fact", value=initial_value, content=content,
                    evidence=source, confidence=0.98, operation="ADD",
                    attributed_to="user")
    proposal.update(changes)
    return parse_llm_proposals(json.dumps({"memories": [proposal]}, ensure_ascii=False),
                               source_message=source)


@pytest.mark.parametrize("source,value,content", [
    ("请记住，我的狗叫松糕。", "狗叫松糕", "用户养了一只叫松糕的狗"),
    ("我的自行车是蓝色的。", "自行车是蓝色", "用户有一辆蓝色自行车"),
    ("我的书架放在客厅。", "书架放在客厅", "用户在客厅放置书架"),
])
def test_generic_paraphrase_uses_grounded_canonical_view(source, value, content):
    result = _parse(source, value, content)
    assert len(result) == 1
    assert result[0].memory.content == f"用户明确提到：{source}"
    assert result[0].evidence == source


def test_fallback_does_not_preserve_unverified_summary_additions():
    source = "我的书架放在客厅。"
    result = _parse(source, "书架放在客厅", "用户在客厅放置昂贵的红木书架")
    assert len(result) == 1
    assert "红木" not in result[0].memory.content
    assert result[0].evidence == source


@pytest.mark.parametrize("changes", [
    {"evidence": "我的自行车是红色的。"},
    {"attributed_to": "assistant"},
    {"confidence": 0.1},
])
def test_fallback_does_not_bypass_grounding_or_admission(changes):
    assert not _parse("我的自行车是蓝色的。", "自行车是蓝色",
                      "用户有一辆蓝色自行车", **changes)


def test_unverified_generic_value_is_an_observation_not_a_fabricated_fact():
    source = "我的自行车是蓝色的。"
    proposal, = _parse(source, "自行车是红色", "用户有一辆红色自行车")
    assert proposal.source_observation
    assert proposal.evidence == source
    assert "蓝色" in proposal.memory.content and "红色" not in proposal.memory.content
    assert "红色" not in proposal.memory.memory_key
