"""Grounded generic facts must not depend on summary word order."""

import json

import pytest

from character.memory_llm import MemoryResponseError, parse_llm_proposals


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
def test_generic_paraphrase_requires_valid_content_or_explicit_empty_display(source, value, content):
    with pytest.raises(MemoryResponseError) as caught:
        _parse(source, value, content)
    assert caught.value.field == "content"
    result = _parse(source, value, "")
    assert len(result) == 1
    assert result[0].memory.content == f"用户明确提到：{source}"
    assert result[0].evidence == source


def test_unverified_summary_additions_are_a_content_error():
    source = "我的书架放在客厅。"
    with pytest.raises(MemoryResponseError) as caught:
        _parse(source, "书架放在客厅", "用户在客厅放置昂贵的红木书架")
    assert caught.value.field == "content"


@pytest.mark.parametrize("changes,error", [
    ({"evidence": "我的自行车是红色的。"}, False),
    ({"attributed_to": "assistant"}, True),
    ({"confidence": 0.1}, False),
])
def test_empty_display_does_not_bypass_grounding_or_admission(changes, error):
    if error:
        with pytest.raises(MemoryResponseError) as caught:
            _parse("我的自行车是蓝色的。", "自行车是蓝色", "", **changes)
        assert caught.value.field == "attributed_to"
    else:
        assert not _parse("我的自行车是蓝色的。", "自行车是蓝色", "", **changes)


def test_unverified_generic_value_is_an_error_not_an_observation():
    source = "我的自行车是蓝色的。"
    with pytest.raises(MemoryResponseError) as caught:
        _parse(source, "自行车是红色", "用户有一辆红色自行车")
    assert caught.value.field == "value"
