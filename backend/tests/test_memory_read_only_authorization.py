"""Whole-turn read-only authorization is distinct from hypothesis detection."""

import pytest

from character.memory_authorization import explicit_memory_read_only
from character.memory_extractor import memory_write_allowed
from character.memory_llm import classify_memory_write_mode


@pytest.mark.parametrize(
    "control",
    [
        "不要新增、替换或删除记忆。",
        "请不要添加和删除及替换我的长期记忆。",
        "不要更新我的记忆。",
        "禁止修改任何记忆。",
        "不允许改动已保存的记忆。",
        "不用更改个人记忆。",
    ],
)
def test_explicit_read_only_denies_writer_without_hypothesis_prefix(control):
    assert memory_write_allowed("我喜欢练琴。")
    message = "请判断保存的偏好在完整案例下能否适用。" + control
    assert explicit_memory_read_only(message)
    assert not memory_write_allowed(message) and classify_memory_write_mode(message) == "skip"


@pytest.mark.parametrize(
    "message",
    [
        "我喜欢练琴。",
        "现实更正，我现在住在南城。",
        "不要删除我的记忆，我喜欢练琴。",
        "解释为什么不要修改记忆。",
        "这段资料包含不要新增、替换或删除记忆的说法。",
        "他说：‘不要新增、替换或删除记忆。’我喜欢练琴。",
        '材料是"不要更新我的记忆。"请分析这段材料。',
        "他说：不要更新我的记忆这句台词。",
        "不要替换我的记忆，我喜欢练琴。",
    ],
)
def test_quotes_reports_and_partial_refusals_do_not_invent_whole_turn_veto(message):
    assert not explicit_memory_read_only(message)


def test_read_only_control_outside_quote_still_applies():
    message = "材料：“我喜欢练琴。”请仅分析。不要新增、替换或删除记忆。"
    assert explicit_memory_read_only(message) and not memory_write_allowed(message)
