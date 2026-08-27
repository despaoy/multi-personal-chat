"""Precision-first write-gate tests for character long-term memory."""

from character.memory_extractor import (
    MAX_EXTRACTED_MEMORIES,
    extract_memories,
    extract_preferred_address,
)


def _by_key(message: str):
    return {item.memory_key: item for item in extract_memories(message)}


def test_extracts_explicit_name_preference_and_promise():
    memories = _by_key("我叫小明，我喜欢咖啡。下次一定带你去书店")

    assert memories["user_name"].content == "用户说自己叫小明"
    assert memories["preference_咖啡"].content == "用户说喜欢咖啡"
    assert memories["promise_带你去书店"].memory_type == "promise"


def test_extracts_stable_profile_facts_and_current_goal():
    memories = _by_key("我的专业是计算机科学。我是大三学生。我住在南京。我正在准备保研")

    assert memories["user_major"].content == "用户说自己的专业是计算机科学"
    assert memories["user_study_stage"].content == "用户说自己是大三"
    assert memories["user_location"].content == "用户说自己来自或居住在南京"
    assert any(item.memory_type == "shared_event" for item in memories.values())


def test_opt_out_prevents_reverse_memory_write():
    assert extract_memories("不要记住我喜欢咖啡") == []
    assert extract_memories("别保存，我叫小明") == []


def test_sensitive_information_is_never_persisted():
    assert extract_memories("请记住我的支付密码是123456") == []
    assert extract_memories("我的 API key 是 sk-example，我喜欢咖啡") == []


def test_uncertain_clause_is_dropped_without_losing_certain_clause():
    memories = _by_key("我叫小明。我可能正在准备保研")

    assert set(memories) == {"user_name"}


def test_questions_are_not_converted_to_user_facts():
    assert extract_memories("我叫什么名字？") == []
    assert extract_memories("我的专业是什么？") == []
    assert extract_memories("我喜欢什么？") == []
    assert extract_preferred_address("你准备叫我什么？") is None


def test_commands_are_not_mistaken_for_names_or_addresses():
    assert extract_memories("我叫你别走") == []
    assert extract_memories("我叫了一辆车") == []
    assert extract_preferred_address("叫我帮你看看") is None


def test_dislike_uses_same_key_shape_as_preference():
    memories = _by_key("我不喜欢咖啡")

    assert memories["preference_咖啡"].content == "用户说不喜欢咖啡"


def test_write_gate_caps_output_and_keeps_high_importance_items():
    memories = extract_memories(
        "我叫小明，我喜欢咖啡，我喜欢红茶，我喜欢电影。我的专业是计算机科学。下次一定带你去书店"
    )

    assert len(memories) == MAX_EXTRACTED_MEMORIES
    keys = {item.memory_key for item in memories}
    assert "user_name" in keys
    assert "user_major" in keys
    assert "promise_带你去书店" in keys


def test_preferred_address_extracts_only_explicit_non_question_value():
    assert extract_preferred_address("以后叫我小林") == "小林"
    assert extract_preferred_address("我喜欢被叫作小林") == "小林"
    assert extract_preferred_address("你想叫我什么？") is None


def test_address_preference_is_not_saved_as_generic_interest():
    memories = extract_memories("我喜欢被叫作小林")

    assert memories == []
