"""CAHM 上下文感知记忆提取的本地硬校验。"""

import json

import pytest

from character.memory_llm import MemoryResponseError, parse_llm_proposals


def _response(*, attributed_to="user", **memory):
    return json.dumps({"memories": [dict(memory, attributed_to=attributed_to)]}, ensure_ascii=False)


def test_supersede_must_use_existing_memory_key_and_overwrites_that_key():
    old = (
        {
            "id": "1",
            "memory_key": "goal_点云补全",
            "memory_type": "shared_event",
            "content": "用户正在进行或准备：点云补全",
        },
    )
    raw = _response(
        kind="goal",
        value="点云识别",
        content="用户正在进行或准备：点云识别",
        evidence="补全方向暂时不考虑了，我最近改做点云识别",
        confidence=0.97,
        operation="SUPERSEDE",
        target_memory_id="1",
        target_memory_key="goal_点云补全",
    )

    proposals = parse_llm_proposals(
        raw,
        source_message="补全方向暂时不考虑了，我最近改做点云识别。",
        existing_memories=old,
    )

    assert len(proposals) == 1
    assert proposals[0].operation == "SUPERSEDE"
    assert proposals[0].memory.memory_key == "goal_点云补全"
    assert proposals[0].memory.content == "用户正在进行或准备：点云识别"


def test_supersede_rejects_invented_target_memory_key():
    raw = _response(
        kind="goal",
        value="点云识别",
        content="用户正在进行或准备：点云识别",
        evidence="我最近改做点云识别",
        confidence=0.99,
        operation="SUPERSEDE",
        target_memory_id="1",
        target_memory_key="goal_不存在",
    )
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(
            raw,
            source_message="我最近改做点云识别。",
            existing_memories=({"id": "1", "memory_key": "goal_点云补全", "content": "用户正在做点云补全"},),
        )
    assert caught.value.field == "target_memory_key"


def test_supersede_rejects_real_but_incompatible_target_key():
    raw = _response(
        kind="name",
        value="小明",
        content="用户说自己叫小明",
        evidence="我叫小明",
        confidence=0.99,
        operation="SUPERSEDE",
        target_memory_id="1",
        target_memory_key="goal_点云补全",
    )
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(
            raw,
            source_message="我叫小明。",
            existing_memories=({"id": "1", "memory_key": "goal_点云补全", "content": "用户正在做点云补全"},),
        )
    assert caught.value.field == "kind"


def test_ellipsis_value_must_be_grounded_in_supplied_context():
    raw = _response(
        kind="goal",
        value="点云补全",
        content="用户正在进行或准备：点云补全",
        evidence="还是上次那个方向",
        confidence=0.95,
        operation="ADD",
        target_memory_key="",
    )
    grounded = parse_llm_proposals(
        raw,
        source_message="还是上次那个方向。",
        history=({"role": "user", "content": "我最近在做点云补全"},),
    )
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(raw, source_message="还是上次那个方向。")
    assert len(grounded) == 1
    assert caught.value.field == "value"


def test_third_party_fact_is_not_saved():
    third_party = _response(
        kind="like",
        value="咖啡",
        content="用户喜欢咖啡",
        evidence="我的朋友喜欢咖啡",
        confidence=0.99,
        operation="ADD",
        target_memory_key="",
    )
    assert parse_llm_proposals(third_party, source_message="我的朋友喜欢咖啡。") == []


def test_add_rejects_invented_target_and_keeps_literal_value_without_target():
    raw = _response(
        kind="goal",
        value="准备保研面试",
        content="用户正在准备保研面试",
        evidence="我正在准备保研面试。",
        confidence=1.0,
        operation="ADD",
        target_memory_key="goal_准备保研面试",
    )
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(raw, source_message="我正在准备保研面试。")
    assert caught.value.field == "target_memory_id"
    payload = json.loads(raw)
    payload["memories"][0]["target_memory_key"] = ""
    proposals = parse_llm_proposals(json.dumps(payload, ensure_ascii=False), source_message="我正在准备保研面试。")
    assert len(proposals) == 1
    assert proposals[0].memory.memory_key == "goal_准备保研面试"
    assert proposals[0].target_memory_key == ""


def test_negative_preference_uses_explicit_model_kind_and_literal_value():
    raw = _response(
        kind="dislike",
        value="太苦的",
        content="用户不喜欢太苦的咖啡",
        evidence="不是完全讨厌咖啡，只是不喜欢太苦的。",
        confidence=0.95,
        operation="SUPERSEDE",
        target_memory_id="1",
        target_memory_key="preference_咖啡",
    )
    proposals = parse_llm_proposals(
        raw,
        source_message="不是完全讨厌咖啡，只是不喜欢太苦的。",
        existing_memories=({"id": "1", "memory_key": "preference_咖啡", "content": "用户说不喜欢咖啡"},),
    )
    assert len(proposals) == 1
    assert proposals[0].memory.content == "用户不喜欢太苦的咖啡"
