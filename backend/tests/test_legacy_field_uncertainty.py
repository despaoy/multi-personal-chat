"""Legacy ambiguous records prevent claims of proven typed-field absence."""

import pytest
from test_memory_query_plan import recall, row
from test_memory_response import context, packets

from character.models import CompiledCharacterContext
from inference.memory_response import read_memory_fields, render_complete_memory_read, render_memory_response


@pytest.mark.asyncio
async def test_current_legacy_location_is_unresolved_not_absent():
    _, _, trace = await recall([row("user_location", "用户的地理信息")], "我来自哪里，目前住哪里？")
    assert trace["field_presence"]["origin"] is None
    assert trace["field_presence"]["residence"] is None
    assert trace["field_presence"]["major"] is False
    compiled = CompiledCharacterContext("", "", "", memory_status="no_match",
                                        memory_field_presence=tuple(trace["field_presence"].items()))
    reads = read_memory_fields("我来自哪里，目前住哪里？", compiled)
    assert {item.status for item in reads} == {"unverified"}
    assert render_complete_memory_read("我来自哪里，目前住哪里？", compiled, ()) is None
    status = render_memory_response("你保存了我的现居地吗？", compiled)
    assert "没有你的" not in status
    assert "不能确认" in status


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["superseded", "retracted", "pending"])
async def test_noncurrent_legacy_location_does_not_block_absence(status):
    _, _, trace = await recall([row("user_location", "用户旧信息", status=status)], "我的住址是什么？")
    assert trace["field_presence"]["residence"] is False


def test_known_packet_cannot_hide_unresolved_storage_dependency():
    from dataclasses import replace

    compiled = replace(context(packets("我住在揭阳。")), memory_field_presence=(("residence", None),))
    assert read_memory_fields("我住哪里？", compiled)[0].status == "unverified"
    assert render_memory_response("我住哪里？", compiled) is None


@pytest.mark.asyncio
async def test_legacy_key_does_not_erase_explicit_typed_storage_state():
    from dataclasses import replace

    _, _, trace = await recall([row("user_location", "用户这周出差"),
                               row("user_residence", "用户说自己居住在揭阳")], "我来自哪里，目前住哪里？")
    assert trace["field_presence"]["residence"] is True
    assert trace["field_presence"]["origin"] is None
    compiled = replace(context(packets("我住在揭阳。")),
                       memory_field_presence=tuple(trace["field_presence"].items()))
    reply = render_complete_memory_read("我来自哪里，目前住哪里？", compiled, ())
    assert "现在住在揭阳" in reply
    assert "关于你的来源地" in reply
    assert "没有你的" not in reply
    assert render_complete_memory_read("我来自哪里，目前住哪里？", compiled,
                                       ({"role": "user", "content": "我搬家了"},)) is None
