"""Actual NoneBot history stays within the canonical user/conversation scope."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message
from nonebot.exception import FinishedException
from services import chat_answer
from test_compiled_cloud_transport import installed as installed
from test_redis_message_boundary import (
    bot_module as bot_module,
)
from test_redis_message_boundary import (
    generation_control as generation_control,
)
from test_redis_message_boundary import (
    message_handler as message_handler,
)
from test_redis_message_boundary import (
    private_event as private_event,
)

from db.database import SQLiteDB

actual_answer = chat_answer.generate_chat_answer


def group_event(private, sender_id, text):
    return GroupMessageEvent.model_validate(
        dict(
            time=private.time,
            self_id=private.self_id,
            post_type="message",
            message_type="group",
            sub_type="normal",
            message_id=sender_id,
            group_id=private.user_id,
            user_id=sender_id,
            message=Message(text),
            raw_message=text,
            font=0,
            anonymous=None,
            sender=dict(user_id=sender_id, nickname=f"synthetic-member-{sender_id}", role="member"),
        )
    )


@pytest.mark.parametrize("cold", [False, True])
async def test_model_history_is_scoped_for_colliding_ids_and_group_members(
    bot_module,
    private_event,
    installed,
    monkeypatch,
    tmp_path,
    cold,
):
    facts = [
        "合成私聊甲专属：尚未付款，仅周五执行，出差顺延。",
        "合成群成员乙专属：尚未签收，仅周六执行，下雨顺延。",
        "合成群成员丙专属：尚未预约，仅周日执行，加班顺延。",
    ]
    events = [
        private_event.model_copy(update={"message": Message(facts[0]), "raw_message": facts[0]}),
        group_event(private_event, 2002, facts[1]),
        group_event(private_event, 3003, facts[2]),
    ]
    database = SQLiteDB(tmp_path / "scoped-history.sqlite")
    database.update_config(installed.config)

    async def submit(factory, **kwargs):
        return await factory()

    runtime = SimpleNamespace(
        check_rate_limits=AsyncMock(), submit=AsyncMock(side_effect=submit), priority_for=lambda *args: 1
    )
    history = bot_module.SessionHistory(max_tokens=4096)
    monkeypatch.setattr(bot_module, "db", database)
    monkeypatch.setattr(bot_module, "session_history", history)
    monkeypatch.setattr(bot_module, "inference_runtime", runtime)
    monkeypatch.setattr(bot_module, "config", SimpleNamespace(REPLY_DELAY=-100))
    monkeypatch.setattr(chat_answer, "generate_chat_answer", actual_answer)
    try:
        for event in events:
            await bot_module.process_message(event)
        if cold:
            history.sessions.clear()
        for index, event in enumerate(events):
            query = f"合成用户{index}：请复述我此前的完整执行安排，包括否定、日期和例外。"
            await bot_module.process_message(event.model_copy(update={"message": Message(query), "raw_message": query}))
            wire = installed.calls[-1]["body"]["messages"]
            assert any(facts[index] in item["content"] for item in wire)
            assert all(other not in item["content"] for other in facts if other != facts[index] for item in wire)
            assert any(query in item["content"] for item in wire)
        assert len(installed.calls) == 6
        assert len(database.get_messages()) == 6
        keys = [call.kwargs["session_id"] for call in runtime.submit.await_args_list]
        assert len(set(keys[:3])) == 3 and keys[:3] == keys[3:]
    finally:
        database.close_connection()


async def test_clear_resets_only_the_callers_scope_without_reloading_or_deleting_archive(
    bot_module,
    message_handler,
    generation_control,
    monkeypatch,
):
    matcher, private, bot = message_handler
    runtime, _, generate = generation_control
    private_scope = bot_module._message_scope(private)
    group_scope = bot_module._message_scope(group_event(private, 2002, "/clear"))
    other_scope = bot_module._message_scope(group_event(private, 3003, "/clear"))
    rows = [
        {"role": "user", "content": "合成档案：周五执行，出差顺延。"},
        {"role": "assistant", "content": "已记录完整条件。"},
    ]
    from unittest.mock import Mock

    reader = Mock(return_value=rows)
    monkeypatch.setattr(bot_module, "db", SimpleNamespace(list_conversation_history=reader))
    history = bot_module.SessionHistory()
    monkeypatch.setattr(bot_module, "session_history", history)
    assert history.get_history(private_scope) == rows
    assert history.get_history(group_scope) == rows
    assert history.get_history(other_scope) == rows
    clear = matcher.commands["clear"]
    with pytest.raises(FinishedException):
        await clear.handlers[0](bot, group_event(private, 2002, "/clear"))
    assert history.get_history(group_scope) == []
    assert history.get_history(group_scope) == []
    assert history.get_history(private_scope) == rows
    assert history.get_history(other_scope) == rows
    assert reader.call_count == 3
    runtime.submit.assert_awaited_once()
    assert json.loads(runtime.submit.await_args.kwargs["session_id"]) == list(group_scope.memory_scope_key)
    clear.finish.assert_awaited_once()
    generate.assert_not_awaited()
    history.add_message(group_scope, "user", "清除后的新安排：周二执行，尚未付款。")
    history.add_message(group_scope, "assistant", "已记录周二及未付款状态。")
    assert history.get_history(group_scope) == [
        {"role": "user", "content": "清除后的新安排：周二执行，尚未付款。"},
        {"role": "assistant", "content": "已记录周二及未付款状态。"},
    ]


@pytest.mark.parametrize("mode", ["cleared", "not_loaded", "provided"])
async def test_loaded_empty_history_survives_mapping_and_character_history_selection(mode):
    from services.character_context import CharacterContextService

    from api.generate import _character_turn_input
    from character.models import build_user_scope
    from db.schemas import MessageRequest

    stored = [
        {"role": "user", "content": "合成旧存档：周五执行，出差顺延。"},
        {"role": "assistant", "content": "合成旧回复。"},
    ]
    provided = [{"role": "user", "content": "合成现场：周二执行，尚未付款。"}]
    request = MessageRequest(
        message="复述当前安排",
        platform="qq",
        adapter="nonebot",
        senderId="2002",
        conversationId="1001",
        conversationType="group",
        history=provided if mode == "provided" else [],
    )
    request._history_loaded = mode != "not_loaded"
    turn = _character_turn_input(request)
    reader = AsyncMock(return_value=stored)
    service = SimpleNamespace(
        _message_repo=SimpleNamespace(list_recent_conversation_history=reader),
        _history_limit=8,
        _history_max_chars=6000,
    )
    scope = build_user_scope("qq", "nonebot", "2002", "1001", "group")
    history = await CharacterContextService._load_history(service, turn, scope, "synthetic-role")
    assert history == {"cleared": [], "not_loaded": stored, "provided": provided}[mode]
    if mode == "not_loaded":
        reader.assert_awaited_once_with(scope, limit=8, max_chars=6000, character_id="synthetic-role")
    else:
        reader.assert_not_awaited()


async def test_clear_queue_failure_keeps_history_and_does_not_report_success(
    bot_module,
    message_handler,
    generation_control,
    monkeypatch,
):
    from unittest.mock import Mock

    matcher, event, bot = message_handler
    runtime, _, _ = generation_control
    history = SimpleNamespace(clear_history=Mock())
    monkeypatch.setattr(bot_module, "session_history", history)
    failure = bot_module.InferenceQueueFull("controlled clear queue unavailable")
    runtime.submit.side_effect = failure
    clear = matcher.commands["clear"]
    with pytest.raises(bot_module.InferenceQueueFull) as caught:
        await clear.handlers[0](bot, event)
    assert caught.value is failure
    history.clear_history.assert_not_called()
    clear.finish.assert_not_awaited()
