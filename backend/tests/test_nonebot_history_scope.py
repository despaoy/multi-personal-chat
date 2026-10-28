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


async def test_model_history_is_scoped_for_colliding_ids_and_group_members(
    bot_module,
    private_event,
    installed,
    monkeypatch,
    tmp_path,
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
    monkeypatch.setattr(bot_module, "db", database)
    monkeypatch.setattr(bot_module, "inference_runtime", runtime)
    monkeypatch.setattr(bot_module, "config", SimpleNamespace(REPLY_DELAY=-100))
    monkeypatch.setattr(chat_answer, "generate_chat_answer", actual_answer)
    try:
        for event in events:
            await bot_module.process_message(event)
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
    installed,
    monkeypatch,
    tmp_path,
):
    matcher, private, bot = message_handler
    facts = [
        "合成私聊原话：周五执行，尚未付款。",
        "合成乙原话：周六执行，下雨顺延。",
        "合成丙原话：周日执行，加班顺延。",
    ]
    events = [
        private.model_copy(update={"message": Message(facts[0]), "raw_message": facts[0]}),
        group_event(private, 2002, facts[1]),
        group_event(private, 3003, facts[2]),
    ]
    database = SQLiteDB(tmp_path / "reset.sqlite")
    database.update_config(installed.config)

    async def submit(factory, **kwargs):
        return await factory()

    runtime = SimpleNamespace(
        check_rate_limits=AsyncMock(), submit=AsyncMock(side_effect=submit), priority_for=lambda *args: 1
    )
    monkeypatch.setattr(bot_module, "db", database)
    monkeypatch.setattr(bot_module, "inference_runtime", runtime)
    monkeypatch.setattr(bot_module, "config", SimpleNamespace(REPLY_DELAY=-100))
    monkeypatch.setattr(chat_answer, "generate_chat_answer", actual_answer)
    try:
        for event in events:
            await matcher.process_message(event)
        assert len(database.get_messages()) == 3
        clear = matcher.commands["clear"]
        with pytest.raises(FinishedException):
            await clear.handlers[0](bot, group_event(private, 2002, "/clear"))
        scope = bot_module._message_scope(events[1])
        assert bot_module._history_resets == {scope.memory_scope_key: 3}
        assert len(database.get_messages()) == 3
        clear_key = runtime.submit.await_args.kwargs["session_id"]
        assert json.loads(clear_key) == list(scope.memory_scope_key)
        clear.finish.assert_awaited_once()
        for index in [1, 0, 2]:
            text = f"合成用户{index}：请复述当前执行安排。"
            await matcher.process_message(
                events[index].model_copy(update={"message": Message(text), "raw_message": text})
            )
            wire = installed.calls[-1]["body"]["messages"]
            assert all(facts[1] not in row["content"] for row in wire)
            if index != 1:
                assert any(facts[index] in row["content"] for row in wire)
        new_fact = "清除后的新安排：周二执行，尚未付款。"
        await matcher.process_message(
            events[1].model_copy(update={"message": Message(new_fact), "raw_message": new_fact})
        )
        query = "请复述清除后的新安排，包括否定状态。"
        await matcher.process_message(events[1].model_copy(update={"message": Message(query), "raw_message": query}))
        wire = installed.calls[-1]["body"]["messages"]
        assert any(new_fact in row["content"] for row in wire)
        assert all(facts[1] not in row["content"] for row in wire)
        assert len(installed.calls) == len(database.get_messages()) == 8
        assert all(any(row["message"] == fact for row in database.get_messages()) for fact in facts)
        assert runtime.submit.await_count == 9
    finally:
        database.close_connection()


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


@pytest.mark.parametrize("stage", ["queue", "boundary"])
async def test_clear_queue_failure_keeps_history_and_does_not_report_success(
    bot_module,
    message_handler,
    generation_control,
    monkeypatch,
    stage,
):
    from unittest.mock import Mock

    matcher, event, bot = message_handler
    runtime, resets, _ = generation_control
    failure = (
        bot_module.InferenceQueueFull("controlled clear queue unavailable")
        if stage == "queue"
        else RuntimeError("controlled archive boundary unavailable")
    )
    reader = Mock(return_value=17, side_effect=failure if stage == "boundary" else None)
    monkeypatch.setattr(bot_module, "db", SimpleNamespace(get_message_high_watermark=reader))
    if stage == "queue":
        runtime.submit.side_effect = failure
    clear = matcher.commands["clear"]
    with pytest.raises(type(failure)) as caught:
        await clear.handlers[0](bot, event)
    assert caught.value is failure
    assert bot_module._history_resets == resets
    if stage == "queue":
        reader.assert_not_called()
    else:
        reader.assert_called_once()
    clear.finish.assert_not_awaited()


async def test_warm_nonebot_history_cannot_reinject_a_revoked_source(
    bot_module, private_event, installed, monkeypatch, tmp_path
):
    from datetime import datetime, timezone

    database = SQLiteDB(tmp_path / "history-source.sqlite")
    database.update_config(installed.config)
    scope = bot_module._message_scope(private_event)
    source_scope = dict(
        character_id="synthetic-role",
        platform="qq",
        adapter="nonebot",
        sender_id=scope.sender_id,
        conversation_type="private",
        conversation_id=scope.conversation_id,
    )
    deleted = "合成待撤销原话：我的代号是青岚，尚未付款，仅周五执行，出差顺延。"
    kept = "合成保留原话：公开档案馆周二09:00至17:00开放。"
    for source, body in [("deleted", deleted), ("kept", kept)]:
        database.add_message(
            dict(
                platform="qq",
                adapter="nonebot",
                senderId=scope.sender_id,
                characterId="synthetic-role",
                conversationType="private",
                conversationId=scope.conversation_id,
                sessionType="private",
                sessionId=scope.conversation_id,
                sourceMessageId=source,
                message=body,
                reply="已收到完整资料。",
                createdAt="2026-10-28T12:00:00",
            )
        )
        database.capture_memory_source(
            **source_scope, source_message_id=source, body=body, observed_at=datetime.now(timezone.utc)
        )
    claim = database.append_character_memory_claim(
        **source_scope,
        memory_type="user_fact",
        memory_key="codename",
        content="合成用户代号是青岚",
        source_message_id="deleted",
    )

    async def submit(factory, **kwargs):
        return await factory()

    monkeypatch.setattr(bot_module, "db", database)
    monkeypatch.setattr(
        bot_module,
        "inference_runtime",
        SimpleNamespace(check_rate_limits=AsyncMock(), submit=submit, priority_for=lambda *args: 1),
    )
    monkeypatch.setattr(bot_module, "config", SimpleNamespace(REPLY_DELAY=-100))
    monkeypatch.setattr(chat_answer, "generate_chat_answer", actual_answer)
    try:
        query = "合成检验：请复述已提供的执行安排，保留否定、时间和例外。"
        event = private_event.model_copy(update={"message": Message(query), "raw_message": query})
        await bot_module.process_message(event)
        database.erase_character_memories(**source_scope, memory_id=claim["id"])
        authoritative = database.list_conversation_history(
            "qq", "nonebot", scope.sender_id, "private", scope.conversation_id
        )
        assert all(deleted not in item["content"] for item in authoritative)
        assert any(kept in item["content"] for item in authoritative)
        await bot_module.process_message(event)
        leaked = any(deleted in row["content"] for row in installed.calls[-1]["body"]["messages"])
        assert not leaked and len(installed.calls) == 2
        assert any(kept in row["content"] for row in installed.calls[-1]["body"]["messages"])
        assert len(database.get_messages()) == 4
    finally:
        database.close_connection()


def test_reset_boundary_uses_archive_sequence_without_removing_rows(tmp_path):
    database = SQLiteDB(tmp_path / "sequence.sqlite")
    try:
        assert database.get_message_high_watermark() == 0
        scope = dict(
            platform="qq",
            adapter="nonebot",
            senderId="1001",
            conversationType="private",
            conversationId="1001",
            sessionType="private",
            sessionId="1001",
        )
        database.add_message(
            dict(scope, message="合成旧安排：周五执行。", reply="已记录旧安排。", createdAt="2026-10-28T12:00:00")
        )
        boundary = database.get_message_high_watermark()
        database.add_message(
            dict(
                scope,
                message="合成新安排：周二执行，尚未付款。",
                reply="已记录新安排。",
                createdAt="2026-10-27T12:00:00",
            )
        )
        assert database.get_message_high_watermark() > boundary
        assert database.list_conversation_history("qq", "nonebot", "1001", "private", "1001", after_id=boundary) == [
            {"role": "user", "content": "合成新安排：周二执行，尚未付款。"},
            {"role": "assistant", "content": "已记录新安排。"},
        ]
        assert len(database.get_messages()) == 2
    finally:
        database.close_connection()


async def test_stored_history_uses_the_resolved_character_before_model_transport(installed, monkeypatch, tmp_path):
    from api import generate
    from db.schemas import MessageRequest

    database = SQLiteDB(tmp_path / "character-history.sqlite")
    database.update_config(installed.config)
    role = "tsukiyashiro_kisaki"
    own = "合成本角色专属原话：周五执行，出差顺延。"
    other = "合成另一角色专属原话：周日执行，下雨顺延。"
    try:
        for character, body in [(role, own), ("synthetic-other", other)]:
            database.add_message(
                dict(
                    platform="qq",
                    adapter="nonebot",
                    senderId="1001",
                    characterId=character,
                    conversationType="private",
                    conversationId="1001",
                    sessionType="private",
                    sessionId="1001",
                    message=body,
                    reply="已收到该角色资料。",
                    createdAt="2026-10-28T12:00:00",
                )
            )
        prepare = AsyncMock(return_value=None)
        monkeypatch.setattr(generate, "_prepare_character_turn", prepare)
        monkeypatch.setattr(chat_answer, "generate_chat_answer", actual_answer)
        request = MessageRequest(
            message="合成检验：请复述当前角色此前的执行安排。",
            characterId=role,
            platform="qq",
            adapter="nonebot",
            senderId="1001",
            conversationId="1001",
            conversationType="private",
        )
        request._history_after_id = 0
        await generate._generate_reply_impl(
            request, message_db=database, persist_message=False, record_invocation=False, character_service=object()
        )
        assert request._history_loaded is True
        prepare.assert_awaited_once()
        assert prepare.await_args.args[1] == role
        wire = installed.calls[0]["body"]["messages"]
        assert any(own in row["content"] for row in wire)
        assert all(other not in row["content"] for row in wire)
        assert len(database.get_messages()) == 2
    finally:
        database.close_connection()
