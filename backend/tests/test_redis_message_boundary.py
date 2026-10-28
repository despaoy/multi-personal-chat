"""Explicit Redis/NoneBot controls: failures must not become local deduplication.

Actual client publication and bot module run; no live Redis or message send.
"""

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError

from cache import redis_client
from db.database import SQLiteDB
from db.schemas import MessageRequest
from services import chat_answer
from test_compiled_cloud_transport import installed as installed

actual_answer = chat_answer.generate_chat_answer


@pytest.fixture
def redis_state(monkeypatch):
    monkeypatch.setattr(redis_client, "_async_client", None)
    monkeypatch.setattr(redis_client, "_async_lock", None)
    monkeypatch.setattr(redis_client, "_get_async_pool", Mock(return_value=object()))
    monkeypatch.setattr(redis_client, "REDIS_URL", "redis://fixture-user:synthetic-private-secret@fixture.invalid/0")


def client():
    return SimpleNamespace(ping=AsyncMock(return_value=True), set=AsyncMock(return_value=True))


async def test_failed_ping_is_not_a_missing_client(redis_state, monkeypatch):
    candidate = client()
    failure = ConnectionError("Controlled Redis connection unavailable")
    candidate.ping.side_effect = failure
    factory = Mock(return_value=candidate)
    monkeypatch.setattr(aioredis, "Redis", factory)
    with pytest.raises(ConnectionError) as caught:
        await redis_client.get_async_redis()
    assert caught.value is failure and redis_client._async_client is None
    factory.assert_called_once()


async def test_concurrent_callers_wait_for_the_same_validated_client(redis_state, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def ping():
        entered.set()
        await release.wait()
        return True

    candidate = client()
    candidate.ping.side_effect = ping
    factory = Mock(return_value=candidate)
    monkeypatch.setattr(aioredis, "Redis", factory)
    first = asyncio.create_task(redis_client.get_async_redis())
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(redis_client.get_async_redis())
    try:
        await asyncio.sleep(0)
        assert not second.done()
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), timeout=1)
    assert all(result is candidate for result in results)
    candidate.ping.assert_awaited_once()
    factory.assert_called_once()


async def test_successful_connection_log_excludes_credentials(redis_state, monkeypatch, caplog):
    candidate = client()
    monkeypatch.setattr(aioredis, "Redis", Mock(return_value=candidate))
    caplog.set_level(logging.INFO, logger="cache.redis_client")
    assert await redis_client.get_async_redis() is candidate
    assert "synthetic-private-secret" not in caplog.text and "fixture-user" not in caplog.text


@pytest.fixture
def bot_module(monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("SUPERUSERS", "[]")
    path = Path(__file__).resolve().parents[1] / "bot" / "bot.py"
    spec = importlib.util.spec_from_file_location("bot._message_boundary_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_duplicate_check_failure_preserves_cause_instead_of_using_memory(bot_module, monkeypatch):
    failure = ConnectionError("Controlled deduplication service unavailable")
    getter = AsyncMock(side_effect=failure)
    monkeypatch.setattr(redis_client, "get_async_redis", getter)
    with pytest.raises(RuntimeError) as caught:
        await bot_module._is_duplicate_message("fiction-message-1")
    assert caught.value.__cause__ is failure
    getter.assert_awaited_once()


@pytest.mark.parametrize("result,duplicate", [(True, False), (None, True), (False, True)])
async def test_atomic_deduplication_keeps_normal_results(bot_module, monkeypatch, result, duplicate):
    shared = client()
    shared.set.return_value = result
    monkeypatch.setattr(redis_client, "get_async_redis", AsyncMock(return_value=shared))
    assert await bot_module._is_duplicate_message("fiction-message-1") is duplicate
    shared.set.assert_awaited_once_with("dedup:msg:fiction-message-1", "1", nx=True, ex=3600)


async def test_set_failure_stops_instead_of_becoming_a_new_message(bot_module, monkeypatch):
    shared = client()
    failure = ConnectionError("Controlled SET unavailable")
    shared.set.side_effect = failure
    monkeypatch.setattr(redis_client, "get_async_redis", AsyncMock(return_value=shared))
    with pytest.raises(bot_module.MessageDeduplicationError) as caught:
        await bot_module._is_duplicate_message("fiction-message-1")
    assert caught.value.__cause__ is failure
    shared.set.assert_awaited_once()


async def test_unknown_set_status_is_not_a_successful_reservation(bot_module, monkeypatch):
    shared = client()
    shared.set.return_value = "synthetic-private-invalid-status"
    monkeypatch.setattr(redis_client, "get_async_redis", AsyncMock(return_value=shared))
    with pytest.raises(bot_module.MessageDeduplicationError) as caught:
        await bot_module._is_duplicate_message("fiction-message-1")
    assert "synthetic-private" not in str(caught.value)


async def test_failed_ping_can_be_retried_by_a_later_call(redis_state, monkeypatch):
    failed, ready = client(), client()
    failed.ping.side_effect = ConnectionError("Controlled initial outage")
    factory = Mock(side_effect=[failed, ready])
    monkeypatch.setattr(aioredis, "Redis", factory)
    with pytest.raises(ConnectionError):
        await redis_client.get_async_redis()
    assert factory.call_count == 1
    assert await redis_client.get_async_redis() is ready
    assert await redis_client.get_async_redis() is ready
    assert factory.call_count == 2
    ready.ping.assert_awaited_once()


async def test_negative_ping_does_not_publish_a_client(redis_state, monkeypatch):
    candidate = client()
    candidate.ping.return_value = False
    monkeypatch.setattr(aioredis, "Redis", Mock(return_value=candidate))
    with pytest.raises(RuntimeError, match="connection check"):
        await redis_client.get_async_redis()
    assert redis_client._async_client is None


async def test_client_initialization_cancellation_is_preserved(redis_state, monkeypatch):
    candidate = client()
    candidate.ping.side_effect = asyncio.CancelledError
    monkeypatch.setattr(aioredis, "Redis", Mock(return_value=candidate))
    with pytest.raises(asyncio.CancelledError):
        await redis_client.get_async_redis()
    assert redis_client._async_client is None


@pytest.fixture
def private_event():
    from nonebot.adapters.onebot.v11 import Message, PrivateMessageEvent

    text = "完整合成记录：尚未付款，仅周五执行，出差例外。"
    return PrivateMessageEvent.model_validate(dict(
        time=1792108800, self_id=9001, post_type="message", message_type="private", sub_type="friend",
        message_id=7, user_id=1001, message=Message(text), raw_message=text, font=0,
        sender=dict(user_id=1001, nickname="fiction-owner"),
    ))


@pytest.fixture
def message_handler(bot_module, private_event, monkeypatch):
    import nonebot
    from nonebot.exception import FinishedException

    matchers = []

    def matcher(*args, **kwargs):
        item = SimpleNamespace(name=args[0] if args else None, handlers=[], finish=AsyncMock(side_effect=FinishedException))

        def handle():
            def register(function):
                item.handlers.append(function)
                return function
            return register

        item.handle = handle
        matchers.append(item)
        return item

    monkeypatch.setattr(nonebot, "init", Mock())
    monkeypatch.setattr(nonebot, "get_driver", Mock(return_value=SimpleNamespace(register_adapter=Mock())))
    monkeypatch.setattr(nonebot, "run", Mock())
    monkeypatch.setattr(bot_module, "on_message", matcher)
    monkeypatch.setattr(bot_module, "on_command", matcher)
    monkeypatch.setattr(bot_module, "should_reply", AsyncMock(return_value=True))
    original_process = bot_module.process_message
    monkeypatch.setattr(bot_module, "process_message", AsyncMock(return_value="不应生成"))
    bot_module.init_bot()
    matchers[0].commands = {item.name: item for item in matchers if item.name}
    matchers[0].process_message = original_process
    return matchers[0], private_event, SimpleNamespace(send=AsyncMock())


async def test_selected_message_reports_deduplication_failure_before_generation(bot_module, message_handler, monkeypatch):
    from nonebot.exception import FinishedException

    matcher, event, bot = message_handler
    logs = []
    sink = bot_module.logger.add(lambda message: logs.append(str(message)))
    monkeypatch.setattr(redis_client, "get_async_redis", AsyncMock(
        side_effect=ConnectionError("synthetic-private-service-detail")))
    try:
        with pytest.raises(FinishedException):
            await matcher.handlers[0](bot, event)
    finally:
        bot_module.logger.remove(sink)
    matcher.finish.assert_awaited_once()
    assert "去重" in matcher.finish.await_args.args[0] and "未生成" in matcher.finish.await_args.args[0]
    assert "synthetic-private" not in ''.join(logs)
    bot_module.process_message.assert_not_awaited()
    bot_module.should_reply.assert_awaited_once_with(event)
    bot.send.assert_not_awaited()


async def test_unselected_message_needs_no_redis_check(bot_module, message_handler, monkeypatch):
    matcher, event, bot = message_handler
    bot_module.should_reply.return_value = False
    getter = AsyncMock(side_effect=ConnectionError("Unselected messages must not require Redis"))
    monkeypatch.setattr(redis_client, "get_async_redis", getter)
    await matcher.handlers[0](bot, event)
    getter.assert_not_awaited()
    bot_module.process_message.assert_not_awaited()
    matcher.finish.assert_not_awaited()


async def test_duplicate_message_is_skipped_without_error_reply(bot_module, message_handler, monkeypatch):
    matcher, event, bot = message_handler
    shared = client()
    shared.set.return_value = None
    monkeypatch.setattr(redis_client, "get_async_redis", AsyncMock(return_value=shared))
    await matcher.handlers[0](bot, event)
    bot_module.process_message.assert_not_awaited()
    matcher.finish.assert_not_awaited()
    bot.send.assert_not_awaited()


@pytest.mark.parametrize("failure_type", [ConnectionError, TimeoutError])
async def test_atomic_reservation_failure_is_not_automatically_replayed(monkeypatch, failure_type):

    monkeypatch.setattr(redis_client, "_async_pool", None)
    monkeypatch.setattr(redis_client, "REDIS_URL", "redis://fixture.invalid/0")
    pool = redis_client._get_async_pool()
    connection = pool.make_connection()
    failure = failure_type("Controlled reservation acknowledgement unavailable")
    operation = AsyncMock(side_effect=failure)
    try:
        with pytest.raises(failure_type) as caught:
            await connection.retry.call_with_retry(operation, AsyncMock())
        assert caught.value is failure
        operation.assert_awaited_once()
    finally:
        await pool.aclose()


async def test_session_setting_read_failure_does_not_enable_replies(bot_module, private_event, monkeypatch):
    failure = RuntimeError("synthetic-private-database-error")
    reader = Mock(side_effect=failure)
    monkeypatch.setattr(bot_module, "db", SimpleNamespace(is_session_bot_enabled=reader))
    with pytest.raises(RuntimeError) as caught:
        await bot_module.should_reply(private_event)
    assert caught.value.__cause__ is failure
    reader.assert_called_once_with("1001", "qq", "1001", "private")


@pytest.fixture
def generation_control(bot_module, private_event, monkeypatch):
    async def submit(factory, **kwargs):
        return await factory()

    runtime = SimpleNamespace(check_rate_limits=AsyncMock(), submit=AsyncMock(side_effect=submit),
                              priority_for=Mock(return_value=10))
    resets = {bot_module._message_scope(private_event).memory_scope_key: 17}
    generate = AsyncMock(return_value=SimpleNamespace(reply="尚未付款；周五执行，出差时顺延。"))
    monkeypatch.setattr(bot_module, "inference_runtime", runtime)
    monkeypatch.setattr(bot_module, "_history_resets", resets)
    monkeypatch.setattr(bot_module, "generate_reply_core", generate)
    monkeypatch.setattr(bot_module, "config", SimpleNamespace(REPLY_DELAY=-100))
    return runtime, dict(resets), generate


@pytest.mark.parametrize("stage", ["rate", "queue", "model"])
async def test_generation_failures_propagate_without_a_reply(bot_module, private_event, generation_control, stage):
    runtime, history, generate = generation_control
    if stage == "rate":
        failure = bot_module.RateLimitExceeded("sender", 2)
        runtime.check_rate_limits.side_effect = failure
    elif stage == "queue":
        failure = bot_module.InferenceQueueFull("synthetic-private-queue-error")
        runtime.submit.side_effect = failure
    else:
        failure = RuntimeError("synthetic-private-model-error")
        generate.side_effect = failure
    with pytest.raises(type(failure)) as caught:
        await bot_module.process_message(private_event)
    assert caught.value is failure
    assert bot_module._history_resets == history
    if stage != "model":
        generate.assert_not_awaited()


@pytest.mark.parametrize("enabled", [True, False])
async def test_private_reply_selection_preserves_explicit_setting(bot_module, private_event, monkeypatch, enabled):
    reader = Mock(return_value=enabled)
    monkeypatch.setattr(bot_module, "db", SimpleNamespace(is_session_bot_enabled=reader))
    assert await bot_module.should_reply(private_event) is enabled


async def test_generation_success_keeps_history_and_request_identity(bot_module, private_event, generation_control):
    runtime, history, generate = generation_control
    assert await bot_module.process_message(private_event) == generate.return_value.reply
    runtime.check_rate_limits.assert_awaited_once_with("qq:nonebot", "private:1001", "1001")
    runtime.submit.assert_awaited_once()
    request = generate.await_args.args[0]
    assert request.message == str(private_event.message)
    assert request.history == [] and request._history_loaded is False
    assert request._history_after_id == 17
    assert (request.platform, request.adapter, request.conversationId, request.senderId) == (
        "qq", "nonebot", "1001", "1001")
    assert bot_module._history_resets == history


async def test_generation_cancellation_is_preserved(bot_module, private_event, generation_control):
    _, history, generate = generation_control
    generate.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await bot_module.process_message(private_event)
    assert bot_module._history_resets == history


@pytest.mark.parametrize("stage,expected", [
    ("selection", "会话回复设置读取失败"), ("rate", "请求过于频繁"),
    ("queue", "推理队列已满"), ("timeout", "消息处理超时"), ("model", "消息处理失败"),
])
async def test_message_entry_reports_failures_once_without_private_details(
    bot_module, message_handler, monkeypatch, stage, expected,
):
    from nonebot.exception import FinishedException

    matcher, event, bot = message_handler
    detail = "synthetic-private-upstream-content"
    failures = {
        "selection": bot_module.SessionReplyCheckError(detail),
        "rate": bot_module.RateLimitExceeded("sender", 2),
        "queue": bot_module.InferenceQueueFull(detail),
        "timeout": asyncio.TimeoutError(detail),
        "model": RuntimeError(detail),
    }
    target = bot_module.should_reply if stage == "selection" else bot_module.process_message
    target.side_effect = failures[stage]
    dedup = AsyncMock(return_value=False)
    monkeypatch.setattr(bot_module, "_is_duplicate_message", dedup)
    logs = []
    sink = bot_module.logger.add(lambda message: logs.append(str(message)))
    try:
        with pytest.raises(FinishedException):
            await matcher.handlers[0](bot, event)
    finally:
        bot_module.logger.remove(sink)
    matcher.finish.assert_awaited_once()
    assert expected in matcher.finish.await_args.args[0]
    assert detail not in str(matcher.finish.await_args) + ''.join(logs)
    bot.send.assert_not_awaited()
    if stage == "selection":
        dedup.assert_not_awaited()
        bot_module.process_message.assert_not_awaited()


async def test_message_entry_sends_the_real_success_reply(bot_module, message_handler, monkeypatch):
    matcher, event, bot = message_handler
    monkeypatch.setattr(bot_module, "_is_duplicate_message", AsyncMock(return_value=False))
    bot_module.process_message.return_value = "尚未付款。"
    await matcher.handlers[0](bot, event)
    matcher.finish.assert_awaited_once()
    assert str(matcher.finish.await_args.args[0]) == "尚未付款。"
    bot_module.process_message.assert_awaited_once_with(event)
    bot.send.assert_not_awaited()


async def test_history_read_failure_stops_before_model(installed, monkeypatch, tmp_path):
    from api.generate import generate_reply_core

    database = SQLiteDB(tmp_path / "history-failure.sqlite")
    database.update_config(installed.config)
    failure = RuntimeError("synthetic-private-history-error")
    monkeypatch.setattr(database, "list_conversation_history", Mock(side_effect=failure))
    request = MessageRequest(message="合成完整问题：请复述此前执行时间和例外。", senderId="1001",
                             conversationId="1001", loraName="default")
    request._history_after_id = 0
    try:
        with pytest.raises(RuntimeError) as caught:
            await generate_reply_core(request, message_db=database)
        assert caught.value is failure
        assert installed.calls == []
        assert database.get_messages() == []
    finally:
        database.close_connection()


async def test_history_empty_result_and_recovery_remain_distinct(installed, monkeypatch, tmp_path):
    from api.generate import generate_reply_core

    database = SQLiteDB(tmp_path / "history-recovery.sqlite")
    database.update_config(installed.config)
    failure = RuntimeError("Controlled history service unavailable")
    rows = [{"role": "user", "content": "周五执行，出差例外。"},
            {"role": "assistant", "content": "已记录例外。"}]
    reader = Mock(side_effect=[failure, [], rows])
    monkeypatch.setattr(database, "list_conversation_history", reader)
    monkeypatch.setattr(chat_answer, "generate_chat_answer", actual_answer)

    async def generate():
        request = MessageRequest(message="合成完整问题：请复述执行时间和例外。", senderId="1001",
                                 conversationId="1001", loraName="default")
        request._history_after_id = 0
        return await generate_reply_core(request, message_db=database)

    try:
        with pytest.raises(RuntimeError) as caught:
            await generate()
        assert caught.value is failure and installed.calls == []
        assert (await generate()).reply
        assert len(installed.calls) == 1
        assert all(rows[0]["content"] not in row["content"] for row in installed.calls[-1]["body"]["messages"])
        assert (await generate()).reply
        assert len(installed.calls) == 2
        wire = installed.calls[-1]["body"]["messages"]
        assert all(any(row["content"] in item["content"] for item in wire) for row in rows)
        assert reader.call_count == 3 and len(database.get_messages()) == 2
    finally:
        database.close_connection()
