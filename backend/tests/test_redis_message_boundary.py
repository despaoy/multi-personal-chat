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
def message_handler(bot_module, monkeypatch):
    import nonebot
    from nonebot.adapters.onebot.v11 import Message, PrivateMessageEvent
    from nonebot.exception import FinishedException

    matchers = []

    def matcher(*args, **kwargs):
        item = SimpleNamespace(handlers=[], finish=AsyncMock(side_effect=FinishedException))

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
    monkeypatch.setattr(bot_module, "process_message", AsyncMock(return_value="不应生成"))
    bot_module.init_bot()
    text = "完整合成记录：尚未付款，仅周五执行，出差例外。"
    event = PrivateMessageEvent.model_validate(dict(
        time=1792108800, self_id=9001, post_type="message", message_type="private", sub_type="friend",
        message_id=7, user_id=1001, message=Message(text), raw_message=text, font=0,
        sender=dict(user_id=1001, nickname="fiction-owner"),
    ))
    return matchers[0], event, SimpleNamespace(send=AsyncMock())


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
