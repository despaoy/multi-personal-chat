"""Malformed cache responses fail rather than acquiring invented metadata."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from api import generate
from db.schemas import MessageRequest
from tests.test_native_generation_concurrency import installed as installed

REPLY = "  合成前提：\n尚未付款；仅周五执行，出差例外。\n"


def request():
    return MessageRequest(message="合成前提：尚未付款；仅周五执行，出差例外。请按原格式复述。",
                          platform="web", adapter="response-contract", senderId="fiction-owner",
                          conversationId="fiction-owner", conversationType="private")


def cached_response():
    return dict(reply=REPLY, model="fixture-cloud (openai_compat)", costTime=.01, modelInvoked=True)


@pytest.mark.parametrize("field,value", [("model", None), ("model", ""), ("model", "  \n"),
                                         ("reply", ""), ("reply", "  \n")])
async def test_invalid_cache_response_stops_without_model_or_save(installed, monkeypatch, field, value):
    state, _manager, _database, _semaphore = installed
    cached = cached_response()
    if value is None:
        del cached[field]
    else:
        cached[field] = value
    cache = SimpleNamespace(get=AsyncMock(return_value=cached), set=AsyncMock())
    save = AsyncMock()
    monkeypatch.setattr(generate, "response_cache", cache)
    monkeypatch.setattr(generate, "_save_message", save)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(request(), persist_message=True, record_invocation=False)
    assert caught.value.status_code == 503
    assert "缓存" in str(caught.value.detail)
    assert caught.value.detail["stage"] == "response_cache"
    assert caught.value.detail["reason"] == "invalid_cached_response"
    assert caught.value.detail["fields"] == [field]
    error_text = str(caught.value.detail) + str(caught.value.__cause__)
    assert REPLY not in error_text and "尚未付款" not in error_text and "出差例外" not in error_text
    assert state.calls == []
    cache.get.assert_awaited_once()
    cache.set.assert_not_awaited()
    save.assert_not_awaited()


async def test_valid_cache_response_preserves_exact_text_and_model(installed, monkeypatch):
    state, _manager, _database, _semaphore = installed
    cached = cached_response()
    cache = SimpleNamespace(get=AsyncMock(return_value=cached), set=AsyncMock())
    monkeypatch.setattr(generate, "response_cache", cache)
    result = await generate._generate_reply_impl(request(), persist_message=False, record_invocation=False)
    assert result.reply == REPLY and result.model == cached["model"]
    assert result.modelInvoked is False and "cached_model_response" in result.warnings
    assert state.calls == []
    cache.get.assert_awaited_once()
    cache.set.assert_not_awaited()


async def test_real_cache_miss_still_generates_once(installed, monkeypatch):
    state, _manager, _database, _semaphore = installed
    cache = SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock())
    monkeypatch.setattr(generate, "response_cache", cache)
    current = request()
    result = await generate._generate_reply_impl(current, persist_message=False, record_invocation=False)
    assert result.reply == current.message and result.model == "fixture-cloud (openai_compat)"
    assert result.modelInvoked is True and len(state.calls) == 1
    cache.get.assert_awaited_once()
    cache.set.assert_awaited_once()


async def test_wrong_cache_root_shape_is_a_data_failure(installed, monkeypatch):
    state, _manager, _database, _semaphore = installed
    cache = SimpleNamespace(get=AsyncMock(return_value=[cached_response()]), set=AsyncMock())
    save = AsyncMock()
    monkeypatch.setattr(generate, "response_cache", cache)
    monkeypatch.setattr(generate, "_save_message", save)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(request(), persist_message=True, record_invocation=False)
    assert caught.value.status_code == 503
    assert caught.value.detail["reason"] == "invalid_cached_response"
    assert caught.value.detail["fields"] == ["$"]
    assert "尚未付款" not in str(caught.value.detail) + str(caught.value.__cause__)
    assert state.calls == []
    cache.get.assert_awaited_once()
    cache.set.assert_not_awaited()
    save.assert_not_awaited()
