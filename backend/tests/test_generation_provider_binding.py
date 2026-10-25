"""Provider selection must not change the shared chat request context."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_lora_provider_selection import install_api

from db.schemas import MessageRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai_compat", "ollama"])
async def test_each_provider_receives_same_bound_request_context(monkeypatch, provider):
    state = install_api(monkeypatch, provider)
    database = SimpleNamespace(config={"temperature": 0.3, "maxTokens": 256, "topP": 1.0}, loras=[])
    reply = "3张票，每张2元，共6元。"
    model = AsyncMock(return_value=reply)
    connection = state.manager.get_current_provider()
    connection.async_complete.side_effect = None
    connection.async_complete.return_value = (reply, 0.1)
    monkeypatch.setattr(state.api, "get_vllm_client", AsyncMock(return_value=SimpleNamespace(generate=model)))
    request = MessageRequest(message="虚构票价：每张2元，共买3张。总价是多少？", loraName="default")
    messages = [{"role": "system", "content": "仅按本题完整票价和数量作答。"},
                {"role": "user", "content": request.message}]
    options = dict(temperature=0.3, max_tokens=256, top_p=1.0)

    async def generated(*args, model_generate, **kwargs):
        answer = await model_generate(messages=messages, **options)
        return answer, False, {"modelInvoked": True}

    state.generation.side_effect = generated

    result = await state.api._generate_reply_impl(
        request,
        message_db=database,
        persist_message=False,
        record_invocation=False,
        enable_rag=False,
    )

    assert result.reply == reply and result.modelInvoked is True
    state.generation.assert_awaited_once()
    args, kwargs = state.generation.await_args
    assert args == (request, None if provider == "vllm" else "default")
    assert kwargs["runtime_config"] == database.config
    assert kwargs["message_db"] is database
    assert kwargs["enable_rag"] is False
    assert kwargs["prepared_character_turn"] is None
    assert kwargs["memory_write_enabled"] is None
    assert callable(kwargs["model_generate"])
    if provider == "vllm":
        model.assert_awaited_once_with(messages=messages, **options)
        connection.async_complete.assert_not_awaited()
        assert kwargs["prompt_lora_name"] == "default"
    else:
        assert "prompt_lora_name" not in kwargs
        connection.async_complete.assert_awaited_once_with(messages=messages, **options)
        model.assert_not_awaited()
    state.save.assert_not_awaited()
    state.manager.set_provider.assert_not_called()
