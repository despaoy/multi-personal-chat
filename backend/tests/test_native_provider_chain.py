"""Native providers must receive compiled messages and explicit output budgets."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from db.schemas import MessageRequest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,available", [("openai_compat", True), ("openai_compat", False), ("ollama", True), ("vllm", False), ("vllm", True)]
)
async def test_selected_provider_receives_messages_and_budget_without_local_override(monkeypatch, provider, available):
    from api import generate
    from inference import model_manager

    calls = []

    class Manager:
        _current_provider = SimpleNamespace(value=provider)

        def set_lora_adapter(self, value):
            pass

        def get_status(self):
            return {"modelName": "selected-native", "currentProvider": provider, "providers": {provider: {"modelName": "selected-native"}}}

        def get_current_provider(self):
            return self

        def snapshot(self):
            return SimpleNamespace(base_url="https://native.invalid/v1", model="selected-native",
                                   async_complete=self.async_complete,
                                   get_status=lambda: {"modelName": "selected-native"})

        async def async_complete(self, messages, *, temperature, max_tokens, top_p,
                                 frequency_penalty, repetition_penalty, enable_thinking, lora_name):
            assert provider == "openai_compat"
            calls.append(dict(messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                              frequency_penalty=frequency_penalty, repetition_penalty=repetition_penalty,
                              enable_thinking=enable_thinking, lora_name=lora_name))
            return "原生提供方回答", 0.01

        # Match the actual manager signature; **kwargs previously hid a real
        # routing failure in which max_tokens never reached any provider.
        async def async_generate(self, prompt, session_history=None, max_tokens_override=None):
            assert provider != "openai_compat", "Cloud must not re-enter legacy prompt generation"
            calls.append(dict(prompt=prompt, history=session_history, budget=max_tokens_override))
            return "原生提供方回答", 0.01

    manager = Manager()
    manager._providers = {model_manager.ModelProvider(provider): manager}
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    local_call = AsyncMock(return_value="原生提供方回答",
        side_effect=None if provider == "vllm" else AssertionError("Cloud must not call local inference"))
    available_check = AsyncMock(return_value=SimpleNamespace(generate=local_call) if available else None)
    monkeypatch.setattr(generate, "get_vllm_client", available_check)

    async def compiled_retrieval(request, lora_name, *, model_generate, **kwargs):
        reply = await model_generate(
            messages=[
                {"role": "system", "content": "编译后角色及表达边界"},
                {"role": "user", "content": "当前住处丽水；完整已知记忆和当前问题"},
            ],
            temperature=0.17, max_tokens=733, top_p=1.0, frequency_penalty=0.0,
            repetition_penalty=1.0, enable_thinking=False, lora_name=None,
        )
        return reply, False, {}

    monkeypatch.setattr(generate, "_generate_with_retrieval", compiled_retrieval)
    if provider == "vllm" and not available:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as caught:
            await generate._generate_reply_impl(
                MessageRequest(message="我的当前住处是哪？"), persist_message=False,
                record_invocation=False, enable_rag=False,
            )
        assert caught.value.status_code == 503
        assert not calls
        local_call.assert_not_awaited()
        return

    response = await generate._generate_reply_impl(
        MessageRequest(message="我的当前住处是哪？"), persist_message=False, record_invocation=False, enable_rag=False
    )
    assert response.reply == "原生提供方回答"
    if provider == "openai_compat":
        assert calls == [dict(messages=[
            {"role": "system", "content": "编译后角色及表达边界"},
            {"role": "user", "content": "当前住处丽水；完整已知记忆和当前问题"},
        ], temperature=0.17, max_tokens=733, top_p=1.0, frequency_penalty=0.0,
            repetition_penalty=1.0, enable_thinking=False, lora_name=None)]
    elif provider == "vllm":
        assert not calls
        local_call.assert_awaited_once()
        assert local_call.await_args.kwargs["max_tokens"] == 733
        assert local_call.await_args.kwargs["messages"][-1]["content"] == "当前住处丽水；完整已知记忆和当前问题"
    else:
        assert calls == [dict(prompt="当前住处丽水；完整已知记忆和当前问题",
                             history=[{"role": "system", "content": "编译后角色及表达边界"}], budget=733)]
    if provider != "vllm":
        local_call.assert_not_awaited()
        available_check.assert_not_awaited()
