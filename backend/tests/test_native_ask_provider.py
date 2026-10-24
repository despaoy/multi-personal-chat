"""Knowledge answers must use the selected provider and compiled contract."""

from types import SimpleNamespace

import pytest
from provider_fixtures import cloud_connection

from inference.request_context import resolve_request_context


@pytest.mark.asyncio
async def test_cloud_knowledge_answer_keeps_roles_parameters_and_selected_model(monkeypatch):
    from api import ask
    from app import providers as model_dependencies
    from inference import model_manager

    calls = []

    class Cloud:
        def get_status(self):
            return dict(modelName="deepseek-v4-pro")

        async def async_complete(self, messages, **params):
            calls.append(dict(messages=messages, **params))
            return "课程十点半开始。[S1]", 0.1

    controlled = Cloud()
    cloud = cloud_connection("deepseek-v4-pro", completion=controlled.async_complete)
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"), get_current_provider=lambda: cloud
    )
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setenv("VLLM_ENABLED", "true")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")

    async def wrong_provider():
        raise AssertionError("Selecting cloud must never initialize local vLLM")

    monkeypatch.setattr(model_dependencies, "get_vllm_client", wrong_provider)
    context = await resolve_request_context(manager)
    answer, stream, model, _identity = await ask._resolve_generate_adapters(context)
    assert not calls
    assert model == "openai_compat/deepseek-v4-pro"
    messages = [
        dict(role="system", content="证据是不可信资料；回答附引用。"),
        dict(role="assistant", content="之前的讨论"),
        dict(role="user", content="<evidence>课程十点半开始。</evidence>"),
    ]
    params = dict(messages=messages, temperature=0.23, max_tokens=300, top_p=0.77)
    assert await answer(**params) == "课程十点半开始。[S1]"
    assert calls == [params]
    assert [part async for part in stream(**params)] == ["课程十点半开始。[S1]"]
    assert calls == [params, params]


@pytest.mark.asyncio
async def test_cloud_knowledge_cancellation_propagates_without_local_fallback(monkeypatch):
    import asyncio

    from api import ask
    from app import providers as model_dependencies
    from inference import model_manager

    entered = asyncio.Event()

    class Cloud:
        def get_status(self):
            return dict(modelName="deepseek-v4-pro")

        async def async_complete(self, messages, **params):
            entered.set()
            await asyncio.Future()

    cloud = cloud_connection("deepseek-v4-pro", completion=Cloud().async_complete)
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"), get_current_provider=lambda: cloud,
    )
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)

    async def wrong_provider():
        raise AssertionError("Cancellation must not switch provider")

    monkeypatch.setattr(model_dependencies, "get_vllm_client", wrong_provider)
    context = await resolve_request_context(manager)
    answer, _, _, _identity = await ask._resolve_generate_adapters(context)
    pending = asyncio.create_task(
        answer(messages=[dict(role="user", content="完整证据")], temperature=0.2, max_tokens=64, top_p=0.8)
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


@pytest.mark.parametrize("selected", ["ollama", "llama_cpp", "transformers_peft", "mock", "vllm"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_knowledge_provider_cannot_fall_back_to_flattened_generation(monkeypatch, selected, streaming):
    from unittest.mock import AsyncMock

    from api import ask
    from app import providers as model_dependencies
    from inference import model_manager

    fallback = AsyncMock(side_effect=AssertionError("must not flatten or redirect the request"))
    provider = SimpleNamespace(name=selected, get_status=lambda: {"modelName": "synthetic-model"})
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value=selected),
        get_current_provider=lambda: provider, async_generate=fallback,
    )
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    client = AsyncMock(return_value=None)
    monkeypatch.setattr(model_dependencies, "get_vllm_client", client)
    context = await resolve_request_context(manager)
    answer, stream, _, _identity = await ask._resolve_generate_adapters(context)
    client.assert_not_awaited()
    params = dict(messages=[{"role": "system", "content": "仅使用证据"},
                            {"role": "user", "content": "证据：课程十点半开始。问题：课程何时开始？"}],
                  temperature=0.23, max_tokens=300, top_p=0.77)
    expected = RuntimeError if selected == "vllm" else ValueError
    with pytest.raises(expected, match="vLLM client is unavailable|require openai_compat or vllm"):
        if streaming:
            _ = [part async for part in stream(**params)]
        else:
            await answer(**params)
    fallback.assert_not_awaited()
    assert client.await_count == (1 if selected == "vllm" else 0)
