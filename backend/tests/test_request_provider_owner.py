"""One selected provider resource owns a request across default changes.

Real manager/provider and compiled-message interfaces use isolated HTTP or
explicit synthetic local resources; no model, database write or live network.
"""

from services import chat_answer

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from api import ask, generate
from db.schemas import MessageRequest
from inference import model_manager, token_counting, vllm_client
from inference.generation_request import GenerationRequest, generate_character_response
from inference.request_context import get_request_context, resolve_request_context, use_request_context
from inference.review_client import get_context_review_client

FACTS = "完整虚构题设：周四14点预约三张票，每张两元，共六元；尚未付款，下雨取消。请复述金额、时间及全部条件。"
ANSWER = "周四14点预约三张票，每张两元，共六元；尚未付款，下雨取消。"


class LocalResource(model_manager.BaseProvider):
    def __init__(self, name, calls):
        super().__init__(name)
        self._loaded = True
        self._model_name = "owned-" + name
        self.calls = calls

    def generate(self, *_args, **_kwargs):
        raise AssertionError("This fixture owns an explicit async transport")

    async def async_complete(self, messages, *, temperature, max_tokens, top_p,
                             frequency_penalty, repetition_penalty, enable_thinking, lora_name):
        assert FACTS in messages[-1]["content"] and max_tokens == 128
        assert any(message["role"] == "system" for message in messages)
        assert temperature == .2 and top_p == 1.0
        assert frequency_penalty == 0.0 and repetition_penalty == 1.0
        assert enable_thinking is False and lora_name is None
        self.calls.append(self.name)
        return ANSWER, 0.0


@pytest.fixture
def owned(monkeypatch):
    config = dict(openaiCompatBaseUrl="https://api.deepseek.com", openaiCompatModel="deepseek-flash",
                  openaiCompatApiKey="synthetic-owned-key", useKnowledgeBase=False,
                  temperature=.2, maxTokens=128, topP=1.0)
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: dict(config))
    cloud = model_manager.OpenAICompatProvider()
    snapshots = Mock(wraps=cloud.snapshot)
    monkeypatch.setattr(cloud, "snapshot", snapshots)
    local_calls, http_calls = [], []
    manager = model_manager.ModelManager.__new__(model_manager.ModelManager)
    manager._current_provider = model_manager.ModelProvider.OPENAI_COMPAT
    resources = {model_manager.ModelProvider.OPENAI_COMPAT: cloud}
    for kind in (model_manager.ModelProvider.TRANSFORMERS_PEFT, model_manager.ModelProvider.OLLAMA, model_manager.ModelProvider.VLLM):
        resources[kind] = LocalResource(kind.value, local_calls)
    manager._providers = resources
    reads = Mock(wraps=manager.get_current_provider)
    monkeypatch.setattr(manager, "get_current_provider", reads)
    legacy_calls = Mock(wraps=manager.async_generate)
    monkeypatch.setattr(manager, "async_generate", legacy_calls)
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)

    def respond(request):
        http_calls.append(dict(body=json.loads(request.content), auth=request.headers["Authorization"]))
        return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, "choices": [{"finish_reason": "stop", "message": {"content": ANSWER}}]})

    @asynccontextmanager
    async def acquire(_self, timeout):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client

    monkeypatch.setattr(model_manager.OpenAICompatProvider, "_acquire_http_client", acquire)
    monkeypatch.setattr(generate, "db", SimpleNamespace(config=config, loras=[]))
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_lora_character_id", lambda _: None)
    semaphore = asyncio.Semaphore(2)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: semaphore)
    state = SimpleNamespace(pause_compile=False, compile_entered=asyncio.Event(), compile_release=asyncio.Event())

    async def compiled(request, lora_name, *, model_generate, runtime_config, **_kwargs):
        if state.pause_compile:
            state.pause_compile = False
            state.compile_entered.set()
            await state.compile_release.wait()
        result = await generate_character_response(GenerationRequest(message=request.message,
            persona_prompt="保留完整虚构题设的金额、时间、否定和条件。", apply_prompt_policy=False,
            temperature=runtime_config["temperature"], max_tokens=runtime_config["maxTokens"],
            top_p=runtime_config["topP"], context_window_tokens=8192), model_generate)
        return result.reply, False, {"modelInvoked": True}

    monkeypatch.setattr(chat_answer, 'generate_chat_answer', compiled)
    return SimpleNamespace(manager=manager, resources=resources, reads=reads, snapshots=snapshots,
        local_calls=local_calls, http_calls=http_calls, legacy_calls=legacy_calls, state=state, semaphore=semaphore)


async def turn():
    return await generate._generate_reply_impl(MessageRequest(message=FACTS, loraName="default"),
        persist_message=False, record_invocation=False, enable_rag=False, message_db=generate.db)


@pytest.mark.asyncio
async def test_bound_cloud_review_ignores_unrelated_default_change(owned):
    context = await resolve_request_context(owned.manager)
    owned.manager.set_provider(model_manager.ModelProvider.OLLAMA)
    with use_request_context(context):
        client = await get_context_review_client()
        assert client.provider is context.provider
        assert await client.generate(messages=[{"role": "user", "content": FACTS}], lora_name=None,
            temperature=0.0, max_tokens=64, stream=False, enable_thinking=False) == ANSWER
        assert token_counting.token_counter_info()["mode"] == "deepseek_v41_flash_bpe"
    assert owned.reads.call_count == owned.snapshots.call_count == 1
    assert len(owned.http_calls) == 1 and owned.http_calls[0]["auth"] == "Bearer synthetic-owned-key"
    assert owned.http_calls[0]["body"]["model"] == "deepseek-flash" and get_request_context() is None


@pytest.mark.asyncio
async def test_cloud_generation_keeps_owner_and_new_request_uses_new_default(owned):
    owned.state.pause_compile = True
    task = asyncio.create_task(turn())
    try:
        await asyncio.wait_for(owned.state.compile_entered.wait(), 2)
        owned.manager.set_provider(model_manager.ModelProvider.OLLAMA)
    finally:
        owned.state.compile_release.set()
    first = await task
    assert first.reply == ANSWER and first.model == "deepseek-flash (openai_compat)"
    second = await turn()
    assert second.reply == ANSWER and second.model == "owned-ollama (ollama)"
    assert len(owned.http_calls) == 1 and owned.local_calls == ["ollama"]
    assert owned.reads.call_count == 2 and owned.legacy_calls.call_count == 0
    assert owned.semaphore._value == 2 and get_request_context() is None


@pytest.mark.asyncio
async def test_local_generation_keeps_resource_across_circuit_wait(owned, monkeypatch):
    owned.manager.set_provider(model_manager.ModelProvider.TRANSFORMERS_PEFT)
    entered, release = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def admission():
        entered.set()
        await release.wait()
        yield

    registry = SimpleNamespace(get_or_create=AsyncMock(return_value=admission))
    monkeypatch.setattr(generate, "circuit_breaker_registry", registry)
    task = asyncio.create_task(turn())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        owned.manager.set_provider(model_manager.ModelProvider.OLLAMA)
    finally:
        release.set()
    result = await task
    assert result.reply == ANSWER and result.model == "owned-transformers_peft (transformers_peft)"
    assert owned.local_calls == ["transformers_peft"] and owned.http_calls == []
    assert owned.legacy_calls.call_count == 0 and owned.reads.call_count == 1
    assert owned.semaphore._value == 2 and get_request_context() is None


@pytest.mark.asyncio
async def test_knowledge_adapter_identity_uses_the_bound_local_resource(owned, monkeypatch):
    owned.manager.set_provider(model_manager.ModelProvider.VLLM)
    context = await resolve_request_context(owned.manager)
    gateway = AsyncMock(side_effect=AssertionError("Adapter construction must remain lazy"))
    from app import providers

    monkeypatch.setattr(providers, "get_vllm_client", gateway)
    first = await ask._resolve_generate_adapters(context)
    owned.manager.set_provider(model_manager.ModelProvider.OPENAI_COMPAT)
    second = await ask._resolve_generate_adapters(context)
    assert first[2] == second[2] == "vllm/owned-vllm" and first[3] == second[3]
    assert context.provider is owned.resources[model_manager.ModelProvider.VLLM]
    gateway.assert_not_awaited()


@pytest.mark.asyncio
async def test_bound_vllm_review_keeps_its_gateway_after_default_change(owned, monkeypatch):
    owned.manager.set_provider(model_manager.ModelProvider.VLLM)
    context = await resolve_request_context(owned.manager)
    owned.manager.set_provider(model_manager.ModelProvider.OPENAI_COMPAT)
    client = SimpleNamespace(generate=AsyncMock(return_value=ANSWER))
    gateway = AsyncMock(return_value=client)
    monkeypatch.setattr(vllm_client, "get_vllm_client", gateway)
    with use_request_context(context):
        assert await get_context_review_client() is client
        assert token_counting.token_counter_info()["mode"] == "generic_character_estimate"
    gateway.assert_awaited_once()
    assert owned.http_calls == owned.local_calls == [] and owned.reads.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_bound_cloud_knowledge_generation_survives_default_change(owned, streaming):
    context = await resolve_request_context(owned.manager)
    answer, stream, model, _identity = await ask._resolve_generate_adapters(context)
    owned.manager.set_provider(model_manager.ModelProvider.OLLAMA)
    params = dict(messages=[{"role": "system", "content": "保留金额、否定和条件。"},
                            {"role": "user", "content": FACTS}],
                  temperature=.2, max_tokens=128, top_p=1.0)
    if streaming:
        from knowledge.grounded_answer.service import GroundedAnswerService

        assert stream is None
        adapted = GroundedAnswerService._wrap_as_stream(answer)
        assert [part async for part in adapted(**params)] == [ANSWER]
    else:
        assert await answer(**params) == ANSWER
    assert model == "openai_compat/deepseek-flash" and owned.local_calls == []
    assert len(owned.http_calls) == 1 and owned.http_calls[0]["body"]["messages"] == params["messages"]
    assert owned.reads.call_count == owned.snapshots.call_count == 1
