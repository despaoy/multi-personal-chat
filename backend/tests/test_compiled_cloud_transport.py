"""Compiled cloud requests and direct-response cache keep truthful provenance."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from api import generate
from cache.response_cache import ResponseCache
from db.schemas import AskRequest, GenerateResponse, MessageRequest
from inference import model_manager
from inference.generation_request import GenerationRequest, RetrievalResult, generate_character_response

QUERY = "虚构档案馆周二09:00开门、17:00闭馆，暴雨暂停开放。请保留时间和暴雨例外，不能把暴雨改成小雨。"
ANSWER = "单元控制：周二09:00开门、17:00闭馆，暴雨暂停开放。"


class Cache(ResponseCache):
    """Count real cache operations without replacing cache behavior."""

    def __init__(self):
        super().__init__(max_size=2, default_ttl=60)
        self.reads = 0
        self.writes = 0

    async def get(self, prompt_hash, key):
        self.reads += 1
        return await super().get(prompt_hash, key)

    async def set(self, prompt_hash, key, row, **kwargs):
        self.writes += 1
        return await super().set(prompt_hash, key, row, **kwargs)


@pytest.fixture
def installed(monkeypatch):
    stored = {
        "openaiCompatBaseUrl": "https://api.deepseek.com", "openaiCompatApiKey": "fiction-key",
        "openaiCompatModel": "fixture-model", "temperature": 1.6, "maxTokens": 999,
    }
    monkeypatch.setattr(model_manager, "_get_db_config", lambda: dict(stored))
    provider = model_manager.OpenAICompatProvider()
    manager = model_manager.ModelManager.__new__(model_manager.ModelManager)
    manager._current_provider = model_manager.ModelProvider.OPENAI_COMPAT
    manager._providers = {model_manager.ModelProvider.OPENAI_COMPAT: provider}
    manager.async_generate = AsyncMock(side_effect=AssertionError("Compiled cloud messages must not re-enter legacy generation"))
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    runtime_config = {"useKnowledgeBase": False, "temperature": 0.83, "maxTokens": 173}
    monkeypatch.setattr(generate, "db", SimpleNamespace(config=runtime_config, loras=[]))
    monkeypatch.setattr(generate, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "只依据本题完整资料；保留时间、否定和例外。")
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=65536, evidence_max_chars=0))
    calls = []
    state = SimpleNamespace(status=200, finish="stop", failure=None, on_request=None)

    def respond(request):
        if state.failure is not None:
            raise state.failure
        calls.append(dict(body=json.loads(request.content), endpoint=str(request.url), auth=request.headers["Authorization"]))
        if state.on_request is not None:
            state.on_request()
        return httpx.Response(state.status, json={"choices": [{"finish_reason": state.finish, "message": {"content": ANSWER}}]})

    @asynccontextmanager
    async def acquire(_self, timeout):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client

    monkeypatch.setattr(model_manager.OpenAICompatProvider, "_acquire_http_client", acquire)

    async def compiled(request, lora_name, *, model_generate, runtime_config, **kwargs):
        result = await generate_character_response(GenerationRequest(
            message=request.message, persona_prompt="只依据本题完整资料；保留时间、否定和例外。",
            retrieval=RetrievalResult(status="ok", evidence=QUERY) if state.evidence else RetrievalResult(),
            temperature=runtime_config["temperature"], max_tokens=runtime_config["maxTokens"],
            top_p=runtime_config["topP"], context_window_tokens=65536,
        ), model_generate)
        return result.reply, result.plan.retrieval.has_evidence, {}

    state.evidence = False
    monkeypatch.setattr(generate, "_generate_with_retrieval", compiled)
    return SimpleNamespace(provider=provider, manager=manager, stored=stored, config=runtime_config, calls=calls, state=state)


async def receive():
    return await generate._generate_reply_impl(MessageRequest(message=QUERY, loraName="default"),
        persist_message=False, record_invocation=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence", [False, True])
async def test_compiled_temperature_limit_and_complete_messages_reach_wire(installed, evidence):
    installed.state.evidence = evidence
    response = await receive()
    assert response.reply == ANSWER and response.modelInvoked
    assert len(installed.calls) == 1
    body = installed.calls[0]["body"]
    assert body["temperature"] == (0.5 if evidence else 0.83)
    assert body["max_tokens"] == 173 and body["top_p"] == 1.0 and body["model"] == "fixture-model"
    assert body["thinking"] == {"type": "disabled"} and "frequency_penalty" not in body
    assert QUERY in str(body["messages"]) and "不能把暴雨改成小雨" in str(body["messages"])
    installed.manager.async_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_compatible_service_keeps_explicit_supported_sampling(installed):
    installed.stored["openaiCompatBaseUrl"] = "https://fiction.example/v1"
    messages = [{"role": "system", "content": "完整编译边界。"}, {"role": "user", "content": QUERY}]
    reply, _ = await installed.provider.async_complete(messages, temperature=0.12, max_tokens=173,
                                                      top_p=0.77, frequency_penalty=0.2)
    assert reply == ANSWER and len(installed.calls) == 1
    body = installed.calls[0]["body"]
    assert body["messages"] == messages and body["temperature"] == 0.12 and body["top_p"] == 0.77
    assert body["frequency_penalty"] == 0.2
    assert "thinking" not in body and "chat_template_kwargs" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("option,value", [("top_p", 0.7), ("frequency_penalty", 0.2),
    ("repetition_penalty", 1.1), ("enable_thinking", True), ("lora_name", "extra-adapter")])
async def test_unsupported_non_neutral_request_fails_before_http(installed, option, value):
    params = dict(temperature=0.12, max_tokens=173, top_p=1.0)
    params[option] = value
    with pytest.raises(ValueError, match=option):
        await installed.provider.async_complete([{"role": "user", "content": QUERY}], **params)
    assert installed.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "truncation", "transport"])
async def test_completion_failure_never_switches_provider_or_returns_a_reply(installed, failure):
    if failure == "http":
        installed.state.status = 503
    elif failure == "truncation":
        installed.state.finish = "length"
    else:
        installed.state.failure = httpx.ConnectError("fiction-private-error")
    with pytest.raises(HTTPException) as error:
        await receive()
    assert error.value.status_code == 500 and installed.manager._current_provider == model_manager.ModelProvider.OPENAI_COMPAT
    installed.manager.async_generate.assert_not_awaited()
    assert "fiction-private" not in str(error.value.detail)


@pytest.mark.asyncio
async def test_explicit_unsupported_main_sampling_is_not_replaced_with_defaults(installed):
    installed.config["topP"] = 0.7
    with pytest.raises(HTTPException) as error:
        await receive()
    assert error.value.status_code == 500 and isinstance(error.value.__cause__, ValueError)
    assert installed.config["topP"] == 0.7 and installed.calls == []


@pytest.mark.asyncio
async def test_default_knowledge_request_is_supported_non_thinking_sampling(installed):
    request = AskRequest(message=QUERY)
    assert request.topP == 1.0
    reply, _ = await installed.provider.async_complete([{"role": "user", "content": QUERY}],
        temperature=request.temperature, max_tokens=request.maxTokens, top_p=request.topP)
    assert reply == ANSWER and installed.calls[0]["body"]["top_p"] == 1.0


@pytest.mark.asyncio
async def test_direct_cache_reuse_is_labelled_and_model_source_changes_miss(installed, monkeypatch):
    cache = Cache()
    monkeypatch.setattr(generate, "response_cache", cache)
    first = await receive()
    second = await receive()
    assert first.modelInvoked and not second.modelInvoked and first.reply == second.reply
    assert "cached_model_response" in second.warnings and len(installed.calls) == 1
    assert second.model_dump()["modelInvoked"] is False
    assert first.model_dump()["modelInvoked"] is True
    installed.stored["openaiCompatModel"] = "next-model"
    third = await receive()
    assert third.modelInvoked and len(installed.calls) == 2 and installed.calls[-1]["body"]["model"] == "next-model"
    installed.stored["openaiCompatApiKey"] = "fiction-next-key"
    fourth = await receive()
    assert fourth.modelInvoked and len(installed.calls) == 3 and installed.calls[-1]["auth"] == "Bearer fiction-next-key"


@pytest.mark.asyncio
async def test_required_configuration_failure_cannot_fall_through_cache_handling(installed, monkeypatch):
    cache = Cache()
    monkeypatch.setattr(generate, "response_cache", cache)
    failure = OSError("fiction-config-error")

    def fail():
        raise failure

    monkeypatch.setattr(model_manager, "_get_db_config", fail)
    with pytest.raises(HTTPException) as error:
        await receive()
    assert error.value.status_code == 503 and error.value.__cause__ is failure
    assert cache.reads == 0 and installed.calls == []


@pytest.mark.asyncio
async def test_enabled_retrieval_never_reuses_chat_response_cache(installed, monkeypatch):
    installed.config["useKnowledgeBase"] = True
    cache = Cache()
    monkeypatch.setattr(generate, "response_cache", cache)
    for _ in range(2):
        assert (await receive()).modelInvoked
    assert len(installed.calls) == 2 and cache.reads == cache.writes == 0


def test_cache_identity_separates_actual_provider_model_and_connection():
    request = MessageRequest(message=QUERY)
    first = {"provider": "openai_compat", "status": {"modelName": "fixture-model"}, "endpoint": "https://first.example"}
    variants = [dict(first, provider="vllm"), dict(first, status={"modelName": "next-model"}),
                dict(first, endpoint="https://next.example")]
    origin = generate._response_cache_keys(request, "default", {"useKnowledgeBase": False}, model_identity=first)
    for identity in variants:
        changed = generate._response_cache_keys(request, "default", {"useKnowledgeBase": False}, model_identity=identity)
        assert changed[0] == origin[0] and changed[1] != origin[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read", "structure", "empty", "write"])
async def test_response_cache_failures_are_explicit_and_never_become_success(installed, monkeypatch, failure):
    cache = Cache()
    monkeypatch.setattr(generate, "response_cache", cache)
    root_cause = OSError("fiction-cache-private-error")
    if failure == "read":
        cache.get = AsyncMock(side_effect=root_cause)
    elif failure == "structure":
        cache.get = AsyncMock(return_value={"model": "fiction-model", "costTime": 0.1})
    elif failure == "empty":
        cache.get = AsyncMock(return_value={})
    else:
        cache.set = AsyncMock(side_effect=root_cause)
    saved = AsyncMock(return_value=True)
    monkeypatch.setattr(generate, "_save_message", saved)
    with pytest.raises(HTTPException) as error:
        await generate._generate_reply_impl(MessageRequest(message=QUERY, loraName="default"),
            persist_message=failure in {"structure", "empty"}, record_invocation=False)
    assert error.value.status_code == 503
    assert "fiction-cache-private" not in str(error.value.detail)
    if failure == "write":
        assert len(installed.calls) == 1 and error.value.__cause__ is root_cause
        assert error.value.detail["stage"] == "response_cache"
        assert error.value.detail["modelInvoked"] is True and error.value.detail["messageSaved"] is False
    else:
        assert installed.calls == []
        assert error.value.__cause__ is not None
    saved.assert_not_awaited()
    installed.manager.async_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_no_model_result_keeps_false_invocation_provenance(installed, monkeypatch):
    async def abstain(*args, **kwargs):
        return "当前资料未找到足够依据。", False, {"modelInvoked": False, "abstained": True, "answerMode": "abstention"}

    monkeypatch.setattr(generate, "_generate_with_retrieval", abstain)
    result = await receive()
    assert result.abstained and not result.modelInvoked and result.model_dump()["modelInvoked"] is False
    assert installed.calls == []
    assert not generate._security_policy_response().modelInvoked
    assert not GenerateResponse(reply="已提交回复的幂等复用", model="fixture", costTime=0).modelInvoked


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [True, False])
async def test_cache_hit_checks_archive_result_without_new_model_call(installed, monkeypatch, saved):
    cache = Cache()
    monkeypatch.setattr(generate, "response_cache", cache)
    assert (await receive()).modelInvoked
    failure = OSError("fiction-cache-archive-error")
    def add_message(record):
        if not saved:
            raise failure
        return {"id": "cached-fixture-row", **record}
    generate.db.add_message = add_message
    archive = AsyncMock(wraps=generate._save_message)
    monkeypatch.setattr(generate, "_save_message", archive)
    request = MessageRequest(message=QUERY, loraName="default")
    if saved:
        result = await generate._generate_reply_impl(request, persist_message=True, record_invocation=False)
        assert result.reply == ANSWER and not result.modelInvoked
    else:
        with pytest.raises(HTTPException) as error:
            await generate._generate_reply_impl(request, persist_message=True, record_invocation=False)
        assert error.value.status_code == 503 and error.value.detail["stage"] == "message_archive"
        assert error.value.detail["modelInvoked"] is False and error.value.detail["messageSaved"] is None
        assert error.value.__cause__ is failure
    archive.assert_awaited_once()
    assert archive.await_args.args[1] == ANSWER
    assert len(installed.calls) == 1 and cache.writes == 1


@pytest.mark.asyncio
async def test_snapshot_owns_settings_and_next_request_reads_new_configuration(installed):
    snapshot = installed.provider.snapshot()
    installed.stored.update(openaiCompatModel="next-model", openaiCompatBaseUrl="https://next.example/v1",
                            openaiCompatApiKey="fiction-next-key")
    installed.provider._refresh_db_config()
    await snapshot.async_complete([{"role": "user", "content": QUERY}], temperature=0.1, max_tokens=173)
    first = installed.calls[-1]
    assert first["body"]["model"] == "fixture-model" and first["auth"] == "Bearer fiction-key"
    assert first["endpoint"] == "https://api.deepseek.com/v1/chat/completions"
    assert snapshot.get_status()["modelName"] == "fixture-model"
    assert installed.provider.snapshot().get_status()["modelName"] == "next-model"


@pytest.mark.asyncio
async def test_cache_identity_and_wire_use_same_snapshot_when_settings_change_before_dispatch(installed, monkeypatch):
    cache = Cache()
    monkeypatch.setattr(generate, "response_cache", cache)
    compiled = generate._generate_with_retrieval
    async def changed(*args, **kwargs):
        installed.stored.update(openaiCompatModel="next-model", openaiCompatApiKey="fiction-next-key")
        installed.provider._refresh_db_config()
        return await compiled(*args, **kwargs)
    monkeypatch.setattr(generate, "_generate_with_retrieval", changed)
    first = await receive()
    assert first.model == "fixture-model (openai_compat)"
    assert installed.calls[0]["body"]["model"] == "fixture-model" and installed.calls[0]["auth"] == "Bearer fiction-key"
    second = await receive()
    assert second.model == "next-model (openai_compat)" and len(installed.calls) == 2
    assert installed.calls[1]["body"]["model"] == "next-model" and installed.calls[1]["auth"] == "Bearer fiction-next-key"
    installed.stored.update(openaiCompatModel="fixture-model", openaiCompatApiKey="fiction-key")
    third = await receive()
    assert not third.modelInvoked and third.model == "fixture-model (openai_compat)" and len(installed.calls) == 2


@pytest.mark.asyncio
async def test_configuration_change_during_transport_does_not_relabel_completed_reply(installed):
    def change():
        installed.stored["openaiCompatModel"] = "next-model"
        installed.provider._refresh_db_config()
    installed.state.on_request = change
    result = await receive()
    assert installed.calls[0]["body"]["model"] == "fixture-model"
    assert result.model == "fixture-model (openai_compat)" and installed.provider.model == "next-model"


@pytest.mark.asyncio
async def test_multiple_model_calls_in_one_turn_share_configuration_snapshot(installed, monkeypatch):
    compiled = generate._generate_with_retrieval
    async def two_calls(*args, **kwargs):
        await compiled(*args, **kwargs)
        installed.stored["openaiCompatModel"] = "next-model"
        installed.provider._refresh_db_config()
        return await compiled(*args, **kwargs)
    monkeypatch.setattr(generate, "_generate_with_retrieval", two_calls)
    result = await receive()
    assert len(installed.calls) == 2 and {call["body"]["model"] for call in installed.calls} == {"fixture-model"}
    assert result.model == "fixture-model (openai_compat)"


@pytest.mark.asyncio
async def test_cache_snapshot_keeps_its_provider_type_when_manager_changes_during_lookup(installed, monkeypatch):
    monkeypatch.setattr(generate, "response_cache", Cache())
    first = await receive()
    original = installed.provider.snapshot
    def changed():
        snapshot = original()
        installed.manager._current_provider = model_manager.ModelProvider.OLLAMA
        return snapshot
    monkeypatch.setattr(installed.provider, "snapshot", changed)
    second = await receive()
    assert first.model == second.model == "fixture-model (openai_compat)"
    assert not second.modelInvoked and len(installed.calls) == 1
    installed.manager.async_generate.assert_not_awaited()
