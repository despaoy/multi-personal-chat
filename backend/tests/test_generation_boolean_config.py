"""Configuration controls exercise the real shared generation and cache key.

The provider and model are explicit controls; no cloud or vector evaluation.
"""

from services import chat_answer

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from provider_fixtures import cloud_connection

from api import generate
from db.schemas import GenerateResponse, MessageRequest
from inference.request_context import InferenceRequestContext, use_request_context
from inference.token_counting import text_token_counter
from tests.test_native_generation_concurrency import installed as installed


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "65536")
    monkeypatch.setenv("RAG_CITATIONS_ENABLED", "true")
    monkeypatch.setenv("CHARACTER_INDEPENDENT_TASKS_ENABLED", "false")
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "合成角色：准确回应当前问题。")
    source = Mock(side_effect=AssertionError("No public source read is expected"))
    monkeypatch.setattr(generate, "db", SimpleNamespace(get_knowledge_document=source))
    monkeypatch.setattr(chat_answer, "public_knowledge_db", generate.db)
    rag = AsyncMock(side_effect=AssertionError("Disabled or invalid configuration must not retrieve"))
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', rag)
    provider = cloud_connection("boolean-config-control")
    context = InferenceRequestContext("openai_compat", provider,
                                      text_token_counter("openai_compat", provider.base_url, provider.model))
    model = AsyncMock(return_value="你好，很高兴见到你。")
    request = MessageRequest(message="请用一句话打招呼。", platform="web", adapter="boolean-config",
                             senderId="fiction-owner", conversationId="fiction-owner", conversationType="private")
    with use_request_context(context):
        yield SimpleNamespace(request=request, model=model, rag=rag, source=source, provider=provider)
    provider.async_complete.assert_not_awaited()


def config(use_kb=False):
    return dict(useKnowledgeBase=use_kb, temperature=.7, maxTokens=256, topP=.9)


@pytest.mark.parametrize("name", ["RAG_CITATIONS_ENABLED", "CHARACTER_INDEPENDENT_TASKS_ENABLED"])
async def test_invalid_environment_boolean_stops_before_model(control, monkeypatch, name):
    monkeypatch.setenv(name, "synthetic-private-invalid-boolean")
    with pytest.raises(ValueError, match=name) as caught:
        await chat_answer.generate_chat_answer(control.request, None, runtime_config=config(), model_generate=control.model)
    assert "synthetic-private" not in str(caught.value)
    control.model.assert_not_awaited()
    control.rag.assert_not_awaited()
    control.source.assert_not_called()


async def test_invalid_knowledge_base_setting_stops_before_retrieval(control):
    with pytest.raises(ValueError, match="useKnowledgeBase"):
        await chat_answer.generate_chat_answer(control.request, None, runtime_config=config("invalid"),
                                                model_generate=control.model)
    control.model.assert_not_awaited()
    control.rag.assert_not_awaited()


@pytest.mark.parametrize("setting", [False, "false"])
async def test_disabled_knowledge_base_reaches_model_without_retrieval(control, setting):
    reply, used, _meta = await chat_answer.generate_chat_answer(
        control.request, None, runtime_config=config(setting), model_generate=control.model
    )
    assert reply == "你好，很高兴见到你。" and used is False
    control.model.assert_awaited_once()
    control.rag.assert_not_awaited()
    control.source.assert_not_called()


async def test_explicit_rag_disable_keeps_precedence(control):
    reply, used, _meta = await chat_answer.generate_chat_answer(
        control.request, None, runtime_config=config(True), enable_rag=False, model_generate=control.model
    )
    assert reply == "你好，很高兴见到你。" and used is False
    control.model.assert_awaited_once()
    control.rag.assert_not_awaited()


@pytest.mark.parametrize("setting,text", [(False, "false"), (True, "true")])
def test_cache_identity_uses_same_explicit_boolean_meaning(control, setting, text):
    identity = dict(provider="openai_compat", status=control.provider.get_status(),
                    endpoint=control.provider.base_url, model=control.provider.model)
    assert generate._response_cache_keys(control.request, "default", config(setting), model_identity=identity) == (
        generate._response_cache_keys(control.request, "default", config(text), model_identity=identity)
    )


def test_invalid_cache_configuration_cannot_create_a_cache_key(control):
    with pytest.raises(ValueError, match="useKnowledgeBase"):
        generate._response_cache_keys(control.request, "default", config("invalid"), model_identity={"model": "control"})


@pytest.mark.parametrize("setting", [None, False])
async def test_cache_hit_requires_valid_configuration(installed, monkeypatch, setting):
    state, _manager, database, _semaphore = installed
    database.config["useKnowledgeBase"] = setting
    cached = GenerateResponse(reply="缓存中的完整合成回复。", model="openai_compat/fixture-cloud", costTime=.01)
    cache = SimpleNamespace(get=AsyncMock(return_value=cached.model_dump()))
    monkeypatch.setattr(generate, "response_cache", cache)
    request = MessageRequest(message="按完整合成输入回答。", platform="web", adapter="cache-boolean",
                             senderId="fiction-owner", conversationId="fiction-owner", conversationType="private")
    if setting is None:
        with pytest.raises(ValueError, match="useKnowledgeBase"):
            await generate._generate_reply_impl(request, persist_message=False, record_invocation=False, message_db=generate.db)
        cache.get.assert_not_awaited()
    else:
        result = await generate._generate_reply_impl(request, persist_message=False, record_invocation=False, message_db=generate.db)
        assert result.reply == cached.reply and result.model == cached.model
        assert result.modelInvoked is False and "cached_model_response" in result.warnings
        cache.get.assert_awaited_once()
    assert state.calls == []


async def test_falsey_explicit_mapping_keeps_disabled_retrieval_and_parameters(control):
    class FalsyMapping(dict):
        def __bool__(self):
            return False

    reply, used, _meta = await chat_answer.generate_chat_answer(
        control.request, None, runtime_config=FalsyMapping(config()), model_generate=control.model)
    assert reply == "你好，很高兴见到你。" and used is False
    control.model.assert_awaited_once()
    assert control.model.await_args.kwargs["max_tokens"] == 256
    control.rag.assert_not_awaited()
    control.source.assert_not_called()
