"""Native providers must receive compiled messages and explicit output budgets."""

from services import chat_answer

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager

import httpx

from provider_fixtures import completed_provider_response

import pytest

from db.schemas import MessageRequest
from services.chat_answer import ChatAnswer


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,available', [('openai_compat', True), ('openai_compat', False), ('ollama', True), ('vllm', False), ('vllm', True)])
async def test_selected_provider_receives_messages_and_budget_without_local_override(monkeypatch, provider, available):
    from api import generate
    from inference import model_manager
    calls = []

    class Manager:
        name = provider
        _current_provider = SimpleNamespace(value=provider)

        def set_lora_adapter(self, value):
            pass

        def get_status(self):
            return {'modelName': 'selected-native', 'currentProvider': provider, 'providers': {provider: {'modelName': 'selected-native'}}}

        def get_current_provider(self):
            return self

        def snapshot(self):
            return SimpleNamespace(base_url='https://native.invalid/v1', model='selected-native', async_complete=self.async_complete, get_status=lambda: {'modelName': 'selected-native'})

        async def async_complete(self, messages, *, temperature, max_tokens, top_p, frequency_penalty, repetition_penalty, enable_thinking, lora_name):
            calls.append(dict(messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, repetition_penalty=repetition_penalty, enable_thinking=enable_thinking, lora_name=lora_name))
            return ('原生提供方回答', 0.01)
    manager = Manager()
    manager._providers = {model_manager.ModelProvider(provider): manager}
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: manager)
    monkeypatch.setattr(generate, 'response_cache', None)
    monkeypatch.setattr(generate, 'db', SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(generate, 'circuit_breaker_registry', None)
    monkeypatch.setattr(generate, 'get_llm_semaphore', lambda: asyncio.Semaphore(1))
    local_call = AsyncMock(return_value='原生提供方回答', side_effect=None if provider == 'vllm' else AssertionError('Cloud must not call local inference'))
    available_check = AsyncMock(return_value=SimpleNamespace(generate=local_call) if available else None)
    monkeypatch.setattr(generate, 'get_vllm_client', available_check)

    async def compiled_retrieval(request, lora_name, *, model_generate, **kwargs):
        reply = await model_generate(messages=[{'role': 'system', 'content': '编译后角色及表达边界'}, {'role': 'user', 'content': '当前住处丽水；完整已知记忆和当前问题'}], temperature=0.17, max_tokens=733, top_p=1.0, frequency_penalty=0.0, repetition_penalty=1.0, enable_thinking=False, lora_name=None)
        return ChatAnswer(reply=reply, used_rag=False, model_invoked=True, metadata={})
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', compiled_retrieval)
    if provider == 'vllm' and (not available):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as caught:
            await generate._generate_reply_impl(MessageRequest(message='我的当前住处是哪？'), persist_message=False, record_invocation=False, enable_rag=False, message_db=generate.db)
        assert caught.value.status_code == 503
        assert not calls
        local_call.assert_not_awaited()
        return
    response = await generate._generate_reply_impl(MessageRequest(message='我的当前住处是哪？'), persist_message=False, record_invocation=False, enable_rag=False, message_db=generate.db)
    assert response.reply == '原生提供方回答'
    if provider != 'vllm':
        assert calls == [dict(messages=[{'role': 'system', 'content': '编译后角色及表达边界'}, {'role': 'user', 'content': '当前住处丽水；完整已知记忆和当前问题'}], temperature=0.17, max_tokens=733, top_p=1.0, frequency_penalty=0.0, repetition_penalty=1.0, enable_thinking=False, lora_name=None)]
    elif provider == 'vllm':
        assert not calls
        local_call.assert_awaited_once()
        assert local_call.await_args.kwargs['max_tokens'] == 733
        assert local_call.await_args.kwargs['messages'][-1]['content'] == '当前住处丽水；完整已知记忆和当前问题'
    if provider != 'vllm':
        local_call.assert_not_awaited()
        available_check.assert_not_awaited()



@pytest.fixture(params=['OllamaProvider', 'LlamaCppProvider'])
def native_http(request, monkeypatch):
    from api import generate
    from inference import model_manager

    def forbidden_global_config(_owner_database=None):
        pytest.fail('A compiled request must not reread a different global configuration')
    monkeypatch.setattr(model_manager, '_get_db_config', forbidden_global_config)
    provider = getattr(model_manager, request.param)()
    manager = SimpleNamespace(get_current_provider=lambda: provider)
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: manager)
    monkeypatch.setattr(generate, 'response_cache', None)
    monkeypatch.setattr(generate, 'circuit_breaker_registry', None)
    monkeypatch.setattr(generate, 'get_llm_semaphore', lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: '只依据完整当前资料，保留否定和条件。')
    bodies = []

    def respond(request):
        import json
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completed_provider_response(provider.__class__.__name__, '总共六元；尚未购买，下雨则在线购买。'))

    @asynccontextmanager
    async def acquire(timeout, **_resource_options):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client
    monkeypatch.setattr(provider, '_acquire_http_client', acquire)
    return SimpleNamespace(api=generate, provider=provider, bodies=bodies)


async def test_native_main_request_keeps_its_own_database_options_and_complete_messages(native_http):
    rig = native_http
    database = SimpleNamespace(config=dict(useKnowledgeBase=False, temperature=.17, topP=.23, maxTokens=733), loras=[])
    request = MessageRequest(loraName="default", message="三张票每张两元。尚未购买，下雨则在线购买。请计算总额，并保留状态及条件。",
        history=[{"role": "user", "content": "此前数量为两张，本轮已更正，请以本轮为准。"},
                 {"role": "assistant", "content": "明白。"}])
    result = await rig.api._generate_reply_impl(request, persist_message=False, record_invocation=False,
        enable_rag=False, message_db=database)
    assert result.modelInvoked and len(rig.bodies) == 1
    body = rig.bodies[0]
    options = body["options"] if rig.provider.name == "ollama" else body
    assert options["temperature"] == .17 and options["top_p"] == .23
    assert options["num_predict" if rig.provider.name == "ollama" else "max_tokens"] == 733
    assert request.message in body["messages"][-1]["content"]
    assert all(item in body["messages"] for item in request.history)


async def test_native_http_maps_explicit_penalties_and_nonthinking_without_config_reads(native_http):
    rig = native_http
    messages = [{"role": "system", "content": "Keep full source facts."},
                {"role": "user", "content": "三张票共六元，尚未购买，下雨改在线。"}]
    reply, _ = await rig.provider.async_complete(messages, temperature=.17, max_tokens=733, top_p=.23,
        frequency_penalty=.31, repetition_penalty=1.07, enable_thinking=False, lora_name=None)
    assert reply and len(rig.bodies) == 1
    body = rig.bodies[0]
    options = body["options"] if rig.provider.name == "ollama" else body
    assert options["repeat_penalty"] == 1.07 and options["frequency_penalty"] == .31
    assert options["temperature"] == .17 and options["top_p"] == .23 and body["messages"] == messages
    if rig.provider.name == "ollama":
        assert body["think"] is False
    else:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize('label,temperature', [(None, 0.0), ('adapter-a', 0.17)])
async def test_transformer_main_keeps_explicit_options_and_the_actual_bound_adapter(tmp_path, monkeypatch, label, temperature):
    from api import generate
    from inference import model_manager
    from test_transformer_lifecycle import REPLIES, arrange_transformer_lifecycle
    rig = arrange_transformer_lifecycle(tmp_path, monkeypatch)
    provider = rig.provider
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: SimpleNamespace(get_current_provider=lambda: provider))
    monkeypatch.setattr(model_manager, '_get_db_config', lambda _owner_database=None: pytest.fail('Main native completion cannot use global settings'))
    monkeypatch.setattr(generate, 'response_cache', None)
    monkeypatch.setattr(generate, 'circuit_breaker_registry', None)
    monkeypatch.setattr(generate, 'get_llm_semaphore', lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(generate, 'get_lora_character_id', lambda _: None)
    monkeypatch.setattr(generate, 'get_lora_path_by_id', lambda *args, **kwargs: str(rig.adapters[label]))

    async def compiled(request, lora_name, *, model_generate, runtime_config, **kwargs):
        reply = await model_generate(messages=[{'role': 'system', 'content': 'Keep full request.'}, {'role': 'user', 'content': request.message}], temperature=runtime_config['temperature'], max_tokens=runtime_config['maxTokens'], top_p=runtime_config['topP'], repetition_penalty=1.07, frequency_penalty=0.0, enable_thinking=False, lora_name=lora_name if lora_name != 'default' else None)
        return ChatAnswer(reply=reply, used_rag=False, model_invoked=True, metadata={})
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', compiled)
    database = SimpleNamespace(config=dict(useKnowledgeBase=False, temperature=temperature, topP=0.23, maxTokens=733), loras=[dict(id=1, name='bound-adapter', status='active')] if label else [])
    result = await generate._generate_reply_impl(MessageRequest(message='Complete isolated request.', loraName='bound-adapter' if label else 'default'), persist_message=False, record_invocation=False, enable_rag=False, message_db=database)
    assert result.reply == REPLIES[label] and rig.generations == [label]
    options, = rig.generation_options
    assert options['max_new_tokens'] == 733 and options['repetition_penalty'] == 1.07
    assert options['do_sample'] is (temperature > 0)
    if temperature > 0:
        assert options['temperature'] == temperature and options['top_p'] == 0.23
    else:
        assert 'temperature' not in options and 'top_p' not in options
    assert rig.template_options[0]['enable_thinking'] is False
