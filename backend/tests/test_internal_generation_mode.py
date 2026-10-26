"""Internal generation mode must not persist chat rows or model invocations."""

from __future__ import annotations

from services import chat_answer

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection

from db.schemas import MessageRequest
from services.chat_answer import ChatAnswer


@pytest.fixture
async def completion_runtime(monkeypatch):
    from services import turn_completion
    runtime = turn_completion.TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", lambda _owner_database=None: runtime)
    yield runtime
    await runtime.shutdown(timeout=1)


def _request():
    return MessageRequest(message="tool-json", sessionType="private", sessionId="claw")


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,label', [('current_constraint', 'statement/constraint'), ('memory_lookup', 'memory/lookup'), ('memory_storage_status', 'memory/storage_status'), ('memory_mentions', 'memory/mentions'), ('memory_field_result', 'memory/field_result'), ('memory_operation', 'memory/operation'), ('task_composite', 'tasks/composite'), ('source_excerpt', 'rag/source_excerpt'), ('source_unavailable', 'rag/abstained'), ('abstention', 'rag/abstained')])
@pytest.mark.parametrize('provider', ['vllm', 'openai_compat'])
async def test_deterministic_response_has_no_model_invocation(monkeypatch, mode, label, provider):
    import api.generate as gen
    from inference import model_manager as mm
    saved, recorded = _install_common(monkeypatch, gen)
    connection = cloud_connection('not-called')
    connection.name = provider
    connection.set_lora_adapter = lambda _: None

    class Manager:
        _current_provider = SimpleNamespace(value=provider)

        def get_current_provider(self):
            return connection

        def set_lora_adapter(self, value):
            pass

        def get_status(self):
            return {'currentProvider': provider, 'providers': {provider: {'modelName': 'not-called'}}}

        async def async_generate(self, **kwargs):
            pytest.fail('No model invocation expected')

    async def deterministic(*args, **kwargs):
        return ChatAnswer(reply='当前声明的回应', used_rag=False, model_invoked=False, metadata={'answerMode': mode})
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: Manager())
    monkeypatch.setattr(gen, 'get_vllm_client', AsyncMock(return_value=SimpleNamespace(generate=AsyncMock())))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', deterministic)
    monkeypatch.setattr(gen, 'get_llm_semaphore', lambda: asyncio.Semaphore(2))
    result = await gen._generate_reply_impl(_request(), persist_message=True, enable_rag=False, record_invocation=True, message_db=gen.db)
    assert result.model == label and recorded == []
    assert len(saved) == 1 and saved[0][2] == label and (saved[0][3] == 'default')


def _install_common(monkeypatch, gen):
    monkeypatch.setattr(gen, "response_cache", None)
    monkeypatch.setattr(gen, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(gen, "circuit_breaker_registry", None)
    saved = []
    recorded = []
    monkeypatch.setattr(gen, "_save_message", _record_save(saved))
    monkeypatch.setattr(gen, "_record_model_invocation", _record_invocation(recorded))
    return saved, recorded


def _record_save(saved):
    async def fake(*args, **kwargs):
        saved.append(args)
    return fake


def _record_invocation(recorded):
    async def fake(*args, **kwargs):
        recorded.append(args)
    return fake


@pytest.mark.asyncio
async def test_internal_mode_success_does_not_persist(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm
    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value='vllm')

        def get_current_provider(self):
            return SimpleNamespace(name='vllm')
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: FakeManager())
    monkeypatch.setattr(gen, 'get_vllm_client', AsyncMock(return_value=SimpleNamespace(generate=AsyncMock())))

    async def fake_vllm(*args, **kwargs):
        return ChatAnswer(reply='tool-reply', used_rag=False, model_invoked=True, metadata={})
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', fake_vllm)
    result = await gen._generate_reply_impl(_request(), persist_message=False, enable_rag=False, record_invocation=False, message_db=gen.db)
    assert result.reply == 'tool-reply'
    assert saved == []
    assert recorded == []


@pytest.mark.asyncio
async def test_internal_mode_vllm_failure_does_not_persist(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm
    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value='vllm')

        def get_current_provider(self):
            return SimpleNamespace(name='vllm')

        def set_lora_adapter(self, value):
            pass

        async def async_generate(self, **kwargs):
            return ('fallback-reply', 0.01)

        def get_status(self):
            return {'currentProvider': 'vllm', 'providers': {'vllm': {'modelName': 'fake'}}}
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: FakeManager())
    monkeypatch.setattr(gen, 'get_vllm_client', AsyncMock(return_value=SimpleNamespace(generate=AsyncMock())))
    monkeypatch.setattr(gen, 'get_llm_semaphore', lambda: asyncio.Semaphore(2))

    async def fake_vllm(*args, **kwargs):
        raise RuntimeError('vllm down')
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', fake_vllm)
    with pytest.raises(HTTPException) as caught:
        await gen._generate_reply_impl(_request(), persist_message=False, enable_rag=False, record_invocation=False, message_db=gen.db)
    assert caught.value.status_code == 503
    assert saved == []
    assert recorded == []


@pytest.mark.asyncio
async def test_internal_mode_model_manager_failure_does_not_record_invocation(monkeypatch, tmp_path):
    import api.generate as gen
    from inference import model_manager as mm
    saved, recorded = _install_common(monkeypatch, gen)
    from db.database import SQLiteDB
    monkeypatch.setattr(gen, 'db', SQLiteDB(tmp_path / 'internal-provider-error.sqlite'))
    failure = RuntimeError('model manager down')
    connection = cloud_connection('fixture-model', completion=AsyncMock(side_effect=failure))

    class FakeManager:
        _current_provider = SimpleNamespace(value='openai_compat')

        def get_current_provider(self):
            return connection

        def set_lora_adapter(self, value):
            pass

        async def async_generate(self, **kwargs):
            pytest.fail('Cloud failure must reach the selected connection async_complete')
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: FakeManager())
    monkeypatch.setattr(gen, 'get_llm_semaphore', lambda: asyncio.Semaphore(2))
    with pytest.raises(HTTPException) as caught:
        await gen._generate_reply_impl(_request(), persist_message=False, enable_rag=False, record_invocation=False, message_db=gen.db)
    assert caught.value.status_code == 500 and caught.value.__cause__ is failure
    connection.async_complete.assert_awaited_once()
    assert saved == []
    assert recorded == []


@pytest.mark.asyncio
@pytest.mark.parametrize('persist,delivery,allowed', [(False, None, False), (True, {}, False), (True, None, True)])
async def test_explicit_memory_operations_respect_internal_and_delivery_modes(monkeypatch, persist, delivery, allowed, completion_runtime):
    import api.generate as gen
    from inference import model_manager as mm
    _install_common(monkeypatch, gen)
    monkeypatch.setattr(gen, 'db', SimpleNamespace(config={}, loras=[]))
    from character.models import RelationshipState, UserScope
    seen = []

    async def prepare(request, character_id, *, character_service=None, execute_memory_operations=False, defer_memory_operations=False):
        seen.append((execute_memory_operations, defer_memory_operations))
        return SimpleNamespace(character_id=character_id, user_scope=UserScope('web', 'fixture', 'owner', 's1', 'private'), relationship=RelationshipState(), history=(), received_at=datetime.now(timezone.utc), compiled=SimpleNamespace(used_memory_ids=()), memory_operation_receipt=None)

    async def generate(*args, **kwargs):
        return ChatAnswer(reply='reply', used_rag=False, model_invoked=True, metadata={})
    monkeypatch.setattr(gen, '_prepare_character_turn', prepare)
    monkeypatch.setattr(gen, '_complete_character_turn', AsyncMock(return_value=None))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', generate)
    monkeypatch.setattr(gen, 'get_vllm_client', AsyncMock(return_value=SimpleNamespace(generate=AsyncMock())))
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: SimpleNamespace(_current_provider=SimpleNamespace(value='vllm'), get_current_provider=lambda: SimpleNamespace(name='vllm')))
    request = MessageRequest(message='忘掉我的住址。', characterId='tsukiyashiro_kisaki', loraName='default', sessionType='private', sessionId='s1')
    await gen._generate_reply_impl(request, persist_message=persist, delivery_context=delivery, enable_rag=False, record_invocation=False, message_db=gen.db)
    assert seen == [(allowed, persist and delivery is not None)]
    if persist and delivery is not None:
        assert delivery['completion_snapshot']['user_scope']['sender_id'] == 'owner'
    assert completion_runtime.reserved == completion_runtime.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['vllm', 'openai_compat'])
async def test_message_db_reaches_save_and_invocation_record(monkeypatch, provider):
    """容器注入的数据库必须贯穿消息保存与模型调用记录。

    只有人物服务走容器库、消息仍写全局库时，自定义容器实例的
    上一轮消息写入全局库，下一轮人物历史从容器库读不到，历史断裂。
    """
    import api.generate as gen
    from inference import model_manager as mm
    _install_common(monkeypatch, gen)
    connection = cloud_connection('unit-model')
    connection.name = provider
    connection.set_lora_adapter = lambda _: None

    class FakeManager:
        _current_provider = SimpleNamespace(value=provider)

        def get_current_provider(self):
            return connection

        def set_lora_adapter(self, value):
            pass

        def get_status(self):
            return {'currentProvider': provider, 'providers': {provider: {'modelName': 'unit-model'}}}
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: FakeManager())
    monkeypatch.setattr(gen, 'get_vllm_client', AsyncMock(return_value=SimpleNamespace(generate=AsyncMock())))

    async def fake_vllm(*args, **kwargs):
        return ChatAnswer(reply='reply', used_rag=False, model_invoked=True, metadata={})
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', fake_vllm)
    captured = {}

    async def fake_save(request, reply, model_name, lora_name, cost_time, *, database=None, character_id=None, saved_receipt=None, model_invoked=None):
        captured['save_db'] = database
        return True

    async def fake_record(request, model_name, lora_name, cost_time, **kwargs):
        captured['record_db'] = kwargs.get('database')
    monkeypatch.setattr(gen, '_save_message', fake_save)
    monkeypatch.setattr(gen, '_record_model_invocation', fake_record)
    sentinel_db = SimpleNamespace(config={}, loras=[])
    result = await gen._generate_reply_impl(_request(), persist_message=True, enable_rag=False, record_invocation=True, message_db=sentinel_db)
    assert result.reply == 'reply'
    assert captured['save_db'] is sentinel_db
    assert captured['record_db'] is sentinel_db


@pytest.mark.asyncio
async def test_save_message_and_invocation_use_injected_database(monkeypatch):
    """_save_message/_record_model_invocation 必须写入注入的数据库对象。"""
    import api.generate as gen

    calls = []

    class FakeDB:
        def add_message(self, record):
            calls.append(("message", record))
            return {"id": "fixture-message", **record}

        def add_model_invocation(self, record):
            calls.append(("invocation", record))
            return 1

    sentinel_db = FakeDB()
    request = _request()

    ok = await gen._save_message(
        request, "reply", "test-model", "default", 0.1, database=sentinel_db
    )
    assert ok is True
    assert calls[-1][0] == "message"
    assert calls[-1][1]["message"] == request.message

    await gen._record_model_invocation(
        request, "test-model", "default", 0.1, database=sentinel_db
    )
    assert calls[-1][0] == "invocation"
    # 两次写入都落在注入的数据库上（而非全局单例）
    assert len(calls) == 2


def test_lora_path_resolution_uses_injected_loras(monkeypatch):
    """LoRA路径只解析调用方已经读取的列表，不选择其他数据库。"""
    from db import database as db_mod

    monkeypatch.setattr(
        db_mod, "refresh_lora_dir_map", lambda: {"kisaki": "/loras/kisaki"}
    )
    container_loras = [{"id": 42, "name": "kisaki"}]

    # 注入容器列表：能解析到路径
    assert db_mod.get_lora_path_by_id(42, loras=container_loras) == "/loras/kisaki"
    # 合法的空登记列表保持无匹配。
    assert db_mod.get_lora_path_by_id(42, loras=[]) is None


@pytest.mark.asyncio
async def test_unavailable_selected_vllm_never_dispatches_another_provider(monkeypatch):
    """The removed fallback is an explicit unavailable error before any dispatch."""
    import api.generate as gen
    from inference import model_manager as mm
    saved, recorded = _install_common(monkeypatch, gen)
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value='vllm'), get_current_provider=lambda: SimpleNamespace(name='vllm'), async_generate=AsyncMock())
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: manager)
    monkeypatch.setattr(gen, 'get_vllm_client', AsyncMock(return_value=None))
    retrieval = AsyncMock()
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', retrieval)
    with pytest.raises(HTTPException) as error:
        await gen._generate_reply_impl(_request(), persist_message=False, enable_rag=False, record_invocation=False, message_db=gen.db)
    assert error.value.status_code == 503 and saved == recorded == []
    manager.async_generate.assert_not_awaited()
    retrieval.assert_not_awaited()


@pytest.mark.asyncio
async def test_injected_character_service_reaches_prepare_and_complete(monkeypatch, completion_runtime):
    """HTTP 层注入的容器角色服务必须贯穿准备与回写两个阶段。

    create_app(custom_container) 的生成链路若回退全局默认服务，
    人物记忆/关系会读写全局数据库，与管理接口（走容器）脱节；
    且准备与回写必须使用同一服务实例（同一容器数据库）。
    """
    import api.generate as gen
    from inference import model_manager as mm
    _install_common(monkeypatch, gen)
    monkeypatch.setattr(gen, 'db', SimpleNamespace(config={}, loras=[]))

    async def fake_save(*args, **kwargs):
        return True
    monkeypatch.setattr(gen, '_save_message', fake_save)
    container_service = object()
    seen = {}
    fake_prepared = SimpleNamespace(character_id='tsukiyashiro_kisaki', compiled=SimpleNamespace(profile_context='人物画像', dynamic_context='关系动态', reference_context='长期记忆'), history=())

    async def fake_prepare(request, character_id, *, character_service=None, execute_memory_operations=False, defer_memory_operations=False):
        seen['prepare_service'] = character_service
        return fake_prepared

    async def fake_complete(prepared, request, *, character_service=None, completion_slot=None, message_db=None):
        seen['complete_service'] = character_service
        assert completion_slot is not None
    monkeypatch.setattr(gen, '_prepare_character_turn', fake_prepare)
    monkeypatch.setattr(gen, '_complete_character_turn', fake_complete)

    class FakeManager:
        _current_provider = SimpleNamespace(value='vllm')

        def get_current_provider(self):
            return SimpleNamespace(name='vllm')
    monkeypatch.setattr(mm, 'get_model_manager', lambda *, database=None: FakeManager())

    async def fake_vllm(*args, **kwargs):
        return ChatAnswer(reply='reply', used_rag=False, model_invoked=True, metadata={})
    monkeypatch.setattr(gen, 'get_vllm_client', AsyncMock(return_value=SimpleNamespace(generate=AsyncMock())))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', fake_vllm)
    request = MessageRequest(message='你好', characterId='tsukiyashiro_kisaki', loraName='default', sessionType='private', sessionId='s1')
    result = await gen._generate_reply_impl(request, persist_message=True, enable_rag=False, record_invocation=False, character_service=container_service, message_db=gen.db)
    assert result.reply == 'reply'
    assert seen['prepare_service'] is container_service
    assert seen['complete_service'] is container_service
