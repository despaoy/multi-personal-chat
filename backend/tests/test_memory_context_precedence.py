"""A saved projection cannot certify current facts over unprocessed context."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from character.models import CompiledCharacterContext, MemoryItem
from inference.generation_request import GenerationRequest, generate_character_response
from inference.memory_response import render_memory_response


def saved():
    item = MemoryItem('saved', 'user_fact', '用户说自己居住在银川',
        memory_key='user_residence', evidence=('我住在银川。',), source_message_ids=('old-source',))
    return CompiledCharacterContext('', '', '已保存的用户陈述：我住在银川。', ('saved',),
        memory_status='available', memory_packets=(item,), memory_field_presence=(('residence', True),))


def test_empty_user_frames_do_not_disable_fast_path():
    assert render_memory_response('我住哪里？', saved(), history=({'role': 'user', 'content': '  '},))


def test_independent_task_cannot_hide_history_dependency():
    from inference.generation_request import RetrievalResult
    from inference.task_execution import prepare_independent_tasks

    message = '先说我住哪里，再说林远是谁。'
    retrieval = RetrievalResult(status='ok', evidence='林远是学生。',
        identity_subtask={'query': message, 'subject': '林远'})
    req = GenerationRequest(message=message, character_context=saved(), retrieval=retrieval,
        independent_tasks_enabled=True, history=({'role': 'user', 'content': '地址不对了。'},))
    assert prepare_independent_tasks(replace(req, history=())) is not None
    assert prepare_independent_tasks(req) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('history', [
    ({'role': 'user', 'content': '我现在住在西宁。'},),
    ({'role': 'user', 'content': '上次那个地址不是我的，是我朋友的。'},),
    ({'role': 'user', 'content': '那个信息已经不对了，新的我以后再告诉你。'},),
])
async def test_saved_fact_does_not_bypass_unresolved_user_history(history):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return '先依据最近的更正回答。'

    result = await generate_character_response(GenerationRequest(message='我住哪里？',
        character_context=saved(), history=history), generate)
    assert result.model_invoked and len(calls) == 1
    assert any(m['content'] == history[0]['content'] for m in calls[0]['messages'])
    assert result.response_mode != 'memory_lookup'


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['branch_context', 'conversation_reference_context'])
async def test_other_context_channels_cannot_be_ignored_by_factual_shortcut(field):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return '结合当前上下文回答。'

    ctx = replace(saved(), **{field: '当前情境含未合并的变化。'})
    result = await generate_character_response(GenerationRequest(message='我住哪里？', character_context=ctx), generate)
    assert result.model_invoked and len(calls) == 1


@pytest.mark.asyncio
async def test_explicit_storage_query_stays_storage_only_with_recent_correction():
    async def generate(**kwargs):
        pytest.fail('Storage inventory requires no semantic generation')

    result = await generate_character_response(GenerationRequest(message='你保存了我的现居地吗？',
        character_context=saved(), history=({'role': 'user', 'content': '旧住址不对了。'},)), generate)
    assert not result.model_invoked and result.response_mode == 'memory_storage_status'
    assert '有你的现居地记录' in result.reply


@pytest.mark.asyncio
async def test_api_passes_the_same_effective_history_to_shortcut_and_generation(monkeypatch):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: '角色设定')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: pytest.fail('Private lookup needs no external RAG'))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '我会依据最新对话。'

    history = ({'role': 'user', 'content': '我现在住在西宁。'},)
    reply, used_rag, meta = await generate._generate_with_vllm(
        MessageRequest(message='我住哪里？'), None, runtime_config={'useKnowledgeBase': True},
        prepared_character_turn=SimpleNamespace(compiled=saved(), history=history), model_generate=model)
    # Ordinary generation need not emit RAG metadata; actual calls are the evidence.
    assert len(calls) == 1 and meta.get('modelInvoked') is not False and not used_rag
    assert reply == '我会依据最新对话。'
