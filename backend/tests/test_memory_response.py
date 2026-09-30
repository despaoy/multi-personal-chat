from dataclasses import replace

import pytest

from character.context_builder import compile_character_context
from character.memory_extractor import extract_memories
from character.models import CharacterContext, CharacterProfile, CompiledCharacterContext, MemoryItem, UserScope
from inference.generation_request import GenerationRequest, generate_character_response
from inference.memory_response import lookup_fields, render_memory_response


def packets(text):
    return tuple(MemoryItem(str(index), 'user_fact', fact.content, evidence=(fact.evidence,),
                            source_message_ids=('user-message',), memory_key=fact.memory_key)
                 for index, fact in enumerate(extract_memories(text)))


def context(items):
    return CompiledCharacterContext('', '', '参考', tuple(item.memory_id for item in items),
                                    memory_status='available', memory_packets=items)


@pytest.mark.parametrize('query,fields', [
    ('我来自哪里，目前住哪里？', ('origin', 'residence')),
    ('你还记得我的专业和工作地点吗？', ('major', 'workplace')),
    ('我的姓名是什么？', ('name',)),
    ('我的籍贯与现居地分别是哪里？', ('origin', 'residence')),
    ('请告诉我我现在住哪里？', ('residence',)),
])
def test_closed_query_grammar(query, fields):
    assert lookup_fields(query) == fields


@pytest.mark.parametrize('query', [
    '我朋友的专业是什么？', '你的名字是什么？', '我的专业和你的工作地点是什么？',
    '我的专业适合什么工作？', '我的籍贯和现居地有什么区别？',
    '我的名字是什么？再帮我写首诗。', '我现在住在大理，原来住哪里？',
    '我以前住哪里？', '我来自哪里，我朋友现在住哪里？', '假如我来自哪里？',
    '不要告诉我我的名字。', '我的专业是什么，能给我建议吗？', '我们住哪里？',
    '我的专业和工作地点是什么，还记得我们第一次见面吗？',
])
def test_non_lookup_requests_are_never_swallowed(query):
    assert lookup_fields(query) == ()


@pytest.mark.parametrize('origin,residence', [('泉州', '合肥'), ('阿坝', '银川'), ('汕尾', '南宁')])
def test_project_values_not_city_specific(origin, residence):
    result = render_memory_response('我来自哪里，目前住哪里？',
                                    context(packets(f'我来自{origin}，现在住在{residence}。')))
    assert result == f'我这里记着的是：你来自{origin}，现在住在{residence}。'


@pytest.mark.parametrize('statement,query,expected', [
    ('我叫林溪。', '我的名字是什么？', '你叫林溪'),
    ('我的专业是地质学。', '我的专业是什么？', '你的专业是地质学'),
    ('我叫林岳，专业是气象学。', '我的专业是什么？', '你的专业是气象学'),
    ('我是研二。', '我的年级是什么？', '你是研二'),
    ('我学的是天文学，现在在观测站工作。', '我的专业和工作地点是什么？',
     '你的专业是天文学，你在观测站工作'),
])
def test_other_slots_preserve_attribution(statement, query, expected):
    assert render_memory_response(query, context(packets(statement))) == f'我这里记着的是：{expected}。'


@pytest.mark.parametrize('update', [
    {'historical': True}, {'status': 'superseded'}, {'status': 'pending'},
    {'valid_to': '2026-01-01'}, {'confidence': .5}, {'evidence': ()},
    {'source_message_ids': ()}, {'memory_key': ''},
    {'evidence': ('我朋友住在大理。',)}, {'evidence': ('假如我住在大理。',)},
    {'evidence': ('我住在昆明。',)}, {'content': '用户自述：我只有周末住在大理'},
])
def test_unverifiable_packet_is_not_asserted(update):
    item = replace(packets('我住在大理。')[0], **update)
    assert render_memory_response('我住哪里？', context((item,))) is None


def test_missing_fields_conflicts_and_budget_exclusions_preserve_generation():
    first = packets('我住在大理。')[0]
    second = replace(packets('我住在昆明。')[0], memory_id='other')
    assert render_memory_response('我住哪里？', context((first, second))) is None
    assert render_memory_response('我来自哪里，目前住哪里？', context((first,))) is None
    assert render_memory_response('我住哪里？', replace(context((first,)), used_memory_ids=())) is None
    assert render_memory_response('我住哪里？', replace(context((first,)), memory_status='retrieval_error')) is None


def test_compilation_carries_only_actually_admitted_packets():
    items = packets('我来自阿坝，现在住在银川。') + (
        MemoryItem('oversized', 'user_fact', '很长' * 1000, memory_key='user_name'),)
    compiled = compile_character_context(CharacterContext(CharacterProfile('test', '角色'),
        UserScope('web', 'test', 'u', 'c', 'private'), memories=items))
    assert compiled.memory_packets
    assert {item.memory_id for item in compiled.memory_packets} == set(compiled.used_memory_ids)
    assert 'oversized' not in compiled.used_memory_ids


@pytest.mark.asyncio
async def test_wrong_assistant_history_cannot_replace_known_values_and_no_extra_call():
    async def model(**kwargs):
        pytest.fail('Structured read should need no model invocation')

    result = await generate_character_response(GenerationRequest(message='我来自哪里，目前住哪里？',
        character_context=context(packets('我来自阿坝，现在住在银川。')),
        history=({'role': 'assistant', 'content': '你现在住在杭州。'},)), model)
    assert '银川' in result.reply and '杭州' not in result.reply
    assert not result.model_invoked and result.response_mode == 'memory_lookup'


@pytest.mark.asyncio
async def test_correction_with_other_task_remains_generation():
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '按你的新情况讨论。'

    result = await generate_character_response(GenerationRequest(message='我现在住在西宁，适合徒步吗？',
        character_context=context(packets('我住在银川。'))), model)
    assert result.model_invoked and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('use_kb', [False, True])
async def test_api_reports_memory_read_as_zero_model_calls_not_rag_source(monkeypatch, use_kb):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    async def model(**kwargs):
        pytest.fail('No model invocation for a verified field read')

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: pytest.fail('Verified field read needs no external RAG'))
    prepared = SimpleNamespace(history=(), compiled=context(packets('我住在银川。')))
    reply, used_rag, meta = await generate._generate_with_vllm(
        MessageRequest(message='我现在住哪里？'), None, runtime_config={'useKnowledgeBase': use_kb},
        prepared_character_turn=prepared, model_generate=model)
    assert '银川' in reply and not used_rag
    assert meta['modelInvoked'] is False and meta['answerMode'] == 'memory_lookup'
    assert meta['abstained'] is False and meta['citations'] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('message,external', [('我来自哪里，目前住哪里？', False), ('我住哪里，那里有什么景点？', True)])
async def test_partial_or_mixed_read_routes_by_dependency(monkeypatch, message, external):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    routed = []

    def needs_rag(query):
        routed.append(query)
        return False, 'test', None

    async def model(**kwargs):
        return '普通生成路径'

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', needs_rag)
    reply, _, _ = await generate._generate_with_vllm(
        MessageRequest(message=message), None, runtime_config={'useKnowledgeBase': True},
        prepared_character_turn=SimpleNamespace(history=(), compiled=context(packets('我住在银川。'))),
        model_generate=model)
    # A missing private field still needs generation, not external knowledge.
    assert bool(routed) is external and reply == '普通生成路径'
