from dataclasses import replace
from types import SimpleNamespace

import pytest

from character.models import CompiledCharacterContext, MemoryItem
from inference.generation_request import GenerationRequest, RetrievalResult, generate_character_response
from inference.task_execution import prepare_independent_tasks


def request(message='不要分析，先说我的专业，再告诉我林远是谁。'):
    item = MemoryItem('one', 'user_fact', '用户说自己的专业是地质学',
                      evidence=('我的专业是地质学。',), source_message_ids=('source',), memory_key='user_major')
    context = CompiledCharacterContext('', '', '参考', ('one',),
                                       memory_status='available', memory_packets=(item,))
    retrieval = RetrievalResult(status='ok', evidence='林远是学生。',
                                identity_subtask={'query': message, 'subject': '林远'})
    return GenerationRequest(message=message, character_context=context, retrieval=retrieval,
                             independent_tasks_enabled=True,
                             history=({'role': 'assistant', 'content': '旧回复'},))


@pytest.mark.asyncio
async def test_default_keeps_full_request_not_experimental_task_isolation():
    assert GenerationRequest(message='任意问题').independent_tasks_enabled is False
    original = replace(request(), independent_tasks_enabled=False)
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        assert original.message in kwargs['messages'][-1]['content']
        return '你的专业是地质学。林远是学生。'

    result = await generate_character_response(original, model)
    assert len(calls) == 1 and result.response_mode == 'generated'
    assert result.plan.history_policy == 'conversation' and not result.task_results


@pytest.mark.asyncio
@pytest.mark.parametrize('message,memory_first', [
    ('不要分析，先说我的专业，再告诉我林远是谁。', True),
    ('告诉我林远是谁，再说我的专业。', False),
])
async def test_independent_tasks_preserve_order_and_use_one_model_call(message, memory_first):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        assert '林远是谁' in kwargs['messages'][-1]['content']
        if '不要分析' in message:
            assert '不要分析' in kwargs['messages'][-1]['content']
        return '林远是学生。'

    original = request(message)
    result = await generate_character_response(original, model)
    assert len(calls) == 1
    assert result.response_mode == 'task_composite' and result.model_invoked
    assert '你的专业是地质学' in result.reply
    assert result.reply.startswith('我这里记着') is memory_first
    assert len(result.task_results) == 2
    assert original.message == message and original.history[0]['content'] == '旧回复'
    assert result.plan.history_policy == 'independent_identity'


@pytest.mark.parametrize('change', ['missing_memory', 'unbound', 'no_evidence', 'branch', 'strict', 'mixed'])
def test_unproven_independence_or_memory_does_not_enable_execution(change):
    original = request()
    if change == 'missing_memory':
        original = replace(original, character_context=None)
    elif change == 'unbound':
        original = replace(original, message='其他问题')
    elif change == 'no_evidence':
        original = replace(original, retrieval=replace(original.retrieval, evidence=''))
    elif change == 'branch':
        original = replace(original, character_context=replace(original.character_context, branch_context='假设'))
    elif change == 'strict':
        original = replace(original, reply_guard_mode='strict')
    else:
        original = request('先说我的专业，再告诉我林远是谁，再比较两人的经历。')
    assert prepare_independent_tasks(original) is None


@pytest.mark.parametrize('flag', ['require_gentle_safety_check', 'require_urgent_safety_check', 'third_party_safety'])
def test_safety_obligations_are_not_split(flag):
    assert prepare_independent_tasks(replace(request(), reply_guard=SimpleNamespace(**{flag: True}))) is None


def test_one_unresolved_memory_task_prevents_partial_execution():
    assert prepare_independent_tasks(request('先说我的专业，再说我的名字，再告诉我林远是谁。')) is None


def test_source_lookup_and_quoted_tasks_remain_on_original_path():
    original = request()
    assert prepare_independent_tasks(replace(original, retrieval=replace(original.retrieval, source_lookup=True))) is None
    assert prepare_independent_tasks(request('请翻译“先说我的专业，再告诉我林远是谁。”')) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('admit', [False, True])
async def test_composition_uses_only_final_budget_evidence(admit):
    original = request()
    short = {'text': '林远是学生。', 'document_ids': ('kept',)}
    large = {'text': '不应入模的证据' * 100, 'document_ids': ('rejected',)}
    original = replace(original, evidence_max_chars=40, retrieval=replace(
        original.retrieval, evidence=large['text'],
        evidence_packets=(large, short) if admit else (large,),
        citations=({'id': 'kept'}, {'id': 'rejected'})))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        assert '不应入模的证据' not in str(kwargs['messages'])
        return '林远是学生。' if admit else '人物资料暂时不足。'

    result = await generate_character_response(original, model)
    assert len(calls) == 1
    assert (result.response_mode == 'task_composite') is admit
    assert result.plan.retrieval.citations == (({'id': 'kept'},) if admit else ())
    if not admit:
        assert original.message in calls[0]['messages'][-1]['content']
        assert result.plan.retrieval.reason == 'evidence_budget_exhausted'
        assert not result.task_results


@pytest.mark.parametrize('state', ['unadmitted', 'conflicting', 'historical', 'retrieval_error'])
def test_unusable_memory_keeps_original_task(state):
    original = request()
    context = original.character_context
    item = context.memory_packets[0]
    if state == 'unadmitted':
        context = replace(context, used_memory_ids=())
    elif state == 'historical':
        context = replace(context, memory_packets=(replace(item, historical=True),))
    elif state == 'retrieval_error':
        context = replace(context, memory_status='retrieval_error')
    else:
        other = replace(item, memory_id='two', content='用户说自己的专业是历史学',
                        evidence=('我的专业是历史学。',))
        context = replace(context, used_memory_ids=('one', 'two'), memory_packets=(item, other))
    assert prepare_independent_tasks(replace(original, character_context=context)) is None


@pytest.mark.asyncio
async def test_multiple_complete_memory_tasks_keep_order():
    original = request('先说我的名字，再告诉我林远是谁，最后说我的专业。')
    name = MemoryItem('name', 'user_fact', '用户说自己叫周宁', evidence=('我叫周宁。',),
                      source_message_ids=('name-source',), memory_key='user_name')
    original = replace(original, character_context=replace(
        original.character_context, used_memory_ids=('one', 'name'),
        memory_packets=(*original.character_context.memory_packets, name)))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '林远是学生。'

    result = await generate_character_response(original, model)
    assert len(calls) == 1 and result.response_mode == 'task_composite'
    assert result.reply.index('你叫周宁') < result.reply.index('林远是学生') < result.reply.index('你的专业是地质学')
    assert [task['mode'] for task in result.task_results] == ['memory_lookup', 'generated', 'memory_lookup']
