import pytest

from character.memory_operation import is_only_memory_erasure, render_operation_response, split_operation_request
from character.models import CompiledCharacterContext
from inference.generation_request import GenerationRequest, generate_character_response


@pytest.mark.parametrize('message', [
    '请把你记住的我的住址彻底删掉。',
    '请把我的宠物名字从记忆里删除，也不要再引用那条原话。',
    '请从记忆里删除我的工作地点。',
    '忘掉我的专业。',
])
async def test_pure_operation_uses_actual_receipt_without_model(message):
    context = CompiledCharacterContext('', '', '', memory_operation_receipt={'status': 'erased', 'persisted': 1})

    async def must_not_call(**kwargs):
        pytest.fail('Execution receipt must not be reinterpreted by a model')

    result = await generate_character_response(GenerationRequest(message=message, character_context=context), must_not_call)
    assert not result.model_invoked and result.response_mode == 'memory_operation'
    assert '已删除' in result.reply and '聊天记录没有删除' in result.reply


@pytest.mark.parametrize('status,count', [('erased', 1), ('pending', 0), ('failed', 0)])
async def test_operation_receipt_does_not_require_a_model_context_budget(status, count):
    context = CompiledCharacterContext('人设' * 10000, '动态' * 10000, '参考' * 10000,
        memory_operation_receipt={'status': status, 'persisted': count})

    async def must_not_call(**kwargs):
        pytest.fail('No model is needed for an operation receipt')

    result = await generate_character_response(GenerationRequest(message='忘掉我的专业。',
        character_context=context, context_window_tokens=100, max_tokens=200), must_not_call)
    assert result.response_mode == 'memory_operation'
    assert not result.model_invoked
    assert result.plan.messages == ()
    assert ('已删除' in result.reply) is (status == 'erased')


@pytest.mark.parametrize('status', ['erased', 'pending', 'failed'])
async def test_api_operation_receipt_has_no_external_retrieval_dependency(monkeypatch, status):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    routed = []
    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda query: (routed.append(query) or (False, 0, None)))
    context = CompiledCharacterContext('', '', '',
        memory_operation_receipt={'status': status, 'persisted': 1 if status == 'erased' else 0})
    prepared = SimpleNamespace(history=(), compiled=context)

    async def must_not_call(**kwargs):
        pytest.fail('Operation receipt must not invoke answer model')

    _, used_rag, meta = await generate._generate_with_vllm(
        MessageRequest(message='请把我的宠物名字从记忆里删除。'), None,
        prepared_character_turn=prepared, runtime_config={'useKnowledgeBase': True}, model_generate=must_not_call)
    assert routed == []
    assert not used_rag and meta['answerMode'] == 'memory_operation'


@pytest.mark.parametrize('message', [
    '请把我的地址删掉，然后解释一下光合作用。',
    '忘掉我的专业，顺便告诉我如何学英语。',
    '忘掉我的地址并告诉我明天的天气。',
    '请把我的名字删掉。我的专业是什么？',
    '不要忘掉我的地址。',
    '如果要删除我的地址，怎么操作？',
    '“忘掉我的地址”是什么意思？',
])
def test_mixed_non_authorizing_or_ambiguous_requests_are_not_swallowed(message):
    assert not is_only_memory_erasure(message)
    assert render_operation_response(message, {'status': 'erased', 'persisted': 1}) is None


@pytest.mark.parametrize('status,count,fragment', [
    ('pending', 0, '还在处理中'), ('partial', 1, '尚未全部完成'),
    ('cancelled', 1, '最终结果还不能确认'), ('conflict', 0, '未完成'),
    ('not_scheduled', 0, '还没有执行'), ('failed', 0, '未能完成'),
    ('skipped', 0, '没有执行'), ('no_change', 0, '没有删除'),
    ('erased', 0, '尚未确认'), ('erased', True, '尚未确认'),
])
def test_only_confirmed_erasure_can_claim_success(status, count, fragment):
    reply = render_operation_response('忘掉我的地址。', {'status': status, 'persisted': count})
    assert fragment in reply
    assert '已删除' not in reply


def test_intent_without_execution_receipt_is_not_success():
    assert render_operation_response('忘掉我的地址。', None) is None


async def test_mixed_task_preserves_remaining_question_without_reinterpreting_receipt():
    message = '请把你记住的我的地址删掉，同时解释一下为什么会有四季。'
    context = CompiledCharacterContext('', '运行回执', '',
        memory_operation_receipt={'status': 'erased', 'persisted': 1})
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return '删除已经完成。地轴倾斜以及地球公转带来了季节变化。'

    result = await generate_character_response(GenerationRequest(message=message, character_context=context), generate)
    assert result.model_invoked and len(calls) == 1
    assert '解释一下为什么会有四季。' in calls[0]['messages'][-1]['content']
    assert '请把你记住的我的地址删掉' not in calls[0]['messages'][-1]['content']
    assert result.reply.startswith('已删除匹配的长期记忆')
    assert result.response_mode == 'task_composite'
    assert '季节变化' in result.reply


@pytest.mark.parametrize('message,remaining', [
    ('请把你记住的我的地址删掉，同时解释四季。', '解释四季。'),
    ('忘掉我的专业。请翻译 hello，并解释用法。', '请翻译 hello，并解释用法。'),
    ('请把我的资料从记忆里删除，然后计算 12+8。', '计算 12+8。'),
])
def test_split_keeps_all_remaining_tasks(message, remaining):
    split = split_operation_request(message)
    assert split is not None and split[1] == remaining


@pytest.mark.parametrize('message', [
    '请把我的地址删掉，然后告诉我删掉了什么。',
    '请把我的地址删掉，然后解释这次结果。',
    '“请把我的地址删掉，同时解释四季”是什么意思？',
    '如果删除成功后，再解释四季。',
    '不要忘掉我的地址，同时解释四季。',
])
def test_dependent_quoted_or_unauthorized_requests_stay_intact(message):
    assert split_operation_request(message) is None


@pytest.mark.parametrize('status,fragment', [('pending', '还在处理中'), ('failed', '未能完成'), ('partial', '尚未全部完成')])
async def test_non_success_operation_does_not_block_independent_task(status, fragment):
    from character.memory_operation import operation_receipt_context

    receipt = {'status': status, 'persisted': 0}
    context = CompiledCharacterContext('', operation_receipt_context(receipt), '', memory_operation_receipt=receipt)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return '20。'

    result = await generate_character_response(GenerationRequest(
        message='请把你记住的我的地址删掉，同时计算12+8。', character_context=context), generate)
    assert fragment in result.reply and result.reply.endswith('20。')
    assert len(calls) == 1
    assert '操作执行回执' not in calls[0]['messages'][0]['content']
