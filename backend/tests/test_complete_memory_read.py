from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_memory_response import context, packets

from inference.generation_request import GenerationRequest, generate_character_response
from inference.memory_response import render_complete_memory_read


def missing_context():
    return replace(context(()), memory_status='no_match',
                   memory_field_presence=(('major', False), ('name', False), ('residence', False),
                                          ('origin', False), ('workplace', False), ('study_stage', False)))


@pytest.mark.parametrize('query,label', [('我的专业是什么？', '专业'), ('我叫什么？', '姓名'),
    ('我来自哪里？', '来源地'), ('我住哪里？', '现居地'), ('我的工作地点是什么？', '工作地点'),
    ('我的年级是什么？', '年级')])
@pytest.mark.asyncio
async def test_proven_absence_reads_without_model(query, label):
    async def model(**_):
        pytest.fail('Closed absent read does not need generation')

    result = await generate_character_response(GenerationRequest(message=query, character_context=missing_context()), model)
    assert result.response_mode == 'memory_field_result' and not result.model_invoked
    assert label in result.reply and '你的' in result.reply
    assert '从未' not in result.reply and 'absent' not in result.reply


def test_partial_read_preserves_known_value_and_absent_state():
    compiled = replace(context(packets('我叫许澄。')), memory_status='available',
                       memory_field_presence=(('name', True), ('major', False)))
    reply = render_complete_memory_read('我的名字和专业是什么？', compiled, ())
    assert reply == '我这里记着的是：你叫许澄。我这里暂时没有你的专业信息。'


@pytest.mark.parametrize('change', [dict(memory_status='retrieval_error'), dict(memory_status='not_checked'),
    dict(memory_field_presence=()), dict(memory_field_presence=(('major', True),)), dict(branch_context='分支')])
def test_unresolved_or_branch_state_never_becomes_absence(change):
    assert render_complete_memory_read('我的专业是什么？', replace(missing_context(), **change), ()) is None


@pytest.mark.parametrize('history', [({'role': 'user', 'content': '我的专业是海洋科学。'},),
                                   ({'role': 'assistant', 'content': '我猜你学化学。'},)])
def test_no_interpretation_of_unsaved_history(history):
    assert render_complete_memory_read('我的专业是什么？', missing_context(), history) is None


@pytest.mark.parametrize('query', ['先说我的专业，再告诉我林远是谁。', '我的专业是园艺学，记住了。',
    '我的专业适合什么工作？', '我朋友的专业是什么？', '请翻译“我的专业是什么？”'])
def test_other_tasks_not_swallowed(query):
    assert render_complete_memory_read(query, missing_context(), ()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('use_kb', [False, True])
async def test_api_skips_external_retrieval_and_reports_zero_calls(monkeypatch, use_kb):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    async def model(**_):
        pytest.fail('Unexpected model call')

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: pytest.fail('No external dependency'))
    reply, rag, meta = await generate._generate_with_vllm(
        MessageRequest(message='我的专业是什么？'), None, runtime_config={'useKnowledgeBase': use_kb},
        prepared_character_turn=SimpleNamespace(history=(), compiled=missing_context()), model_generate=model)
    assert not rag and '你的专业' in reply
    assert meta['modelInvoked'] is False and meta['answerMode'] == 'memory_field_result'
    assert generate._DETERMINISTIC_MODEL_LABELS[meta['answerMode']] == 'memory/field_result'


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['我来自哪里？', '我的专业是什么？', '我住哪里？', '我叫什么？'])
@pytest.mark.parametrize('status', ['no_match', 'retrieval_error'])
async def test_personal_read_with_history_never_requires_external_knowledge(monkeypatch, query, status):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '这点还不能确定。'

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: pytest.fail('Personal read routed to external RAG'))
    history = ({'role': 'user', 'content': '说回我本人，我来自贵阳。'},)
    _, rag, meta = await generate._generate_with_vllm(
        MessageRequest(message=query), None, runtime_config={'useKnowledgeBase': True},
        prepared_character_turn=SimpleNamespace(history=history, compiled=replace(missing_context(), memory_status=status)),
        model_generate=model)
    assert len(calls) == 1 and not rag
    assert any(m['content'] == history[0]['content'] for m in calls[0]['messages'])


@pytest.mark.asyncio
@pytest.mark.parametrize('query,branch', [
    ('我的专业适合什么工作？', ''), ('我朋友的专业是什么？', ''),
    ('我的专业是什么，以及夜子的哥哥是谁？', ''), ('我来自哪里？', '分支场景'),
])
async def test_nonpersonal_dependencies_keep_rag_routing(monkeypatch, query, branch):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    routed = []

    def needs_rag(text):
        routed.append(text)
        return False, 0, None

    async def model(**_):
        return '还不能确定。'

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', needs_rag)
    await generate._generate_with_vllm(
        MessageRequest(message=query), None, runtime_config={'useKnowledgeBase': True},
        prepared_character_turn=SimpleNamespace(history=(), compiled=replace(missing_context(), branch_context=branch)),
        model_generate=model)
    assert len(routed) == 1
