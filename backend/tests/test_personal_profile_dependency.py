"""Complete personal profile requests have no external knowledge dependency."""
from types import SimpleNamespace

import pytest

from character.memory_query import plan_memory_query
from character.models import CompiledCharacterContext, MemoryItem
from db.schemas import MessageRequest
from knowledge.intent_detector import RAGIntentDetector
from knowledge.task_dependency import local_context_only

QUERY = '请用一句话核对我的个人资料：姓名、居住地和大学专业各是什么？只写我本人的当前资料，不做建议。'

@pytest.mark.parametrize('query,fields', [
    (QUERY, ('name', 'residence', 'major')),
    ('请用一句话核对我的当前个人资料：姓名、籍贯、现居地和大学专业各是什么？只写当前本人资料，不做建议。', ('name', 'origin', 'residence', 'major')),
    ('请列出我的基本资料。', ('name', 'origin', 'residence', 'major', 'workplace', 'study_stage')),
    ('用表格整理我的个人信息：姓名和工作单位分别是什么？', ('name', 'workplace')),
    ('回忆一下我的个人资料：专业和年级是什么？', ('major', 'study_stage')),
    ('请用列表说明我的基本信息：故乡、住址与学习阶段各是什么？不要建议。', ('origin', 'residence', 'study_stage')),
])
def test_complete_profile_dependency_and_field_coverage(query, fields):
    assert local_context_only(query)
    assert RAGIntentDetector().needs_rag(query)[0] is False
    assert plan_memory_query(query).fields == fields

@pytest.mark.parametrize('query', [
    QUERY + '请介绍环境工程课程。',
    QUERY.replace('不做建议', '请推荐专业教材'),
    '请用一句话核对我的个人资料：姓名和月社妃的哥哥各是什么？',
    '请整理我的个人资料：姓名和专业课程大纲是什么？',
    '请介绍我的专业教材。',
    '请用一句话核对他的个人资料：姓名、现居地和专业各是什么？',
    '请列出我的基本资料以及月社妃的哥哥是谁。',
    '请列出我的基本资料。原文依据在哪里？',
    '请列出我的基本资料，不做建议是什么意思？',
    '请列出我的基本资料，请不要分析量子力学。',
])
def test_unknown_foreign_and_external_profile_clauses_keep_retrieval(query):
    assert not local_context_only(query)
    assert RAGIntentDetector().needs_rag(query)[0] is True

@pytest.mark.asyncio
@pytest.mark.parametrize('saved', [True, False])
async def test_profile_routes_to_actual_generation_with_or_without_saved_records(monkeypatch, saved):
    from api import generate
    from knowledge import intent_detector

    async def retrieval(*args):
        pytest.fail('External knowledge cannot establish private user fields')

    calls = []
    async def model(**kwargs):
        calls.append(kwargs)
        return '你叫岚舟，目前住在舟山，大学专业是环境工程。' if saved else '当前没有可核对的个人资料。'

    monkeypatch.setattr(intent_detector, '_ML_AVAILABLE', False)
    monkeypatch.setattr(intent_detector, '_load_ml_model', lambda: pytest.fail('Personal dependency needs no classifier'))
    monkeypatch.setattr(generate, '_retrieve_rag_bundle', retrieval)
    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    item = MemoryItem('profile', 'user_fact', '用户说自己叫岚舟', memory_key='user_name',
                      evidence=('我叫岚舟。',), source_message_ids=('source',))
    ctx = CompiledCharacterContext('', '', '我叫岚舟。我目前住在舟山。我的大学专业是环境工程。' if saved else '',
        ('profile',) if saved else (), memory_status='available' if saved else 'no_match',
        memory_packets=(item,) if saved else (), memory_field_presence=tuple((f, saved) for f in ('name', 'residence', 'major')))
    prepared = SimpleNamespace(compiled=ctx, history=({'role':'user','content':'今天整理了资料。'},), reply_guard=None)
    reply, used_rag, meta = await generate._generate_with_vllm(MessageRequest(message=QUERY), None,
        runtime_config={'useKnowledgeBase':True}, prepared_character_turn=prepared, model_generate=model)
    assert reply and len(calls) == 1 and used_rag is False
    assert not meta.get('abstained') and meta.get('answerMode') != 'abstention'
    assert QUERY in calls[0]['messages'][-1]['content']
    assert '【本轮证据不足】' not in calls[0]['messages'][0]['content']

@pytest.mark.asyncio
async def test_mixed_profile_keeps_actual_external_retrieval_and_uncertainty(monkeypatch):
    from api import generate
    from knowledge import intent_detector

    retrieved, calls = [], []
    async def retrieval(*args):
        retrieved.append(args)
        return {'context_text':'untrusted candidate', 'citations':[{'id':'rejected'}], 'abstained':True}
    async def model(**kwargs):
        calls.append(kwargs)
        return '你叫岚舟，原作人物的关系暂时无法核实。'
    monkeypatch.setattr(intent_detector, '_ML_AVAILABLE', False)
    monkeypatch.setattr(intent_detector, '_load_ml_model', lambda: None)
    monkeypatch.setattr(generate, '_retrieve_rag_bundle', retrieval)
    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    query=QUERY+'月社妃和琉璃是什么关系？'
    _, used_rag, meta=await generate._generate_with_vllm(MessageRequest(message=query),None,
        runtime_config={'useKnowledgeBase':True},model_generate=model)
    assert len(retrieved)==len(calls)==1 and used_rag and meta['abstained']
    assert meta['answerMode']=='abstention' and not meta['citations']
    assert query in calls[0]['messages'][-1]['content'] and 'untrusted candidate' not in str(calls)
