from dataclasses import replace

import pytest

from character.memory_extractor import extract_memories
from character.models import CompiledCharacterContext, MemoryItem
from inference.constraint_response import render_constraint_response
from inference.generation_request import GenerationRequest, generate_character_response


def context(condition='休息好', action='开长途车'):
    fact, = extract_memories(f'我只有{condition}才{action}。')
    item = MemoryItem('rule', 'user_fact', fact.content, memory_key=fact.memory_key,
                      evidence=(fact.evidence,), qualifiers=fact.qualifiers,
                      source_message_ids=('source',), confidence=.9)
    return CompiledCharacterContext('', '', '条件原文', ('rule',),
                                    memory_status='available', memory_packets=(item,))


@pytest.mark.parametrize('condition,action', [('休息好', '开长途车'), ('完成工作', '看电影'),
                                            ('放假', '去游泳'), ('得到授权', '分享照片')])
def test_general_actions_do_not_invert_necessary_condition(condition, action):
    compiled = context(condition, action)
    reply = render_constraint_response(f'如果{condition}，我就一定{action}吗？', compiled)
    assert reply.startswith('不能仅凭') and condition in reply and action in reply
    assert '不代表行动一定发生' in reply
    assert '必要条件' in render_constraint_response(f'我{action}需要什么条件？', compiled)


@pytest.mark.parametrize('query', ['如果休息好，我就一定开长途车吗？请再规划路线。',
                                 '她开长途车需要什么条件？', '假设我开长途车需要什么条件？',
                                 '不要告诉我开长途车需要什么条件。', '我今天会开长途车吗？',
                                 '如果天气晴，我就一定开长途车吗？', '我开车需要什么条件？',
                                 '我开长途车有什么限制？再写首诗。'])
def test_no_implicit_atom_translation_or_mixed_task_swallowing(query):
    assert render_constraint_response(query, context()) is None


@pytest.mark.parametrize('updates', [{'status': 'pending'}, {'historical': True},
                                    {'source_message_ids': ()}, {'confidence': .2},
                                    {'evidence': ('假设我只有休息好才开长途车。',)},
                                    {'qualifiers': (('kind', 'necessary_condition'),
                                                    ('action', '开长途车'), ('condition', '有空'))}])
def test_rule_requires_actual_current_evidence(updates):
    compiled = context()
    compiled = replace(compiled, memory_packets=(replace(compiled.memory_packets[0], **updates),))
    assert render_constraint_response('我开长途车需要什么条件？', compiled) is None


def test_conflicts_and_budget_omissions_are_not_silently_certified():
    compiled = context()
    other = replace(context('有空').memory_packets[0], memory_id='other')
    conflicting = replace(compiled, memory_packets=(*compiled.memory_packets, other),
                           used_memory_ids=('rule', 'other'))
    assert render_constraint_response('如果休息好，我就一定开长途车吗？', conflicting) is None
    assert render_constraint_response('我开长途车需要什么条件？',
                                      replace(compiled, used_memory_ids=())) is None


def test_legacy_context_and_branch_keep_normal_generation():
    from types import SimpleNamespace

    query = '我开长途车需要什么条件？'
    assert render_constraint_response(query, SimpleNamespace(reference_context='旧上下文')) is None
    assert render_constraint_response(query, replace(context(), branch_context='当前另有假设')) is None


@pytest.mark.parametrize('action', ['自杀', '自残', '伤害自己'])
def test_logic_shortcut_never_replaces_safety_handling(action):
    compiled = context('失业', action)
    assert render_constraint_response(f'我{action}需要什么条件？', compiled) is None
    assert render_constraint_response(f'如果失业，我就一定{action}吗？', compiled) is None


@pytest.mark.asyncio
async def test_shared_generation_executes_logic_without_model():
    async def model(**kwargs):
        pytest.fail('Necessary-condition read must not invoke generation')

    result = await generate_character_response(GenerationRequest(
        message='仅凭休息好就能确定我开长途车吗？', character_context=context()), model)
    assert not result.model_invoked
    assert result.reply.startswith('不能仅凭')


@pytest.mark.asyncio
async def test_condition_read_uses_no_external_retrieval(monkeypatch):
    from types import SimpleNamespace

    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector

    async def model(**kwargs):
        pytest.fail('No generation needed')

    monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
    monkeypatch.setattr(intent_detector, 'needs_rag', lambda _: pytest.fail('No external RAG needed'))
    reply, rag, meta = await generate._generate_with_vllm(
        MessageRequest(message='如果休息好，我就一定开长途车吗？'), None,
        runtime_config={'useKnowledgeBase': True}, model_generate=model,
        prepared_character_turn=SimpleNamespace(history=(), compiled=context()))
    assert reply.startswith('不能仅凭') and not rag and meta['modelInvoked'] is False


@pytest.mark.asyncio
async def test_persisted_constraint_is_executable_in_new_session(tmp_path, monkeypatch):
    from character.context_builder import build_user_scope
    from character.memory_service import CharacterMemoryService
    from character.rule_memory_writer import write_rule_memory
    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from services.character_context import TurnInput, build_character_context_service

    monkeypatch.setenv('CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'false')
    db = SQLiteDB(tmp_path / 'conditions.sqlite')
    repo = DatabaseCharacterMemoryRepository(db)
    scope = build_user_scope('web', 'test', 'user', 'old', 'private')
    fact, = extract_memories('我只有检查完成才使用设备。')
    await write_rule_memory(repo, 'tsukiyashiro_kisaki', scope, fact, 'source')
    service = build_character_context_service(db)
    service._memory_service = CharacterMemoryService(repo, semantic_enabled=False)
    query = '如果检查完成，我就一定使用设备吗？'
    prepared = await service.prepare_turn(TurnInput(query, 'web', 'test', 'user', 'new', 'private'),
                                          'tsukiyashiro_kisaki')
    assert prepared.history == ()
    assert render_constraint_response(query, prepared.compiled).startswith('不能仅凭')
