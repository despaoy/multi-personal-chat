"""Explicit additive evidence must survive versioning and cross-session reads."""
import json
from dataclasses import replace

import pytest

from character.conditional_memory import verified_constraint_atoms
from character.context_builder import build_user_scope
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from inference.constraint_response import render_constraint_response
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import TurnInput, build_character_context_service


@pytest.mark.asyncio
@pytest.mark.parametrize('action,first,second', [
    ('预约会议', '收到确认', '所有人有空'),
    ('启动仪器', '设备检查完成', '获得授权'),
    ('去游泳', '周末', '天气合适'),
])
async def test_add_read_repeat_correct_across_sessions(tmp_path, monkeypatch, action, first, second):
    monkeypatch.setenv('CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'false')
    db = SQLiteDB(tmp_path / 'additive.sqlite')
    repo = DatabaseCharacterMemoryRepository(db)
    scope = build_user_scope('web', 'test', 'u1', 'old', 'private')
    texts = [f'我只有{first}才{action}。', f'补充一下，我只有{second}才{action}。']
    for index, text in enumerate(texts):
        item, = extract_memories(text)
        assert await write_rule_memory(repo, 'tsukiyashiro_kisaki', scope, item, f'm{index}')
    rows = await repo.list_memory_records('tsukiyashiro_kisaki', scope, limit=None, include_inactive=True)
    assert sorted(r['status'] for r in rows) == ['active', 'superseded']
    active, = [r for r in rows if r['status'] == 'active']
    assert active['evidence'] == texts and active['source_message_ids'] == ['m0', 'm1']
    assert verified_constraint_atoms(active['metadata']['qualifiers'], active['content'], active['evidence']) == ((first, second), action)
    for text in (texts[1], f'我只有{first}才{action}。'):
        item, = extract_memories(text)
        assert not await write_rule_memory(repo, 'tsukiyashiro_kisaki', scope, item, 'repeat')
    service = build_character_context_service(db)
    service._memory_service = CharacterMemoryService(repo, semantic_enabled=False)
    for query in (f'我{action}需要什么条件？', f'如果{first}，我就一定{action}吗？'):
        prepared = await service.prepare_turn(TurnInput(query, 'web', 'test', 'u1', 'new', 'private'), 'tsukiyashiro_kisaki')
        reply = render_constraint_response(query, prepared.compiled)
        assert first in reply and second in reply and '不表示行动一定发生' in reply
        from api import generate
        from db.schemas import MessageRequest

        async def forbidden(*args, **kwargs):
            pytest.fail('Structured additive reads require neither RAG nor generation')

        monkeypatch.setattr(generate, '_retrieve_rag_bundle', forbidden)
        monkeypatch.setattr(generate, '_get_system_prompt', lambda _: 'persona')
        delivered, _, meta = await generate._generate_with_vllm(
            MessageRequest(message=query), None, runtime_config={'useKnowledgeBase': True},
            model_generate=forbidden, prepared_character_turn=prepared)
        assert delivered == reply and meta['modelInvoked'] is False
    correction, = extract_memories(f'更正一下，我只有取得新许可才{action}。')
    assert await write_rule_memory(repo, 'tsukiyashiro_kisaki', scope, correction, 'correction')
    prepared = await service.prepare_turn(TurnInput(f'我{action}需要什么条件？', 'web', 'test', 'u1', 'third', 'private'), 'tsukiyashiro_kisaki')
    reply = render_constraint_response(f'我{action}需要什么条件？', prepared.compiled)
    assert '取得新许可' in reply and first not in reply and second not in reply
    other = await service.prepare_turn(TurnInput(f'我{action}需要什么条件？', 'web', 'test', 'u2', 'new', 'private'), 'tsukiyashiro_kisaki')
    assert not other.compiled.memory_packets


@pytest.mark.asyncio
async def test_third_addition_then_replace_with_existing_member(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'three.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    texts = ['我只有周末才去游泳。', '补充，我只有天气合适才去游泳。',
             '补充，我只有有空才去游泳。', '更正一下，我只有周末才去游泳。']
    for i, text in enumerate(texts):
        item, = extract_memories(text)
        assert await write_rule_memory(repo, 'c', scope, item, str(i))
    rows = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    assert sorted(r['status'] for r in rows) == ['active', 'superseded', 'superseded', 'superseded']
    active, = [r for r in rows if r['status'] == 'active']
    assert active['source_message_ids'] == ['3']
    assert verified_constraint_atoms(active['metadata']['qualifiers'], active['content'], active['evidence']) == (('周末',), '去游泳')


@pytest.mark.asyncio
async def test_forged_append_operation_cannot_merge(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'forged.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    old, = extract_memories('我只有周末才去游泳。')
    new, = extract_memories('我只有天气合适才去游泳。')
    await write_rule_memory(repo, 'c', scope, old, 'old')
    await write_rule_memory(repo, 'c', scope, replace(new, operation='append'), 'new')
    rows = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    assert sorted(r['status'] for r in rows) == ['active', 'pending']


@pytest.mark.parametrize('text', ['假设补充，我只有周末才去游泳。',
    '不要记住：补充，我只有周末才去游泳。', '补充，我只有周末才去游泳吗？',
    '她说补充，我只有周末才去游泳。'])
def test_addition_preserves_assertion_gates(text):
    assert not extract_memories(text)


def test_set_verification_requires_explicit_chain_not_just_two_sources():
    from character.conditional_memory import constraint_set_content

    content = constraint_set_content('去游泳', ('周末', '天气合适'))
    qualifiers = dict(kind='necessary_condition_set', action='去游泳',
                      conditions=json.dumps(('周末', '天气合适'), ensure_ascii=False), context=content)
    base = '我只有周末才去游泳。'
    assert verified_constraint_atoms(qualifiers, content, [base, '我只有天气合适才去游泳。']) is None
    assert verified_constraint_atoms(qualifiers, content, [base, '补充，我只有天气合适才去游泳。'])
    assert verified_constraint_atoms({**qualifiers, 'exception': '假期'}, content, [base]) is None
