"""Conditional assertions must survive the storage/read boundary without inversion."""
import pytest

from character.conditional_memory import parse_necessary_condition
from character.memory_extractor import extract_memories
from character.memory_service import CharacterMemoryService, _row_qualifiers
from character.models import UserScope
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.parametrize('change', [
    {'condition': '放假'}, {'action': '看电影'}, {'kind': 'sufficient_condition'},
    {'context': '假设我只有周末才去游泳。'}, {'exception': '今天'},
    {'context': None},
])
def test_condition_identity_does_not_ignore_unknown_or_inconsistent_qualifiers(change):
    from character.conditional_memory import same_necessary_condition

    original = dict(parse_necessary_condition('我只有周末才去游泳。').qualifiers('我只有周末才去游泳。'))
    assert not same_necessary_condition(original, {**original, **change})
    assert not same_necessary_condition({**original, **change}, original)


def test_semantically_identical_correction_does_not_shadow_saved_fact():
    from character.current_turn_memory import shadowed_memory_ids

    item, = extract_memories('我只有周末才去游泳。')
    row = dict(id=1, memory_key=item.memory_key, content=item.content, status='active',
               metadata={'origin': 'rule_v2', 'qualifiers': dict(item.qualifiers)})
    assert shadowed_memory_ids('更正一下，我只有周末才去游泳。', [row]) == set()


@pytest.mark.parametrize('condition,action', [
    ('周末', '去游泳'), ('休息好', '开长途车'), ('吃过饭', '喝咖啡'),
    ('有空', '看电影'), ('完成工作', '打游戏'), ('你同意', '分享照片'),
])
def test_necessary_rules_preserve_atoms_and_evidence(condition, action):
    text = f'我只有{condition}才{action}。'
    rule = parse_necessary_condition(text)
    item, = extract_memories(text)
    assert rule.action == action and rule.condition == condition
    assert item.evidence == text and item.content == '用户自述：' + text
    assert dict(item.qualifiers) == {
        'kind': 'necessary_condition', 'condition': condition, 'action': action, 'context': text}
    assert '今天' not in item.content


def test_same_turn_competing_conditions_reach_the_version_writer():
    items = extract_memories('我只有周末才去游泳。我只有放假才去游泳。')
    assert len(items) == 2 and items[0].memory_key == items[1].memory_key
    assert [dict(item.qualifiers)['condition'] for item in items] == ['周末', '放假']


@pytest.mark.parametrize('text', [
    '我只有周末才去游泳？', '我只有周末才去游泳吗',
    '她只有周末才去游泳。', '假设我只有周末才去游泳。',
    '我朋友说我只有周末才去游泳。', '不要记住：我只有周末才去游泳。',
    '我只有今天才去游泳。', '我只有周末才可能去游泳。',
    '我只有周末才去游泳，但今天例外。', '我只有周末才她去游泳。',
])
def test_unsupported_or_nonasserted_rules_do_not_persist(text):
    assert not extract_memories(text)


@pytest.mark.asyncio
async def test_constraint_roundtrip_and_condition_change_is_not_silent(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'conditions.sqlite'))
    scope = UserScope('web', 'test', 'u1', 's1', 'private')
    for i, text in enumerate(['我只有周末才去游泳。', '我只有放假才去游泳。']):
        item, = extract_memories(text)
        await write_rule_memory(repo, 'kisaki', scope, item, f'm{i}')
    rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
    assert sorted(row['status'] for row in rows) == ['active', 'pending']
    items, _ = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
        'kisaki', scope, '我的限制有哪些？')
    assert len(items) == 1
    assert dict(items[0].qualifiers)['condition'] == '周末'
    assert items[0].source_message_ids == ('m0',)
    assert '我只有周末才去游泳。' in items[0].evidence


@pytest.mark.asyncio
@pytest.mark.parametrize('first,repeat', [
    ('我只有周末才去游泳。', '我只有周末才去游泳'),
    ('更正一下，我只有完成工作才看电影。', '我只有完成工作才看电影。'),
    ('我只有收到确认才预约会议', '我只有收到确认才预约会议。'),
])
async def test_constraint_repetition_does_not_create_false_conflict(tmp_path, first, repeat):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'repeat.sqlite'))
    scope = UserScope('web', 'test', 'u1', 's1', 'private')
    original, = extract_memories(first)
    repeated, = extract_memories(repeat)
    assert await write_rule_memory(repo, 'kisaki', scope, original, 'first')
    assert not await write_rule_memory(repo, 'kisaki', scope, repeated, 'repeat')
    rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
    assert len(rows) == 1 and rows[0]['status'] == 'active'
    assert rows[0]['evidence'] == [first]
    assert rows[0]['source_message_ids'] == ['first']


@pytest.mark.asyncio
async def test_repeated_pending_constraint_is_not_added_or_promoted(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'pending-repeat.sqlite'))
    scope = UserScope('web', 'test', 'u1', 's1', 'private')
    for text in ['我只有周末才去游泳。', '我只有放假才去游泳。']:
        item, = extract_memories(text)
        assert await write_rule_memory(repo, 'kisaki', scope, item, text)
    repeat, = extract_memories('我只有放假才去游泳')
    assert not await write_rule_memory(repo, 'kisaki', scope, repeat, 'repeat')
    rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
    assert sorted(row['status'] for row in rows) == ['active', 'pending']
    assert next(r for r in rows if r['status'] == 'active')['metadata']['qualifiers']['condition'] == '周末'


@pytest.mark.asyncio
async def test_condition_packet_survives_new_private_session_preparation(tmp_path, monkeypatch):
    from character.context_builder import build_user_scope
    from services.character_context import TurnInput, build_character_context_service

    monkeypatch.setenv('CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'false')
    database = SQLiteDB(tmp_path / 'cross-session.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = build_user_scope('web', 'test', 'u1', 'old-session', 'private')
    item, = extract_memories('我只有休息好才开长途车。')
    await write_rule_memory(repo, 'tsukiyashiro_kisaki', scope, item, 'source-constraint')
    service = build_character_context_service(database)
    service._memory_service = CharacterMemoryService(repo, semantic_enabled=False)
    prepared = await service.prepare_turn(
        TurnInput('我的限制有哪些？', 'web', 'test', 'u1', 'new-session', 'private'),
        'tsukiyashiro_kisaki')
    packet, = prepared.compiled.memory_packets
    assert packet.memory_id in prepared.compiled.used_memory_ids
    assert dict(packet.qualifiers)['kind'] == 'necessary_condition'
    assert dict(packet.qualifiers)['condition'] == '休息好'
    assert prepared.history == ()
    other = await service.prepare_turn(
        TurnInput('我的限制有哪些？', 'web', 'test', 'u2', 'new-session', 'private'),
        'tsukiyashiro_kisaki')
    assert not other.compiled.memory_packets


@pytest.mark.parametrize('metadata', [None, [], {'qualifiers': []},
    {'qualifiers': {'condition': '周末', 'action': None}}])
def test_invalid_qualifiers_are_not_partly_repaired(metadata):
    assert _row_qualifiers({'metadata': metadata}) == ()


@pytest.mark.parametrize('text', [
    '假设更正一下，我只有放假才去游泳。',
    '她说更正一下，我只有放假才去游泳。',
    '更正一下，我只有放假才去游泳？',
    '更正一下，我只有放假才去游泳，但今天例外。',
    '不要保存：更正一下，我只有放假才去游泳。',
    '“更正一下，我只有放假才去游泳。”',
    '更正一下，我只有今天才去游泳。',
])
def test_correction_does_not_bypass_assertion_gates(text):
    assert not extract_memories(text)


@pytest.mark.asyncio
@pytest.mark.parametrize('old,new,action', [
    ('周末', '放假', '去游泳'), ('吃过饭', '午饭后', '喝咖啡'),
    ('休息好', '睡够八小时', '开长途车'), ('有空', '完成工作', '看电影'),
])
async def test_explicit_correction_replaces_only_current_same_action(tmp_path, old, new, action):
    from character.context_builder import build_user_scope
    from character.memory_mentions import review_mentions

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'correction.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    texts = [f'我只有{old}才{action}。', f'更正一下，我只有{new}才{action}。']
    for i, text in enumerate(texts):
        item, = extract_memories(text)
        assert await write_rule_memory(repo, 'kisaki', scope, item, f'm{i}')
    repeat, = extract_memories(f'纠正一下：我只有{new}才{action}。')
    assert not await write_rule_memory(repo, 'kisaki', scope, repeat, 'repeat')
    rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
    assert sorted(r['status'] for r in rows) == ['active', 'superseded']
    active, = [r for r in rows if r['status'] == 'active']
    previous, = [r for r in rows if r['status'] == 'superseded']
    assert active['supersedes_memory_id'] == previous['id']
    assert active['source_message_ids'] == ['m1']
    assert previous['source_message_ids'] == ['m0']
    assert active['metadata']['operation'] == 'replace'
    fresh = build_user_scope('web', 'test', 'u1', 's2', 'private')
    memories, _ = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
        'kisaki', fresh, '我的限制有哪些？')
    assert len(memories) == 1 and dict(memories[0].qualifiers)['condition'] == new
    review = review_mentions('我提过哪些限制？', rows, complete_read=True)
    assert '[旧版本]' in review and '[当前记录]' in review
    assert all(t in review for t in texts)


@pytest.mark.asyncio
async def test_forged_correction_flag_cannot_replace(tmp_path):
    from dataclasses import replace

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'forged.sqlite'))
    scope = UserScope('web', 'test', 'u1', 's1', 'private')
    old, = extract_memories('我只有周末才去游泳。')
    new, = extract_memories('我只有放假才去游泳。')
    await write_rule_memory(repo, 'kisaki', scope, old, 'm0')
    await write_rule_memory(repo, 'kisaki', scope, replace(new, operation='replace'), 'm1')
    rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
    assert sorted(r['status'] for r in rows) == ['active', 'pending']


@pytest.mark.asyncio
async def test_correction_keeps_other_scope_action_and_pending_provenance(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'isolation.sqlite'))
    scope = UserScope('web', 'test', 'u1', 's1', 'private')
    other = UserScope('web', 'test', 'u2', 's1', 'private')
    for owner, text in [(scope, '我只有周末才去游泳。'),
                        (scope, '我只有有空才去游泳。'),
                        (scope, '我只有周末才看电影。'),
                        (other, '我只有周末才去游泳。'),
                        (scope, '更正一下，我只有放假才去游泳。')]:
        item, = extract_memories(text)
        await write_rule_memory(repo, 'kisaki', owner, item, text)
    rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
    assert sorted(r['status'] for r in rows) == ['active', 'active', 'pending', 'superseded']
    active = {r['metadata']['qualifiers']['action']: r['metadata']['qualifiers']['condition']
              for r in rows if r['status'] == 'active'}
    assert active == {'去游泳': '放假', '看电影': '周末'}
    other_rows = await repo.list_memory_records('kisaki', other, limit=None, include_inactive=True)
    assert len(other_rows) == 1 and other_rows[0]['status'] == 'active'


@pytest.mark.asyncio
async def test_correction_retries_concurrent_change_without_losing_old_source(tmp_path, monkeypatch):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'retry.sqlite'))
    scope = UserScope('web', 'test', 'u1', 's1', 'private')
    old, = extract_memories('我只有周末才去游泳。')
    new, = extract_memories('更正一下，我只有放假才去游泳。')
    await write_rule_memory(repo, 'kisaki', scope, old, 'old')
    append = repo.append_claim
    calls = 0

    async def retry_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError('rule memory changed concurrently')
        return await append(*args, **kwargs)

    monkeypatch.setattr(repo, 'append_claim', retry_once)
    assert await write_rule_memory(repo, 'kisaki', scope, new, 'new')
    assert calls == 2
    rows = await repo.list_memory_records('kisaki', scope, limit=None, include_inactive=True)
    assert len(rows) == 2 and {r['source_message_ids'][0] for r in rows} == {'old', 'new'}
