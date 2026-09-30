import sqlite3

import pytest

from character.context_builder import build_user_scope
from character.memory_extractor import extract_memories
from character.models import MemoryItem
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from db.pending_resolution import pending_resolution_ids
from repositories.character_memory import DatabaseCharacterMemoryRepository


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['补充', '更正'])
async def test_explicit_adoption_closes_only_matching_candidate(tmp_path, operation):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'resolve.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    texts = ['我只有收到确认才预约会议。', '我只有所有人有空才预约会议。',
             '我只有会议室可用才预约会议。', f'{operation}，我只有所有人有空才预约会议。']
    for i, text in enumerate(texts):
        item, = extract_memories(text)
        assert await write_rule_memory(repo, 'c', scope, item, f'm{i}')
    rows = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    adopted, = [r for r in rows if r['source_message_ids'] == ['m1']]
    assert adopted['status'] == 'archived' and adopted['valid_to']
    assert adopted['evidence'] == [texts[1]]
    remaining, = [r for r in rows if r['source_message_ids'] == ['m2']]
    assert remaining['status'] == 'pending'
    current, = [r for r in rows if r['status'] == 'active']
    assert current['metadata']['resolved_pending_ids'] == [adopted['id']]
    assert adopted['id'] != current['id']


@pytest.mark.asyncio
@pytest.mark.parametrize('mismatch', ['owner', 'key', 'status'])
async def test_invalid_resolution_rolls_back_entire_append(tmp_path, mismatch):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'rollback.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    other = build_user_scope('web', 'test', 'u2', 's1', 'private')
    pending = await repo.append_claim('c', other if mismatch == 'owner' else scope,
        MemoryItem('', 'user_fact', '候选'), memory_key='other' if mismatch == 'key' else 'target',
        relation_type='ADD' if mismatch == 'status' else 'PENDING')
    before = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    with pytest.raises(ValueError, match='exact rule scope'):
        await repo.append_claim('c', scope, MemoryItem('', 'user_fact', '新版本'), memory_key='target',
            metadata={'origin': 'rule_v2', 'operation': 'replace', 'resolved_pending_ids': [pending['id']]})
    after = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    assert after == before


@pytest.mark.parametrize('ids', [[True], ['1'], [0], [-1], [1, 1], [], None])
def test_resolution_ids_require_exact_unique_positive_integers(ids):
    with pytest.raises(ValueError):
        pending_resolution_ids({'origin': 'rule_v2', 'operation': 'append', 'resolved_pending_ids': ids}, 'SUPERSEDE', 'active')


@pytest.mark.asyncio
async def test_archive_failure_rolls_back_insert_and_supersession(tmp_path):
    database = SQLiteDB(tmp_path / 'atomic.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    for i, text in enumerate(['我只有周末才去游泳。', '我只有天气合适才去游泳。']):
        item, = extract_memories(text)
        await write_rule_memory(repo, 'c', scope, item, str(i))
    before = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    connection = database._get_connection()
    connection.execute("CREATE TRIGGER fail_archive BEFORE UPDATE ON character_memories "
                       "WHEN NEW.status = 'archived' BEGIN SELECT RAISE(ABORT, 'test archive failure'); END")
    connection.commit()
    item, = extract_memories('补充，我只有天气合适才去游泳。')
    with pytest.raises(sqlite3.IntegrityError, match='test archive failure'):
        await write_rule_memory(repo, 'c', scope, item, 'new')
    assert await repo.list_memory_records('c', scope, limit=None, include_inactive=True) == before


@pytest.mark.asyncio
async def test_resolution_preserves_other_character_and_other_action(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'isolation.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    for character in ('c', 'other'):
        for action in ('去游泳', '去爬山'):
            for condition in ('周末', '天气合适'):
                item, = extract_memories(f'我只有{condition}才{action}。')
                await write_rule_memory(repo, character, scope, item, f'{character}-{action}-{condition}')
    item, = extract_memories('补充，我只有天气合适才去游泳。')
    await write_rule_memory(repo, 'c', scope, item, 'adoption')
    rows = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    assert len([r for r in rows if r['status'] == 'pending']) == 1
    assert '去爬山' in next(r['content'] for r in rows if r['status'] == 'pending')
    rows = await repo.list_memory_records('other', scope, limit=None, include_inactive=True)
    assert len([r for r in rows if r['status'] == 'pending']) == 2


@pytest.mark.asyncio
async def test_resolution_reloads_after_concurrent_candidate_change(tmp_path, monkeypatch):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'retry.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    for i, text in enumerate(['我只有周末才去游泳。', '我只有天气合适才去游泳。']):
        item, = extract_memories(text)
        await write_rule_memory(repo, 'c', scope, item, str(i))
    append = repo.append_claim
    attempts = []

    async def retry(*args, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise ValueError('pending memory is not available in the exact rule scope')
        return await append(*args, **kwargs)

    monkeypatch.setattr(repo, 'append_claim', retry)
    item, = extract_memories('补充，我只有天气合适才去游泳。')
    assert await write_rule_memory(repo, 'c', scope, item, 'adoption')
    assert len(attempts) == 2
    rows = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    assert sorted(r['status'] for r in rows) == ['active', 'archived', 'superseded']


@pytest.mark.asyncio
async def test_archived_adoption_does_not_suppress_a_later_new_candidate(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'new-candidate.sqlite'))
    scope = build_user_scope('web', 'test', 'u1', 's1', 'private')
    texts = ['我只有周末才去游泳。', '我只有天气合适才去游泳。',
             '补充，我只有天气合适才去游泳。', '更正，我只有周末才去游泳。',
             '我只有天气合适才去游泳。']
    for i, text in enumerate(texts):
        item, = extract_memories(text)
        assert await write_rule_memory(repo, 'c', scope, item, str(i))
    rows = await repo.list_memory_records('c', scope, limit=None, include_inactive=True)
    assert next(r for r in rows if r['status'] == 'pending')['source_message_ids'] == ['4']
    assert next(r for r in rows if r['status'] == 'archived')['source_message_ids'] == ['1']
