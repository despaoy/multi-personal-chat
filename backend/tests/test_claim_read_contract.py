from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.models import MemoryItem, UserScope
from db.database import SQLiteDB

SCOPE = UserScope('test', 'claim-read', 'reader', 'room', 'private')


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['list_memory_records', 'get_memory_record', 'source_windows'])
async def test_missing_formal_reader_never_scans_legacy_records(method):
    legacy = Mock(return_value=[])
    repo = DatabaseCharacterMemoryRepository(SimpleNamespace(list_character_memories=legacy))
    with pytest.raises(AttributeError):
        if method == 'get_memory_record':
            await repo.get_memory_record(900, 'role', SCOPE)
        elif method == 'source_windows':
            await repo.source_windows('role', SCOPE, source_message_ids=('source',))
        else:
            await repo.list_memory_records('role', SCOPE, scope_levels=('user_global',), include_inactive=True)
    legacy.assert_not_called()


@pytest.mark.asyncio
async def test_database_remains_authority_for_active_history_and_owner_scope(tmp_path):
    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'claims.sqlite'))
    first = await repo.append_claim('role', SCOPE, MemoryItem('', 'user_fact', '用户专业是统计学', .9), memory_key='user_major')
    current = await repo.append_claim('role', SCOPE, MemoryItem('', 'user_fact', '用户专业是海洋工程', .9),
        memory_key='user_major', relation_type='SUPERSEDE', supersedes_memory_id=first['id'])
    pending = await repo.append_claim('role', SCOPE, MemoryItem('', 'user_fact', '用户可能喜欢绘画', .6),
        memory_key='preference_绘画', relation_type='PENDING', status='pending')
    assert [r['id'] for r in await repo.list_memory_records('role', SCOPE)] == [current['id']]
    history = await repo.list_memory_records('role', SCOPE, include_inactive=True)
    assert {r['id'] for r in history} == {first['id'], current['id'], pending['id']}
    assert (await repo.get_memory_record(first['id'], 'role', SCOPE))['status'] == 'superseded'
    foreign = UserScope('test', 'claim-read', 'other-reader', 'room', 'private')
    assert await repo.list_memory_records('role', foreign, include_inactive=True) == []
    assert await repo.get_memory_record(first['id'], 'role', foreign) is None
    assert await repo.get_memory_record(current['id'] + 999, 'role', SCOPE) is None



@pytest.mark.asyncio
async def test_exact_id_filter_keeps_history_and_scope_and_empty_means_no_rows(tmp_path):
    database = SQLiteDB(tmp_path / 'exact-claims.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    rows = []
    try:
        for index, level in enumerate(('conversation', 'user_character', 'user_global')):
            row = await repo.append_claim('role', SCOPE, MemoryItem('', 'user_fact', f'完整虚构事实{index}'),
                memory_key=f'fact-{index}', scope_level=level, status='superseded' if index == 0 else 'active')
            rows.append(row)
        ids = tuple(row['id'] for row in rows)
        selected = await repo.list_memory_records('role', SCOPE, limit=None, include_inactive=True, memory_ids=ids)
        expected = {row['id']: await repo.get_memory_record(row['id'], 'role', SCOPE) for row in rows}
        assert {row['id']: row for row in selected} == expected
        assert await repo.list_memory_records('role', SCOPE, limit=None, include_inactive=True, memory_ids=()) == []
        assert await repo.list_memory_records('role', SCOPE, limit=None, include_inactive=True, memory_ids=(999999,)) == []
        foreign = UserScope('test', 'claim-read', 'foreign-owner', 'room', 'private')
        assert await repo.list_memory_records('role', foreign, limit=None, include_inactive=True, memory_ids=ids) == []
        narrowed = await repo.list_memory_records('role', SCOPE, limit=None, memory_ids=ids, scope_levels=('user_global',))
        assert [row['id'] for row in narrowed] == [rows[2]['id']]
        fields = ('role', SCOPE.platform, SCOPE.adapter, SCOPE.sender_id, SCOPE.conversation_type, SCOPE.conversation_id)
        both = database.list_character_memory_claims(*fields, limit=None, include_inactive=True,
                                                     memory_ids=ids, memory_keys=('fact-1',))
        assert [row['id'] for row in both] == [rows[1]['id']]
    finally:
        database.close_connection()
