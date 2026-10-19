import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import MemoryItem, UserScope

SCOPE = UserScope('test', 'fixture', 'reader', 'reader', 'private')
ITEM = MemoryItem('', 'user_fact', '用户今年大三', .9)


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'relation_type': operation} for operation in ('ADD', 'PENDING', 'SUPERSEDE', 'RETRACT', 'MERGE', 'COEXIST')
] + [{'status': status} for status in ('pending', 'retracted', 'superseded', 'archived')]
  + [{'scope_level': 'user_global'}, {'scope_level': 'user_character'},
     {'parent_memory_id': 1}, {'supersedes_memory_id': 1}, {'valid_to': '2026-09-01'},
     {'valid_from': '2026-09-01'}])
async def test_repository_requires_versioned_database_without_legacy_upsert(changes):
    database = SimpleNamespace(add_or_update_character_memory=Mock())
    repo = DatabaseCharacterMemoryRepository(database)
    with pytest.raises(AttributeError, match='append_character_memory_claim'):
        await repo.append_claim('role', SCOPE, ITEM, memory_key='user_study_stage', **changes)
    database.add_or_update_character_memory.assert_not_called()


async def test_explicit_noop_needs_no_legacy_write():
    database = SimpleNamespace(add_or_update_character_memory=Mock())
    result = await DatabaseCharacterMemoryRepository(database).append_claim(
        'role', SCOPE, ITEM, memory_key='user_study_stage', relation_type='NOOP')
    assert result == {'id': 0, 'persisted': False, 'relation_type': 'NOOP', 'memory_key': 'user_study_stage'}
    database.add_or_update_character_memory.assert_not_called()


async def test_versioned_repository_add_read_and_original_source_round_trip(tmp_path):
    from datetime import datetime, timezone

    from db.database import SQLiteDB

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'versioned.sqlite'))
    clock = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    assert await repo.capture_source('role', SCOPE, source_message_id='fixture-source',
        body='我今年大三。', observed_at=clock) == 'recorded'
    saved = await repo.append_claim('role', SCOPE, ITEM, memory_key='user_study_stage',
        evidence=('我今年大三',), source_message_id='fixture-source', observed_at=clock.isoformat())
    assert saved['persisted'] is True and saved['revision'] > 0
    read = await repo.get_memory_record(saved['id'], 'role', SCOPE)
    assert read['content'] == ITEM.content and read['memory_key'] == 'user_study_stage'
    assert read['source_message_ids'] == ['fixture-source'] and read['evidence'] == ['我今年大三']


@pytest.mark.asyncio
@pytest.mark.parametrize('target', [dict(memory_id=7), dict(memory_key='user_study_stage'),
                                  dict(memory_key='user_study_stage', protected_memory_keys=('keep',))])
async def test_repository_erasure_requires_complete_erasure_interface(target):
    database = SimpleNamespace(delete_character_memory=Mock())
    repo = DatabaseCharacterMemoryRepository(database)
    with pytest.raises(AttributeError, match='erase_character_memories'):
        await repo.erase_memory('role', SCOPE, **target)
    database.delete_character_memory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('record', [None, {}, {'persisted': None}, {'persisted': 1},
                                 {'persisted': True}, {'persisted': False}, 'missing-interface'])
async def test_writer_reports_only_explicit_persistence_results(record):
    legacy = AsyncMock()
    repo = SimpleNamespace(storage_owner=object(), list_memory_records=AsyncMock(return_value=[]),
        capture_source=AsyncMock(return_value='recorded'), add_or_update_memory=legacy)
    if record != 'missing-interface':
        repo.append_claim = AsyncMock(return_value=record)
    completion = SimpleNamespace(close=AsyncMock(), complete=AsyncMock(return_value=json.dumps({'memories': [
        dict(kind='study_stage', value='大三', evidence='我今年大三', confidence=.99, operation='ADD', attributed_to='user')]})))
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'http://unused', 'fixture'), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(repository=repo, character_id='role', user_scope=SCOPE,
            message='请记住，我今年大三。', rule_hints=(), source_message_id='full-source', timeout_seconds=2)
        if record == {'persisted': True}:
            # bool and int compare equal; the identity check is the actual contract.
            expected = 'saved' if record['persisted'] is True else 'failed'
        elif record == {'persisted': False}:
            expected = 'no_change'
        else:
            expected = 'failed'
        assert receipt['status'] == expected
        assert receipt['persisted'] == int(expected == 'saved')
        assert receipt['source_capture'] == 'recorded' and receipt['stage'] == 'persistence'
        legacy.assert_not_awaited()
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
