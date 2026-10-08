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
@pytest.mark.parametrize('operation', ['ADD', 'PENDING', 'SUPERSEDE', 'RETRACT'])
async def test_repository_requires_versioned_database_without_legacy_upsert(operation):
    database = SimpleNamespace(add_or_update_character_memory=Mock())
    repo = DatabaseCharacterMemoryRepository(database)
    with pytest.raises(AttributeError, match='append_character_memory_claim'):
        await repo.append_claim('role', SCOPE, ITEM, memory_key='user_study_stage', relation_type=operation)
    database.add_or_update_character_memory.assert_not_called()


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
    repo = SimpleNamespace(list_memory_records=AsyncMock(return_value=[]),
        capture_source=AsyncMock(return_value='recorded'), add_or_update_memory=legacy)
    if record != 'missing-interface':
        repo.append_claim = AsyncMock(return_value=record)
    completion = SimpleNamespace(close=AsyncMock(), complete=AsyncMock(return_value=json.dumps({'memories': [
        dict(kind='study_stage', value='大三', evidence='我今年大三', confidence=.99, operation='ADD')]})))
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
