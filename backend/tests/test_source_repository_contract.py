from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope
from db.database import SQLiteDB

SCOPE = UserScope('web', 'test', 'reader', 'reader', 'private')
WHEN = datetime(2026, 10, 9, tzinfo=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize(('method', 'kwargs'), [
    ('capture_source', dict(source_message_id='source', body='完整的合成原文', observed_at=WHEN)),
    ('list_sources', {}), ('linked_source_receipts', dict(claim_sources=())),
    ('linked_source_revisions', dict(claim_sources=())), ('erase_unlinked_sources', dict(source_message_ids=('source',))),
])
async def test_missing_required_database_interface_raises(method, kwargs):
    repo = DatabaseCharacterMemoryRepository(object())
    with pytest.raises(AttributeError):
        await getattr(repo, method)('role', SCOPE, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('scope', [UserScope('web', 'narrative', 'reader', 'reader', 'private'),
    UserScope('web', '*', 'reader', 'reader', 'private'), UserScope('web', 'test', 'reader', '', 'group')])
async def test_invalid_source_authority_stops_before_model_and_claim_write(tmp_path, scope):
    db = SQLiteDB(tmp_path / 'source.sqlite')
    repo = DatabaseCharacterMemoryRepository(db)
    completion = SimpleNamespace(complete=AsyncMock(), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'fixture'), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(repository=repo, character_id='role', user_scope=scope,
            message='请记住，我今年大三。', rule_hints=(), source_message_id='source', observed_at=WHEN, timeout_seconds=2)
        assert receipt['status'] == 'failed' and receipt['stage'] == 'source_capture'
        assert receipt['error'] == 'ValueError' and receipt['persisted'] == 0
        completion.complete.assert_not_awaited()
    finally:
        await worker.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_database_capture_failure_is_not_an_unsupported_status():
    def fail(*args, **kwargs):
        raise OSError('fixture storage unavailable')
    repo = DatabaseCharacterMemoryRepository(SimpleNamespace(capture_memory_source=fail))
    with pytest.raises(OSError, match='fixture storage unavailable'):
        await repo.capture_source('role', SCOPE, source_message_id='source', body='完整的合成原文', observed_at=WHEN)
