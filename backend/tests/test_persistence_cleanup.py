"""Internal persistence failures must not be reported as successful no-ops."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, ValidatedMemoryProposal, _MemoryJob
from character.models import UserScope


def setup():
    repo = SimpleNamespace(erase_memory=AsyncMock(),append_claim=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True,'unused','fixture'),
        completion=SimpleNamespace(close=AsyncMock()))
    job = _MemoryJob(repository=repo,character_id='role',
        user_scope=UserScope('web','contract','fiction','fiction','private'),
        message='请从记忆中删除红茶偏好。',rule_hints=(),history=(),source_message_id='synthetic',
        observed_at=datetime(2026,10,9,tzinfo=timezone.utc))
    return worker,job,repo

@pytest.mark.parametrize('operation',['ADD','PENDING','MERGE','SUPERSEDE','COEXIST','RETRACT'])
async def test_write_without_memory_content_is_an_error(operation):
    worker,job,repo=setup()
    try:
        with pytest.raises(ValueError,match='memory'):
            await worker._persist_proposal(job,ValidatedMemoryProposal(operation=operation))
        repo.append_claim.assert_not_awaited()
        repo.erase_memory.assert_not_awaited()
    finally:
        await worker.shutdown(timeout=1)

@pytest.mark.parametrize('deleted',[None,'1',0,2])
async def test_delete_result_uses_repository_count_without_default_or_conversion(deleted):
    worker,job,repo=setup()
    repo.erase_memory.return_value=deleted
    proposal=ValidatedMemoryProposal(operation='ERASE',target_memory_id='7',
        target_memory_key='preference_红茶',evidence=job.message)
    try:
        if deleted is None or isinstance(deleted,str):
            with pytest.raises(TypeError):
                await worker._persist_proposal(job,proposal)
            assert worker.status.erased==0
        else:
            assert await worker._persist_proposal(job,proposal)==('erased' if deleted else 'no_change')
            assert worker.status.erased==bool(deleted)
    finally:
        await worker.shutdown(timeout=1)

async def test_explicit_noop_does_not_require_memory_or_write():
    worker,job,repo=setup()
    try:
        assert await worker._persist_proposal(job,ValidatedMemoryProposal(operation='NOOP'))=='no_change'
        repo.append_claim.assert_not_awaited()
        repo.erase_memory.assert_not_awaited()
    finally:
        await worker.shutdown(timeout=1)
