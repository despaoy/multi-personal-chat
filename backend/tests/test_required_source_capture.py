from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import UserScope


@pytest.mark.asyncio
@pytest.mark.parametrize('source_only', [False, True])
@pytest.mark.parametrize('mode', ['missing_id', 'missing_interface', 'unknown_result', 'empty_result', 'storage_error', 'recorded'])
async def test_capture_must_complete_before_source_only_success_or_interpretation(mode, source_only):
    repo = SimpleNamespace(list_memory_records=AsyncMock(return_value=[]), append_claim=AsyncMock())
    if mode != 'missing_interface':
        repo.capture_source = AsyncMock(return_value=(None if mode == 'empty_result' else
            'unsupported_adapter' if mode == 'unknown_result' else 'recorded'))
        if mode == 'storage_error':
            repo.capture_source.side_effect = OSError('fixture storage failure')
    completion = SimpleNamespace(complete=AsyncMock(return_value='{"memories":[]}'), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unused', 'fixture'), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(repository=repo, character_id='role',
            user_scope=UserScope('test', 'capture', 'reader', 'reader', 'private'), message='请记住，我今年大三。',
            rule_hints=(), source_message_id=None if mode == 'missing_id' else 'complete-source',
            source_only=source_only, timeout_seconds=2)
        if mode == 'recorded':
            assert receipt['status'] == ('source_only' if source_only else 'no_change')
            assert receipt['source_capture'] == 'recorded'
            assert completion.complete.await_count == (0 if source_only else 1)
        else:
            assert receipt['status'] == 'failed' and receipt['stage'] == 'source_capture'
            assert receipt['persisted'] == 0
            repo.list_memory_records.assert_not_awaited()
            completion.complete.assert_not_awaited()
        repo.append_claim.assert_not_awaited()
    finally:
        await worker.shutdown(timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['missing_id', 'missing_interface', 'unknown_result'])
async def test_completion_cannot_enqueue_without_confirmed_capture(monkeypatch, mode):
    from unittest.mock import Mock

    from services.character_context import CharacterContextService, PreparedCharacterTurn, TurnInput

    from character.models import CompiledCharacterContext, RelationshipState

    scope = UserScope('test', 'capture', 'reader', 'reader', 'private')
    repo = SimpleNamespace(increment_interaction=AsyncMock(return_value=1))
    if mode != 'missing_interface':
        repo.capture_source = AsyncMock(return_value=None)
    scheduler = SimpleNamespace(enabled=True, schedule=Mock())
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda: scheduler)
    service = CharacterContextService(object(), repo, object())
    prepared = PreparedCharacterTurn('role', scope, CompiledCharacterContext('', '', ''), (),
                                     RelationshipState(), 0, 0, None)
    turn = TurnInput('请记住，我今年大三。', 'test', 'capture', 'reader', 'reader', 'private')
    outcome = await service.complete_turn(prepared, turn, '已生成的回复',
                                         source_message_id='' if mode == 'missing_id' else 'source')
    assert outcome.memory_enrichment_status == ('failed' if mode == 'unknown_result' else 'source_capture_failed')
    assert not outcome.memory_enrichment_scheduled
    scheduler.schedule.assert_not_called()
