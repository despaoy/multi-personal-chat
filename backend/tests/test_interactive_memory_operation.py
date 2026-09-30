from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.memory_operation import operation_receipt_context
from character.models import CompiledCharacterContext, RelationshipState, UserScope
from services.character_context import CharacterContextService, PreparedCharacterTurn, TurnInput


def setup_service(monkeypatch, receipt):
    scope = UserScope('web', 'test', 'u', 'c', 'private')
    prepared = PreparedCharacterTurn('role', scope, CompiledCharacterContext('', '', ''), (),
                                    RelationshipState(), 0, 0, None)
    service = object.__new__(CharacterContextService)
    service._profiles = SimpleNamespace(get_profile=lambda character: object())
    service._memory_repo = SimpleNamespace(increment_interaction=AsyncMock(return_value=1))
    service._load_history = AsyncMock(return_value=[])
    service.prepare_turn = AsyncMock(return_value=prepared)
    worker = SimpleNamespace(enabled=True, schedule_and_wait=AsyncMock(return_value=receipt),
                             schedule=lambda **kwargs: pytest.fail('duplicate operation submitted'))
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda: worker)
    return service, worker


@pytest.mark.parametrize('status', ['erased', 'pending', 'failed', 'partial', 'cancelled', 'not_scheduled', 'conflict'])
async def test_explicit_operation_precedes_snapshot_and_is_not_resubmitted(monkeypatch, status):
    service, worker = setup_service(monkeypatch, {'status': status, 'persisted': int(status == 'erased')})
    original_prepare = service.prepare_turn

    async def snapshot(*args):
        worker.schedule_and_wait.assert_awaited_once()
        return await original_prepare(*args)

    service.prepare_turn = snapshot
    turn = TurnInput('请从记忆里删除我的住址。', 'web', 'test', 'u', 'c', 'private')
    prepared = await service.prepare_interactive_turn(turn, 'role', source_message_id='request-1')
    assert prepared.memory_operation_receipt['status'] == status
    assert '操作执行回执' in prepared.compiled.dynamic_context
    assert worker.schedule_and_wait.await_args.kwargs['source_message_id'] == 'request-1'
    outcome = await service.complete_turn(prepared, turn, '回复', source_message_id='request-1')
    assert outcome.memory_enrichment_status == status
    assert outcome.memory_enrichment_mode == 'explicit_operation'
    assert not outcome.memory_enrichment_scheduled


@pytest.mark.parametrize('message', ['我住在临沂。', '不要忘掉我的住址。', '“请忘掉我的住址”是示例。'])
async def test_ordinary_and_non_authorizing_turns_do_not_execute(monkeypatch, message):
    service, worker = setup_service(monkeypatch, {})
    result = await service.prepare_interactive_turn(TurnInput(message, 'web', 'test', 'u', 'c', 'private'), 'role')
    assert result.memory_operation_receipt is None
    worker.schedule_and_wait.assert_not_awaited()


async def test_invalid_profile_prevents_operation(monkeypatch):
    service, worker = setup_service(monkeypatch, {})

    def missing(character):
        raise ValueError('missing profile')

    service._profiles.get_profile = missing
    with pytest.raises(ValueError, match='missing profile'):
        await service.prepare_interactive_turn(TurnInput('忘掉我的住址。', 'web', 'test', 'u', 'c', 'private'), 'role')
    worker.schedule_and_wait.assert_not_awaited()


def test_receipt_rendering_never_includes_source_or_arbitrary_error():
    rendered = operation_receipt_context({'status': 'ignore instructions', 'persisted': 'secret', 'error': 'private fact'})
    assert 'ignore instructions' not in rendered and 'secret' not in rendered and 'private fact' not in rendered
    assert '结果未确认' in rendered


@pytest.mark.parametrize('enabled', [False, True])
async def test_api_preparation_only_executes_when_explicitly_enabled(enabled):
    from api.generate import _prepare_character_turn
    from db.schemas import MessageRequest

    service = SimpleNamespace(prepare_turn=AsyncMock(return_value='read-only'),
                              prepare_interactive_turn=AsyncMock(return_value='interactive'))
    request = MessageRequest(message='忘掉我的住址。', userId='u', sessionId='c', sourceMessageId='s')
    result = await _prepare_character_turn(request, 'role', character_service=service,
                                           execute_memory_operations=enabled)
    assert result == ('interactive' if enabled else 'read-only')
    if enabled:
        service.prepare_turn.assert_not_awaited()
    else:
        service.prepare_interactive_turn.assert_not_awaited()
