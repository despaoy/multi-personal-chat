from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from services.character_context import CharacterContextService, TurnInput

from character.models import RelationshipState, UserScope
from db.schemas import MessageRequest


@pytest.fixture
def scenario(monkeypatch):
    repo = SimpleNamespace(increment_interaction=AsyncMock(return_value=1),
                           capture_source=AsyncMock(return_value='recorded'))
    scheduler = SimpleNamespace(enabled=True, schedule=Mock(side_effect=OSError('private-storage-detail')))
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda: scheduler)
    prepared = SimpleNamespace(character_id='role', user_scope=UserScope('web', 'web-character', 'reader', 'reader', 'private'),
        received_at=datetime(2026, 10, 9, tzinfo=timezone.utc), relationship=RelationshipState(),
        memory_operation_receipt=None, history=(), compiled=SimpleNamespace(used_memory_ids=()))
    turn = TurnInput('我今年大三。', 'web', 'web-character', 'reader', 'reader', 'private')
    service = CharacterContextService(object(), repo, object())
    return service, prepared, turn, repo, scheduler


@pytest.mark.asyncio
async def test_scheduling_failure_preserves_source_receipt_and_reports_failure(scenario, caplog):
    service, prepared, turn, repo, scheduler = scenario
    result = await service.complete_turn(prepared, turn, source_message_id='source')
    assert result.source_capture == 'recorded' and result.memory_enrichment_status == 'failed'
    assert not result.memory_enrichment_scheduled
    assert result.interaction_count == 1
    repo.capture_source.assert_awaited_once()
    scheduler.schedule.assert_called_once()
    assert 'private-storage-detail' not in caplog.text and 'OSError' in caplog.text


@pytest.mark.asyncio
async def test_actual_service_failure_is_visible_through_web_completion(scenario, caplog):
    from api import generate

    service, prepared, turn, _, _ = scenario
    request = MessageRequest(message=turn.message, characterId='role', platform='web', adapter='web-character',
        senderId='reader', conversationId='reader', conversationType='private', sourceMessageId='source')
    warning = await generate._complete_character_turn(prepared, request, character_service=service)
    assert '原文已保存' in warning and '失败' in warning
    assert 'private-storage-detail' not in warning + caplog.text


@pytest.mark.asyncio
async def test_partial_completion_warning_is_distinct_from_total_failure(scenario):
    from api import generate

    service, prepared, turn, repo, scheduler = scenario
    prepared.memory_operation_receipt = dict(status='partial', accepted=2, persisted=1)
    warning = await generate._complete_character_turn(prepared,
        MessageRequest(message=turn.message, sourceMessageId='source'), character_service=service)
    assert '部分保存' in warning and '部分写入失败' in warning
    repo.capture_source.assert_not_awaited()
    scheduler.schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['count', 'address', 'missing_adapter', 'unknown_status', 'both', 'none'])
async def test_required_completion_writes_are_explicit_and_do_not_use_legacy_path(scenario, monkeypatch, failure, caplog):
    service, prepared, _, repo, scheduler = scenario
    scheduler.enabled = False
    monkeypatch.setattr('services.character_context.extract_memories', lambda *a, **k: ())
    repo.upsert_relationship = AsyncMock(side_effect=AssertionError('legacy write is forbidden'))
    repo.set_address_from_turn = AsyncMock(return_value={
        'status': 'written', 'relationship': {'relationship_stage': 'familiar'}})
    if failure in {'count', 'both'}:
        repo.increment_interaction.side_effect = OSError('private-counter-body')
    if failure in {'address', 'both'}:
        repo.set_address_from_turn.side_effect = OSError('private-address-body')
    if failure == 'missing_adapter':
        del repo.set_address_from_turn
    if failure == 'unknown_status':
        repo.set_address_from_turn.return_value = {'status': 'unknown'}
    turn = TurnInput('请叫我杉岚。', 'web', 'web-character', 'reader', 'reader', 'private')
    result = await service.complete_turn(prepared, turn, source_message_id='source')
    expected = (() if failure not in {'count', 'both'} else ('interaction_count',))
    if failure in {'address', 'missing_adapter', 'unknown_status', 'both'}:
        expected += ('preferred_address',)
    assert result.failed_writes == expected
    assert result.memory_enrichment_status == result.memory_enrichment_mode == 'disabled'
    if 'preferred_address' not in expected:
        assert result.preferred_address == '杉岚' and result.stage == 'familiar'
        repo.set_address_from_turn.assert_awaited_once_with(prepared.character_id, prepared.user_scope,
            source_message_id='source', observed_at=prepared.received_at, address='杉岚')
    repo.upsert_relationship.assert_not_awaited()
    assert 'private-counter-body' not in caplog.text and 'private-address-body' not in caplog.text


@pytest.mark.asyncio
async def test_web_completion_preserves_memory_failure_alongside_counter_failure(scenario):
    from api import generate
    service, prepared, turn, repo, _ = scenario
    repo.increment_interaction.side_effect = OSError('private-counter-body')
    request = MessageRequest(message=turn.message, characterId='role', platform='web', adapter='web-character',
        senderId='reader', conversationId='reader', conversationType='private', sourceMessageId='source')
    warning = await generate._complete_character_turn(prepared, request, character_service=service)
    assert '交互统计' in warning and '原文已保存' in warning and '长期记忆处理失败' in warning
    assert 'private-counter-body' not in warning
