from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from services.character_context import CharacterContextService, TurnInput

from character.memory_extractor import ExtractedMemory
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
    result = await service.complete_turn(prepared, turn, '已生成的回复', source_message_id='source')
    assert result.source_capture == 'recorded' and result.memory_enrichment_status == 'failed'
    assert result.new_memories == 0 and not result.memory_enrichment_scheduled
    assert result.interaction_count == 1
    repo.capture_source.assert_awaited_once()
    scheduler.schedule.assert_called_once()
    assert 'private-storage-detail' not in caplog.text and 'OSError' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('saved', [0, 1])
async def test_rule_failure_keeps_completed_count_and_never_reports_skipped(scenario, monkeypatch, saved):
    service, prepared, turn, _, scheduler = scenario
    scheduler.enabled = False
    monkeypatch.setattr('services.character_context.extract_memories', lambda *a, **k: (
        ExtractedMemory('user_fact', 'user_study_stage', '用户今年大三', .9),
        ExtractedMemory('user_fact', 'user_major', '用户的专业是统计学', .9)))
    turn = TurnInput('我今年大三，专业是统计学。', 'web', 'web-character', 'reader', 'reader', 'private')
    write = AsyncMock(side_effect=[True] * saved + [OSError('private-storage-detail')])
    monkeypatch.setattr('services.character_context.write_rule_memory', write)
    result = await service.complete_turn(prepared, turn, '已生成的回复', source_message_id='source')
    assert result.new_memories == saved
    assert result.memory_enrichment_status == ('partial' if saved else 'failed')
    assert write.await_count == saved + 1
    scheduler.schedule.assert_not_called()


@pytest.mark.asyncio
async def test_actual_service_failure_is_visible_through_web_completion(scenario, caplog):
    from api import generate

    service, prepared, turn, _, _ = scenario
    request = MessageRequest(message=turn.message, characterId='role', platform='web', adapter='web-character',
        senderId='reader', conversationId='reader', conversationType='private', sourceMessageId='source')
    warning = await generate._complete_character_turn(prepared, request, '模型原始回复', character_service=service)
    assert '原文已保存' in warning and '失败' in warning
    assert 'private-storage-detail' not in warning + caplog.text


@pytest.mark.asyncio
async def test_partial_completion_warning_is_distinct_from_total_failure():
    from api import generate

    service = SimpleNamespace(complete_turn=AsyncMock(return_value=SimpleNamespace(
        source_capture='', memory_enrichment_status='partial', new_memories=1)))
    warning = await generate._complete_character_turn(SimpleNamespace(character_id='role'),
        MessageRequest(message='我今年大三，专业是统计学。', sourceMessageId='source'), '原始回复', character_service=service)
    assert '部分保存' in warning and '部分写入失败' in warning
