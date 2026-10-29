from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from services.character_context import CharacterContextService, TurnInput

from character.models import RelationshipState, UserScope
from db.schemas import MessageRequest


@pytest.fixture
def scenario(monkeypatch):
    repo = SimpleNamespace(storage_owner=SimpleNamespace(), increment_interaction=AsyncMock(return_value=1),
                           capture_source=AsyncMock(return_value='recorded'))
    scheduler = SimpleNamespace(enabled=True, schedule=Mock(side_effect=OSError('private-storage-detail')))
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda _owner_database=None: scheduler)
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

    service, prepared, turn, repo, _ = scenario
    request = MessageRequest(message=turn.message, characterId='role', platform='web', adapter='web-character',
        senderId='reader', conversationId='reader', conversationType='private', sourceMessageId='source')
    warning = await generate._complete_character_turn(prepared, request, character_service=service, message_db=repo.storage_owner)
    assert '原文已保存' in warning and '失败' in warning
    assert 'private-storage-detail' not in warning + caplog.text


@pytest.mark.asyncio
async def test_partial_completion_warning_is_distinct_from_total_failure(scenario):
    from api import generate

    service, prepared, turn, repo, scheduler = scenario
    prepared.memory_operation_receipt = dict(status='partial', accepted=2, persisted=1)
    warning = await generate._complete_character_turn(prepared,
        MessageRequest(message=turn.message, sourceMessageId='source'), character_service=service,
                                                                        message_db=repo.storage_owner)
    assert '部分保存' in warning and '部分写入失败' in warning
    repo.capture_source.assert_not_awaited()
    scheduler.schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize(
    "message", ["请叫我杉岚。", "周末请叫我杉岚，工作日请叫我知禾。", "杉岚是朋友的别名，不要这样叫我。"]
)
async def test_disabled_semantic_memory_never_learns_address_via_another_writer(
    scenario, monkeypatch, failure, message, caplog
):
    service, prepared, _, repo, scheduler = scenario
    scheduler.enabled = False
    prepared.relationship = RelationshipState(stage="familiar", preferred_address="知禾")
    repo.upsert_relationship = AsyncMock(side_effect=AssertionError("Chat must not change the default setting"))
    if failure:
        repo.increment_interaction.side_effect = OSError("private-counter-body")
    turn = TurnInput(message, "web", "web-character", "reader", "reader", "private")
    result = await service.complete_turn(prepared, turn, source_message_id="source")
    assert result.failed_writes == (("interaction_count",) if failure else ())
    assert result.interaction_count == (0 if failure else 1)
    assert result.memory_enrichment_status == result.memory_enrichment_mode == "disabled"
    assert result.preferred_address == "知禾" and result.stage == "familiar"
    repo.upsert_relationship.assert_not_awaited()
    repo.capture_source.assert_not_awaited()
    scheduler.schedule.assert_not_called()
    assert "private-counter-body" not in caplog.text


@pytest.mark.asyncio
async def test_web_completion_preserves_memory_failure_alongside_counter_failure(scenario):
    from api import generate
    service, prepared, turn, repo, _ = scenario
    repo.increment_interaction.side_effect = OSError('private-counter-body')
    request = MessageRequest(message=turn.message, characterId='role', platform='web', adapter='web-character',
        senderId='reader', conversationId='reader', conversationType='private', sourceMessageId='source')
    warning = await generate._complete_character_turn(prepared, request, character_service=service, message_db=repo.storage_owner)
    assert '交互统计' in warning and '原文已保存' in warning and '长期记忆处理失败' in warning
    assert 'private-counter-body' not in warning


@pytest.mark.parametrize("status,capture", [("erased", "erase_request"), ("failed", "recorded"), ("partial", "recorded")])
async def test_existing_operation_receipt_needs_no_unused_interpretation_or_writer(scenario, monkeypatch, status, capture):
    service, prepared, turn, repo, scheduler = scenario
    receipt = {"status": status, "source_capture": capture, "persisted": int(status != "failed")}
    prepared.memory_operation_receipt = dict(receipt)
    unused = Mock(side_effect=AssertionError("Already executed operation must not be reinterpreted"))
    monkeypatch.setattr("services.character_context.parse_command", unused)
    monkeypatch.setattr("services.character_context.extract_memories", unused)
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", unused)
    outcome = await service.complete_turn(prepared, turn, source_message_id="already-executed")
    assert outcome.memory_enrichment_mode == "explicit_operation"
    assert outcome.memory_enrichment_status == status and outcome.source_capture == capture
    assert prepared.memory_operation_receipt == receipt and not outcome.memory_enrichment_scheduled
    unused.assert_not_called()
    repo.capture_source.assert_not_awaited()
    repo.increment_interaction.assert_awaited_once()
