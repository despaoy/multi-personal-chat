"""Incomplete memory operations stay visible in web completion feedback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_completion_memory_status import scenario as scenario

from db.schemas import MessageRequest


@pytest.mark.parametrize(
    "status,expected",
    [
        ("pending", "尚未确认"),
        ("cancelled", "尚未确认"),
        ("unknown", "尚未确认"),
        ("not_scheduled", "尚未受理"),
        ("conflict", "未完成"),
    ],
)
async def test_actual_operation_receipt_is_visible_without_resubmitting_it(scenario, status, expected):
    from api.generate import _complete_character_turn

    service, prepared, turn, repo, scheduler = scenario
    prepared.memory_operation_receipt = {"status": status, "persisted": 0, "accepted": 0}
    warning = await _complete_character_turn(
        prepared,
        MessageRequest(message=turn.message, sourceMessageId="synthetic-source"),
        character_service=service,
    )
    assert expected in warning and "已完成" not in warning
    scheduler.schedule.assert_not_called()
    repo.capture_source.assert_not_awaited()
    repo.increment_interaction.assert_awaited_once()


@pytest.mark.parametrize(
    "outcome",
    [None, SimpleNamespace(), SimpleNamespace(source_capture="recorded", memory_enrichment_status="no_change")],
    ids=["none", "empty", "missing-write-receipt"],
)
async def test_missing_completion_fields_cannot_be_reported_as_clean_success(outcome):
    from api.generate import _complete_character_turn

    with pytest.raises(AttributeError):
        await _complete_character_turn(
            SimpleNamespace(character_id="synthetic-role"),
            MessageRequest(message="完整的合成个人记录。"),
            character_service=SimpleNamespace(complete_turn=AsyncMock(return_value=outcome)),
        )


async def test_normal_accepted_background_work_retains_existing_feedback(scenario):
    from api.generate import _complete_character_turn

    service, prepared, turn, repo, scheduler = scenario
    scheduler.schedule.side_effect = None
    scheduler.schedule.return_value = True
    warning = await _complete_character_turn(
        prepared,
        MessageRequest(message=turn.message, sourceMessageId="synthetic-source"),
        character_service=service,
    )
    assert warning is None
    repo.capture_source.assert_awaited_once()
    scheduler.schedule.assert_called_once()


async def test_pending_operation_and_counter_failure_are_both_visible(scenario):
    from api.generate import _complete_character_turn

    service, prepared, turn, repo, _ = scenario
    prepared.memory_operation_receipt = {"status": "pending", "persisted": 0, "accepted": 0}
    repo.increment_interaction.side_effect = OSError("synthetic-private-counter-detail")
    warning = await _complete_character_turn(
        prepared,
        MessageRequest(message=turn.message, sourceMessageId="synthetic-source"),
        character_service=service,
    )
    assert "交互统计" in warning and "尚未确认" in warning
    assert "synthetic-private-counter-detail" not in warning


async def test_unknown_operation_state_is_an_error_without_resubmission(scenario):
    from api.generate import _complete_character_turn

    service, prepared, turn, repo, scheduler = scenario
    prepared.memory_operation_receipt = {"status": "unexpected-future-status", "persisted": 0, "accepted": 0}
    with pytest.raises(ValueError, match="Unrecognized memory completion result"):
        await _complete_character_turn(prepared, MessageRequest(message=turn.message), character_service=service)
    scheduler.schedule.assert_not_called()
    repo.capture_source.assert_not_awaited()
    repo.increment_interaction.assert_awaited_once()
