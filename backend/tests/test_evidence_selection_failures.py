"""Evidence review failures cannot masquerade as a successful empty selection."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from services.character_context import CharacterContextService

from api.generate import _prepare_character_turn
from character.evidence_selector import ContextualEvidenceSelector, create_evidence_selector
from character.models import CharacterProfile, MemoryItem
from db.schemas import MessageRequest
from scripts.evaluate_contextual_memory import evaluate


def item(key="preference"):
    return MemoryItem(key, "user_fact", "用户喜欢咖啡但晚上不喝", evidence=("我喜欢咖啡，但晚上不喝。",))


@pytest.mark.parametrize("value", ["", "bad", "0", "-1", "nan", "inf", "121"])
def test_enabled_invalid_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        create_evidence_selector()


def test_explicit_disable_and_valid_timeout(monkeypatch):
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_TIMEOUT_SECONDS", "bad")
    assert create_evidence_selector() is None
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_TIMEOUT_SECONDS", "120")
    assert create_evidence_selector().timeout_seconds == 120


@pytest.mark.parametrize("keys", [[""], ["same", "same"], list(map(str, range(25)))])
async def test_invalid_or_over_budget_candidates_are_not_silently_dropped(keys):
    reviewer = AsyncMock()
    with pytest.raises(ValueError):
        await ContextualEvidenceSelector(reviewer).select("我的饮食偏好", [item(key) for key in keys])
    reviewer.assert_not_awaited()


@pytest.mark.parametrize("cap", [-1, 1.5, True])
async def test_invalid_output_cap_does_not_become_empty_success(cap):
    reviewer = AsyncMock()
    with pytest.raises(ValueError, match="max_items"):
        await ContextualEvidenceSelector(reviewer).select("我的饮食偏好", [item()], max_items=cap)
    reviewer.assert_not_awaited()


async def test_empty_candidates_skip_review_and_complete_selection_keeps_all_evidence():
    reviewer = AsyncMock()
    selector = ContextualEvidenceSelector(reviewer)
    assert (await selector.select("你好", [])).status == "empty"
    reviewer.assert_not_awaited()
    candidates = tuple(item(str(i)) for i in range(24))
    reviewer.return_value = json.dumps({"decisions": [{"id": x.memory_id, "label": "use"} for x in candidates]})
    result = await selector.select("逐条核对完整偏好", candidates)
    assert result.memories == candidates and result.status == "selected"
    assert all(a is b for a, b in zip(result.memories, candidates, strict=True))


@pytest.mark.parametrize("failure", [RuntimeError("synthetic private provider details"), ValueError("invalid output")])
async def test_review_failure_stops_api_preparation(failure):
    reviewer = AsyncMock(side_effect=failure)
    profile = CharacterProfile("role", "虚构角色", boundaries=("尊重明确的否定",))
    service = CharacterContextService(SimpleNamespace(get_profile=Mock(return_value=profile)),
        SimpleNamespace(get_relationship_record=AsyncMock(return_value={}), list_relationship_notes=AsyncMock(return_value=[])),
        SimpleNamespace(list_recent_conversation_history=AsyncMock(return_value=[])),
        memory_service=SimpleNamespace(recall_with_diagnostics=AsyncMock(return_value=((item(),), 1, {"status": "selected"}))),
        memory_selector=ContextualEvidenceSelector(reviewer), source_recall_enabled=False)
    request = MessageRequest(message="我晚上喝咖啡吗？", userId="reader", sessionId="reader")
    with pytest.raises(HTTPException) as caught:
        await _prepare_character_turn(request, "role", character_service=service)
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert "synthetic private" not in str(caught.value.detail)
    reviewer.assert_awaited_once()


async def test_timeout_cancels_inflight_review():
    cancelled = asyncio.Event()
    async def reviewer(messages):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()
    with pytest.raises(TimeoutError):
        await ContextualEvidenceSelector(reviewer, timeout_seconds=.001).select("我的偏好", [item()])
    assert cancelled.is_set()


async def test_evaluation_does_not_report_failed_review_as_empty_success():
    reviewer = AsyncMock(side_effect=RuntimeError("review failed"))
    case = {"id": "fiction", "query": "我晚上喝咖啡吗？", "gold_ids": ["preference"],
        "memories": [{"id": "preference", "memory_type": "user_fact", "content": "用户喜欢咖啡但晚上不喝",
                      "evidence": ["我喜欢咖啡，但晚上不喝。"]}]}
    with pytest.raises(RuntimeError, match="review failed"):
        await evaluate([case], "offline_model", reviewer=reviewer)
    reviewer.assert_awaited_once()
