"""Known input-capacity failures stay distinct from provider outages at APIs."""

import pytest
from test_ask_request_context import ask_rig as ask_rig, request
from test_provider_error_surface import provider_api as provider_api, events

from inference.context_budget import ReviewContextBudget
from knowledge.grounded_answer.packet import EvidencePacketBuilder
from knowledge.grounded_answer.prompt import GroundedPromptBuilder

pytestmark = pytest.mark.usefixtures("no_live_context_review", "literal_grounded_query")


@pytest.mark.parametrize("streaming", [False, True])
async def test_real_capacity_error_is_explicit_without_calling_provider_or_caching(provider_api, monkeypatch, streaming):
    rig = provider_api.state.rig
    monkeypatch.setattr("inference.context_budget.estimated_tokens", len)
    rig.service.packet_builder = EvidencePacketBuilder(None)
    rig.service.prompt_builder = GroundedPromptBuilder(None, context_budget=ReviewContextBudget(1024))
    response = await provider_api.client.post("/api/ask/stream" if streaming else "/api/ask",
        json={**request().model_dump(), "maxTokens": 1024})
    if streaming:
        emitted = events(response)
        assert response.status_code == 200 and [item["type"] for item in emitted] == ["meta", "error"]
        detail = emitted[-1]
    else:
        assert response.status_code == 503 and response.json()["success"] is False
        detail = response.json()
    assert detail["kind"] == "context_budget_exceeded" and detail["stage"] == "context_budget"
    assert detail["reason"] == "capacity_exceeded" and "预算" in detail["message"]
    assert not rig.calls and rig.service.cache.stats()["entries"] == 0



async def test_stream_setup_capacity_error_uses_one_scope_and_error_event(provider_api, monkeypatch):
    from unittest.mock import AsyncMock
    from api import ask
    from inference.context_budget import InputBudgetError
    from inference.request_context import get_request_context

    failure = InputBudgetError("synthetic private input must not appear in API output")
    monkeypatch.setattr(ask, "get_grounded_answer_service", AsyncMock(side_effect=failure))
    response = await provider_api.client.post("/api/ask/stream", json=request().model_dump())
    emitted = events(response)
    assert response.status_code == 200 and [item["type"] for item in emitted] == ["error"]
    assert emitted[0]["kind"] == "context_budget_exceeded"
    assert str(failure) not in response.text
    assert provider_api.state.scope_exits == [None] and get_request_context() is None
    assert not provider_api.state.rig.calls
