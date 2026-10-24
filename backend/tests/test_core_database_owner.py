"""Admission and generation must use the same explicitly selected database."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from api import generate
from db.schemas import GenerateResponse, MessageRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["recorded", "conflict", "unavailable"])
async def test_core_binds_source_admission_and_generation_to_selected_database(monkeypatch, status):
    admission = Mock(return_value=status)
    if status == "unavailable":
        admission.side_effect = OSError("synthetic selected database failure")
    selected = SimpleNamespace(memory_source_admission=admission)
    global_admission = Mock(side_effect=AssertionError("global database must not be read"))
    monkeypatch.setattr(generate, "db", SimpleNamespace(memory_source_admission=global_admission))
    reply = GenerateResponse(reply="verified", model="synthetic", costTime=0)
    handler = AsyncMock(return_value=reply)
    monkeypatch.setattr(generate, "_generate_reply_impl", handler)
    # Rebuild the ordinary service so both implementations use this test's handler.
    monkeypatch.setattr(generate, "_chat_generation_service", generate._build_chat_generation_service(generate.inference_runtime))
    request = MessageRequest(message="完整来源消息", characterId="fiction-role", platform="web",
                             adapter="web-character", senderId="owner", userId="owner",
                             sessionId="conversation", conversationId="conversation", sourceMessageId="source-1")
    if status == "recorded":
        result = await generate.generate_reply_core(request, {"id": "owner"}, message_db=selected)
        assert result is reply
        handler.assert_awaited_once()
        assert handler.call_args.kwargs["message_db"] is selected
    else:
        with pytest.raises(HTTPException) as caught:
            await generate.generate_reply_core(request, {"id": "owner"}, message_db=selected)
        assert caught.value.status_code == (409 if status == "conflict" else 503)
        if status == "unavailable":
            assert isinstance(caught.value.__cause__, OSError)
        handler.assert_not_awaited()
    global_admission.assert_not_called()
    admission.assert_called_once()
    assert admission.call_args.args[:4] == ("fiction-role", "web", "web-character", "owner")
    assert admission.call_args.kwargs == {"source_message_id": "source-1", "body": request.message}
