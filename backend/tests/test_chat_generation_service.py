"""Unit tests for the injectable chat generation use-case boundary."""

from __future__ import annotations

from typing import Any

import pytest

from db.schemas import GenerateResponse, MessageRequest
from services.chat_generation import ChatGenerationService


class _FakeInferenceRuntime:
    def __init__(self) -> None:
        self.priority_calls: list[tuple[str, str]] = []
        self.rate_limit_calls: list[tuple[str, str, str]] = []
        self.submit_calls: list[tuple[str, int]] = []

    def priority_for(self, platform: str, session_type: str) -> int:
        self.priority_calls.append((platform, session_type))
        return 7

    async def check_rate_limits(
        self,
        platform: str,
        session_id: str,
        identity: str,
    ) -> None:
        self.rate_limit_calls.append((platform, session_id, identity))

    async def submit(self, job, *, session_id: str, priority: int):
        self.submit_calls.append((session_id, priority))
        return await job()


def _make_service(handler, runtime=None, *, high_risk=lambda _message: False):
    return ChatGenerationService(
        generate_handler=handler,
        inference_runtime=runtime or _FakeInferenceRuntime(),
        sanitize_message=lambda message: message.replace("\x00", ""),
        is_high_risk_prompt=high_risk,
        security_response_factory=lambda: GenerateResponse(
            reply="blocked",
            model="security-policy",
            costTime=0.0,
        ),
        trace_id_factory=lambda: "trace-generated",
    )


@pytest.mark.asyncio
async def test_generate_normalizes_request_and_delegates_to_injected_handler():
    captured: dict[str, Any] = {}

    async def handler(request, current_user):
        captured["message"] = request.message
        captured["trace_id"] = request.traceId
        captured["current_user"] = current_user
        return GenerateResponse(reply="ok", model="fake", costTime=0.01)

    service = _make_service(handler)
    request = MessageRequest(message=" \x00hello ")
    current_user = {"user_id": 42}

    response = await service.generate(request, current_user)

    assert response.reply == "ok"
    assert captured == {
        "message": "hello",
        "trace_id": "trace-generated",
        "current_user": current_user,
    }


@pytest.mark.asyncio
async def test_generate_short_circuits_high_risk_prompt():
    handler_called = False

    async def handler(_request, _current_user):
        nonlocal handler_called
        handler_called = True
        raise AssertionError("high-risk prompt must not reach the model handler")

    service = _make_service(handler, high_risk=lambda message: message == "secret")

    response = await service.generate(MessageRequest(message=" secret "))

    assert response.model == "security-policy"
    assert handler_called is False


@pytest.mark.asyncio
async def test_generate_short_circuits_high_risk_history_user_message():
    handler_called = False

    async def handler(_request, _current_user):
        nonlocal handler_called
        handler_called = True
        raise AssertionError("high-risk history must not reach the model handler")

    service = _make_service(handler, high_risk=lambda message: message == "secret")
    request = MessageRequest(
        message="continue",
        history=[
            {"role": "user", "content": "secret"},
            {"role": "assistant", "content": "ok"},
        ],
    )

    response = await service.generate(request)

    assert response.model == "security-policy"
    assert handler_called is False


@pytest.mark.asyncio
async def test_generate_sanitizes_history_user_messages():
    captured: dict[str, Any] = {}

    async def handler(request, current_user):
        captured["history"] = request.history
        return GenerateResponse(reply="ok", model="fake", costTime=0.0)

    service = _make_service(handler)
    request = MessageRequest(
        message="hello",
        history=[
            {"role": "user", "content": " \x00hello "},
            {"role": "assistant", "content": "keep"},
        ],
    )

    await service.generate(request)

    assert captured["history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "keep"},
    ]


@pytest.mark.asyncio
async def test_generate_queued_owns_admission_control_and_session_identity():
    runtime = _FakeInferenceRuntime()

    async def handler(_request, _current_user):
        return GenerateResponse(reply="queued-ok", model="fake", costTime=0.01)

    service = _make_service(handler, runtime)
    request = MessageRequest(message="hello", sessionType="group")

    response = await service.generate_queued(request, {"user_id": 42})

    assert response.reply == "queued-ok"
    assert runtime.priority_calls == [("admin", "group")]
    assert runtime.rate_limit_calls == [("admin", "manual:42", "42")]
    assert runtime.submit_calls == [("manual:42", 7)]


@pytest.mark.asyncio
async def test_generate_reply_core_keeps_compatibility_entry_point(monkeypatch):
    from api import generate

    expected = GenerateResponse(reply="compat", model="fake", costTime=0.0)

    class _FakeService:
        async def generate(self, request, current_user, **kwargs):
            assert request.message == "hello"
            assert current_user == {"username": "integration"}
            return expected

    monkeypatch.setattr(
        generate,
        "get_chat_generation_service",
        lambda: _FakeService(),
    )

    actual = await generate.generate_reply_core(
        MessageRequest(message="hello"),
        {"username": "integration"},
    )

    assert actual is expected


@pytest.mark.asyncio
async def test_generate_endpoint_uses_injected_service():
    from api.generate import generate_reply

    expected = GenerateResponse(reply="injected", model="fake", costTime=0.0)
    captured: dict[str, Any] = {}

    class _FakeService:
        async def generate_queued(self, request, current_user):
            captured["request"] = request
            captured["current_user"] = current_user
            return expected

    request = MessageRequest(message="hello")
    current_user = {"user_id": 9}

    actual = await generate_reply(request, current_user, _FakeService())

    assert actual is expected
    assert captured == {"request": request, "current_user": current_user}
