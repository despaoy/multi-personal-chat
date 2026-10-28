"""Unit tests for the injectable chat generation use-case boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest
from services.chat_generation import ChatGenerationService

from db.schemas import GenerateResponse, MessageRequest


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


def _make_service(handler, runtime=None):
    return ChatGenerationService(
        generate_handler=handler,
        inference_runtime=runtime or _FakeInferenceRuntime(),
        trace_id_factory=lambda: "trace-generated",
    )


@pytest.mark.asyncio
async def test_generate_preserves_original_and_delegates_to_injected_handler():
    captured: dict[str, Any] = {}

    async def handler(request, current_user):
        captured["message"] = request.message
        captured["trace_id"] = request.traceId
        captured["current_user"] = current_user
        return GenerateResponse(reply="ok", model="fake", costTime=0.01)

    service = _make_service(handler)
    request = MessageRequest(message="    hello\n  ")
    current_user = {"user_id": 42}

    response = await service.generate(request, current_user)

    assert response.reply == "ok"
    assert captured == {
        "message": "    hello\n  ",
        "trace_id": "trace-generated",
        "current_user": current_user,
    }


@pytest.mark.asyncio
async def test_generate_preserves_original_history():
    captured: dict[str, Any] = {}

    async def handler(request, current_user):
        captured["history"] = request.history
        return GenerateResponse(reply="ok", model="fake", costTime=0.0)

    service = _make_service(handler)
    request = MessageRequest(
        message="hello",
        history=[
            {"role": "user", "content": "    hello\n  "},
            {"role": "assistant", "content": "keep"},
        ],
    )

    await service.generate(request)

    assert captured["history"] == [
        {"role": "user", "content": "    hello\n  "},
        {"role": "assistant", "content": "keep"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_user,identity",
    [
        ({"user_id": 42, "username": "ignored-name", "role": "user"}, "42"),
        ({"user_id": 0, "username": "ignored-name", "role": "user"}, "0"),
        ({"user_id": None, "username": "isolated-alice", "role": "user"}, "isolated-alice"),
        ({"user_id": None, "username": "isolated-bob", "role": "user"}, "isolated-bob"),
        ({"username": "legacy-user", "role": "user"}, "legacy-user"),
    ],
)
async def test_generate_queued_owns_admission_control_and_session_identity(current_user, identity):
    runtime = _FakeInferenceRuntime()

    async def handler(_request, _current_user):
        return GenerateResponse(reply="queued-ok", model="fake", costTime=0.01)

    service = _make_service(handler, runtime)
    request = MessageRequest(message="hello", sessionType="group")

    response = await service.generate_queued(request, current_user)

    assert response.reply == "queued-ok"
    assert runtime.priority_calls == [("admin", "group")]
    key = json.dumps(("manual", identity, ""), separators=(",", ":"))
    assert runtime.rate_limit_calls == [("admin", key, identity)]
    assert runtime.submit_calls == [(key, 7)]


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
        "_build_chat_generation_service",
        lambda _runtime, _character=None, *, message_db, **_resource_options: _FakeService(),
    )
    actual = await generate.generate_reply_core(
        MessageRequest(message="hello"), {"username": "integration"}, message_db=object()
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


async def test_management_tab_names_cannot_alias_narrative_or_character_queue_names():
    runtime = _FakeInferenceRuntime()

    async def handler(_request, _current_user):
        return GenerateResponse(reply="key-contract", model="fake", costTime=0.0)

    service = _make_service(handler, runtime)
    user = {"user_id": 42, "username": "synthetic-owner"}
    branch = MessageRequest(message="合成队列协议测试。", branchId="synthetic-branch")
    await service.generate_queued(branch, user)
    branch_key = runtime.submit_calls[-1][0]
    await service.generate_queued(MessageRequest(message="合成管理请求。", sessionId=branch_key), user)
    assert runtime.submit_calls[-1][0] != branch_key
    character = MessageRequest(
        message="合成角色队列协议。",
        platform="web",
        adapter="web-character",
        senderId="42",
        conversationId="tab",
        characterId="synthetic-character",
    )
    await service.generate_queued(character, user)
    character_key = runtime.submit_calls[-1][0]
    await service.generate_queued(MessageRequest(message="合成管理请求。", sessionId=character_key), user)
    assert runtime.submit_calls[-1][0] != character_key
    assert json.loads(branch_key) == ["narrative", "42", "synthetic-branch"]
    assert json.loads(character_key) == ["character", "synthetic-character", "web", "web-character", "private", "42"]
