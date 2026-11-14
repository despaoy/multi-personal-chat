"""Transport-agnostic orchestration for chat generation."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from character.models import build_user_scope
from db.schemas import GenerateResponse, MessageRequest

GenerateHandler = Callable[
    ...,
    Awaitable[GenerateResponse],
]


def chat_user_identity(user: dict[str, Any]) -> str:
    """Resolve the authenticated owner consistently for chat and its source records."""
    for key in ("id", "user_id", "username"):
        value = user.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


class InferenceRuntime(Protocol):
    """Minimal runtime contract required by the chat use case."""

    def priority_for(self, platform: str, session_type: str) -> int: ...

    async def check_rate_limits(
        self,
        platform: str,
        session_id: str,
        identity: str,
    ) -> None: ...

    async def submit(
        self,
        job: Callable[[], Awaitable[GenerateResponse]],
        *,
        session_id: str,
        priority: int,
    ) -> GenerateResponse: ...


class ChatGenerationService:
    """Coordinate chat request policy independently from HTTP transport.

    The concrete model/RAG implementation is injected as ``generate_handler``.
    This keeps the use-case entry point replaceable and unit-testable while the
    existing provider implementation can be migrated incrementally.
    """

    def __init__(
        self,
        *,
        generate_handler: GenerateHandler,
        inference_runtime: InferenceRuntime,
        trace_id_factory: Callable[[], str],
        validate_request: Callable[[MessageRequest, dict[str, Any] | None], Awaitable[None]] | None = None,
    ) -> None:
        self._generate_handler = generate_handler
        self._inference_runtime = inference_runtime
        self._trace_id_factory = trace_id_factory
        self._validate_request = validate_request

    async def generate(
        self,
        request: MessageRequest,
        current_user: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerateResponse:
        """Apply policy to accepted text without rewriting the conversation."""

        request.traceId = request.traceId or self._trace_id_factory()
        if request.branchId:
            request.history = []  # Server-owned branch history only.
        if self._validate_request is not None:
            await self._validate_request(request, current_user)
        return await self._generate_handler(request, current_user, **kwargs)

    async def generate_queued(
        self,
        request: MessageRequest,
        current_user: dict[str, Any],
    ) -> GenerateResponse:
        """Apply admission control and execute one queued generation."""

        identity = chat_user_identity(current_user)
        session_key = ("manual", identity, request.sessionId)
        if request.branchId:
            session_key = ("narrative", identity, request.branchId)
        elif request.platform == "web" and request.adapter == "web-character":
            scope = build_user_scope(
                request.platform,
                request.adapter,
                identity,
                request.conversationId or request.sessionId,
                request.conversationType or request.sessionType,
            )
            session_key = ("character", request.characterId, *scope.memory_scope_key)
        session_id = json.dumps(session_key, separators=(",", ":"))
        priority = self._inference_runtime.priority_for("admin", request.sessionType)
        await self._inference_runtime.check_rate_limits(
            "admin",
            session_id,
            identity,
        )
        return await self._inference_runtime.submit(
            lambda: self.generate(request, current_user),
            session_id=session_id,
            priority=priority,
        )
