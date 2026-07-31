"""Transport-agnostic orchestration for chat generation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from db.schemas import GenerateResponse, MessageRequest


GenerateHandler = Callable[
    [MessageRequest, dict[str, Any] | None],
    Awaitable[GenerateResponse],
]


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
        sanitize_message: Callable[[str], str],
        is_high_risk_prompt: Callable[[str], bool],
        security_response_factory: Callable[[], GenerateResponse],
        trace_id_factory: Callable[[], str],
    ) -> None:
        self._generate_handler = generate_handler
        self._inference_runtime = inference_runtime
        self._sanitize_message = sanitize_message
        self._is_high_risk_prompt = is_high_risk_prompt
        self._security_response_factory = security_response_factory
        self._trace_id_factory = trace_id_factory

    async def generate(
        self,
        request: MessageRequest,
        current_user: dict[str, Any] | None = None,
    ) -> GenerateResponse:
        """Normalize and validate request policy before model orchestration."""

        request.message = self._sanitize_message(request.message or "").strip()
        request.traceId = request.traceId or self._trace_id_factory()
        if self._is_high_risk_prompt(request.message):
            return self._security_response_factory()
        return await self._generate_handler(request, current_user)

    async def generate_queued(
        self,
        request: MessageRequest,
        current_user: dict[str, Any],
    ) -> GenerateResponse:
        """Apply admission control and execute one queued generation."""

        identity = str(
            current_user.get(
                "user_id",
                current_user.get("username", "unknown"),
            )
        )
        session_id = request.sessionId or f"manual:{identity}"
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
