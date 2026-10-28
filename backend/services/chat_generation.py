"""Transport-agnostic orchestration for chat generation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from db.schemas import GenerateResponse, MessageRequest

GenerateHandler = Callable[
    ...,
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

        user_id = current_user.get("user_id")
        identity = str(user_id if user_id is not None else current_user["username"])
        session_id = request.sessionId or f"manual:{identity}"
        if request.branchId:
            session_id = f"narrative:{identity}:{request.branchId}"
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
