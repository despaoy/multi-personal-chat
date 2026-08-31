"""Default provider adapter for selective semantic state review.

This module intentionally knows nothing about the character generation
pipeline.  It calls the shared vLLM client directly so a semantic review can
never re-enter reply generation, RAG, memory, or persona assembly.  Parsing
and fail-closed recovery remain the responsibility of
``SemanticStateEstimator``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

SEMANTIC_REVIEW_ENABLED_ENV = "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED"
SEMANTIC_REVIEW_TIMEOUT_ENV = "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS"
# Real Qwen 7B/8B reviewer runs on the supported local/server transports
# complete in roughly 3.0-3.8 seconds.  Five seconds leaves bounded headroom
# while preserving fail-closed recovery under load.
DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS = 5.0
MIN_SEMANTIC_REVIEW_TIMEOUT_SECONDS = 0.1
MAX_SEMANTIC_REVIEW_TIMEOUT_SECONDS = 30.0
SEMANTIC_REVIEW_MAX_TOKENS = 384

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class SemanticReviewClient(Protocol):
    """Small structural contract implemented by :class:`VLLMClient`."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        lora_name: str | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
        enable_thinking: bool,
    ) -> object: ...


SemanticReviewClientFactory = Callable[[], Awaitable[SemanticReviewClient | None]]


@dataclass(frozen=True, slots=True)
class SemanticReviewSettings:
    """Feature-switch and latency budget read from the process environment."""

    enabled: bool = False
    timeout_seconds: float = DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, object] | None = None) -> SemanticReviewSettings:
        source = os.environ if env is None else env
        enabled = str(source.get(SEMANTIC_REVIEW_ENABLED_ENV, "false")).strip().lower() in _TRUE_VALUES
        timeout = _parse_timeout(source.get(SEMANTIC_REVIEW_TIMEOUT_ENV))
        return cls(enabled=enabled, timeout_seconds=timeout)


class VLLMSemanticReviewer:
    """Callable reviewer that uses only the shared low-level vLLM client."""

    def __init__(self, client_factory: SemanticReviewClientFactory | None = None) -> None:
        self._client_factory = client_factory or _default_vllm_client_factory

    async def __call__(self, messages: Sequence[Mapping[str, str]]) -> object:
        request_messages = _copy_messages(messages)
        client = await self._client_factory()
        if client is None:
            # Do not convert this into a semantic answer.  The estimator owns
            # the fail-closed fallback to the original deterministic state.
            raise RuntimeError("semantic review vLLM client is unavailable")
        return await client.generate(
            messages=request_messages,
            lora_name=None,
            temperature=0.0,
            max_tokens=SEMANTIC_REVIEW_MAX_TOKENS,
            stream=False,
            enable_thinking=False,
        )


@dataclass(frozen=True, slots=True)
class SemanticReviewRuntime:
    """One-shot wiring result used by the character-context service."""

    reviewer: VLLMSemanticReviewer | None
    timeout_seconds: float


def create_default_semantic_reviewer(
    *,
    settings: SemanticReviewSettings | None = None,
    env: Mapping[str, object] | None = None,
    client_factory: SemanticReviewClientFactory | None = None,
) -> VLLMSemanticReviewer | None:
    """Create the opt-in reviewer without importing higher-level services."""

    resolved = settings or SemanticReviewSettings.from_env(env)
    if not resolved.enabled:
        return None
    return VLLMSemanticReviewer(client_factory=client_factory)


def create_default_semantic_review_runtime(
    *,
    env: Mapping[str, object] | None = None,
    client_factory: SemanticReviewClientFactory | None = None,
) -> SemanticReviewRuntime:
    """Read settings once and return the reviewer plus its timeout budget."""

    settings = SemanticReviewSettings.from_env(env)
    return SemanticReviewRuntime(
        reviewer=create_default_semantic_reviewer(settings=settings, client_factory=client_factory),
        timeout_seconds=settings.timeout_seconds,
    )


async def _default_vllm_client_factory() -> SemanticReviewClient:
    # Keep the dependency lazy: vllm_client imports application configuration
    # and infrastructure modules, while this adapter is imported by character
    # services during startup.
    from inference.vllm_client import get_vllm_client

    return await get_vllm_client()


def _copy_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("semantic review messages must be mappings")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("semantic review message has an invalid role or content")
        copied.append({"role": role, "content": content})
    if not copied:
        raise ValueError("semantic review messages cannot be empty")
    return copied


def _parse_timeout(raw: object) -> float:
    if raw is None:
        return DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS
    if not math.isfinite(timeout) or timeout <= 0.0:
        return DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS
    return min(MAX_SEMANTIC_REVIEW_TIMEOUT_SECONDS, max(MIN_SEMANTIC_REVIEW_TIMEOUT_SECONDS, timeout))


__all__ = [
    "DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS",
    "MAX_SEMANTIC_REVIEW_TIMEOUT_SECONDS",
    "MIN_SEMANTIC_REVIEW_TIMEOUT_SECONDS",
    "SEMANTIC_REVIEW_ENABLED_ENV",
    "SEMANTIC_REVIEW_MAX_TOKENS",
    "SEMANTIC_REVIEW_TIMEOUT_ENV",
    "SemanticReviewClient",
    "SemanticReviewClientFactory",
    "SemanticReviewRuntime",
    "SemanticReviewSettings",
    "VLLMSemanticReviewer",
    "create_default_semantic_review_runtime",
    "create_default_semantic_reviewer",
]
