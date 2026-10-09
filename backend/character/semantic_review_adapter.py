"""Default provider adapter for selective semantic state review.

This module intentionally knows nothing about the character generation
pipeline.  It calls the selected low-level provider directly so a semantic review can
never re-enter reply generation, RAG, memory, or persona assembly.  Parsing
remains the responsibility of ``SemanticStateEstimator``; failures propagate.
"""

from __future__ import annotations

import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from infra.environment import read_bool

SEMANTIC_REVIEW_ENABLED_ENV = "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED"
SEMANTIC_REVIEW_TIMEOUT_ENV = "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS"
# Real Qwen 7B/8B reviewer runs on the supported local/server transports
# complete in roughly 3.0-3.8 seconds.  Five seconds leaves bounded headroom
# while keeping the review latency bounded under load.
DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS = 5.0
MIN_SEMANTIC_REVIEW_TIMEOUT_SECONDS = 0.1
MAX_SEMANTIC_REVIEW_TIMEOUT_SECONDS = 30.0
# A complete state includes three signal maps plus scalar fields. Providers
# may serialize all allowed zero-score signals with whitespace; 384 tokens
# truncated an otherwise valid real response before conversation_phase.
# This is an output ceiling, not a minimum or an extra inference pass.
SEMANTIC_REVIEW_MAX_TOKENS = 768

class SemanticReviewClient(Protocol):
    """Small structural contract for local and cloud review clients."""

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
        enabled = read_bool(source, SEMANTIC_REVIEW_ENABLED_ENV)
        timeout = _parse_timeout(source.get(SEMANTIC_REVIEW_TIMEOUT_ENV))
        return cls(enabled=enabled, timeout_seconds=timeout)


class VLLMSemanticReviewer:
    """Legacy public name for a reviewer using the selected low-level provider."""

    def __init__(self, client_factory: SemanticReviewClientFactory | None = None) -> None:
        self._client_factory = client_factory or _default_vllm_client_factory

    async def __call__(self, messages: Sequence[Mapping[str, str]]) -> object:
        request_messages = _copy_messages(messages)
        client = await self._client_factory()
        if client is None:
            raise RuntimeError("semantic review client is unavailable")
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
    # Keep provider resolution lazy: this adapter is imported during character
    # service startup, before a reviewer request needs its selected client.
    from inference.review_client import get_context_review_client

    return await get_context_review_client()


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
    except (TypeError, ValueError) as exc:
        raise ValueError("semantic review timeout must be a number") from exc
    if isinstance(raw, bool) or not math.isfinite(timeout) or not MIN_SEMANTIC_REVIEW_TIMEOUT_SECONDS <= timeout <= MAX_SEMANTIC_REVIEW_TIMEOUT_SECONDS:
        raise ValueError("semantic review timeout must be between 0.1 and 30 seconds")
    return timeout



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
