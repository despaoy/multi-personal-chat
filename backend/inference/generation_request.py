"""Shared character-generation request construction.

Production and evaluation callers provide their own model and retrieval
adapters, while this module owns the prompt, conversation, grounding, and
generation-parameter contract seen by the model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from inference.prompt_policy import (
    PROMPT_POLICY_VERSION,
    build_grounded_user_message,
    compose_system_prompt,
)


Message = dict[str, str]
RetrievalStatus = Literal["not_requested", "ok", "abstained", "error"]


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieval state supplied by a production or benchmark adapter."""

    status: RetrievalStatus = "not_requested"
    evidence: str = ""
    documents: tuple[Mapping[str, Any], ...] = ()
    citations: tuple[Mapping[str, Any], ...] = ()
    confidence: float | None = None
    reason: str = ""

    @property
    def has_evidence(self) -> bool:
        return self.status == "ok" and bool(self.evidence.strip())


@dataclass(frozen=True)
class GenerationRequest:
    """Transport-neutral inputs that affect one character response."""

    message: str
    persona_prompt: str = ""
    interlocutor: str = ""
    history: Sequence[Mapping[str, str]] = ()
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    lora_name: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    enable_thinking: bool = False
    evidence_max_chars: int = 800
    apply_prompt_policy: bool = True


@dataclass(frozen=True)
class GenerationPlan:
    """The complete model-facing request built from ``GenerationRequest``."""

    messages: tuple[Message, ...]
    generation: Mapping[str, Any]
    prompt_policy_version: str
    lora_name: str | None
    retrieval: RetrievalResult

    @property
    def should_generate(self) -> bool:
        return self.retrieval.status not in {"abstained", "error"}


@dataclass(frozen=True)
class GenerationResult:
    reply: str
    plan: GenerationPlan


def _conversation_history(history: Sequence[Mapping[str, str]]) -> list[Message]:
    messages: list[Message] = []
    for item in history:
        role = str(item.get("role", ""))
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def _system_prompt(request: GenerationRequest) -> str:
    if request.apply_prompt_policy:
        prompt = compose_system_prompt(
            request.persona_prompt,
            include_rag=request.retrieval.has_evidence,
        )
    else:
        prompt = request.persona_prompt.strip()
    if request.interlocutor.strip():
        prompt = f"{prompt}\n\n当前对话者：{request.interlocutor.strip()}。".strip()
    return prompt


def build_generation_request(request: GenerationRequest) -> GenerationPlan:
    """Build the one canonical model-facing message and parameter contract."""

    system_prompt = _system_prompt(request)
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(_conversation_history(request.history))
    messages.append(
        {
            "role": "user",
            "content": build_grounded_user_message(
                request.message,
                request.retrieval.evidence if request.retrieval.has_evidence else "",
                max_chars=request.evidence_max_chars,
            ),
        }
    )

    temperature = (
        min(request.temperature, 0.5)
        if request.retrieval.has_evidence
        else request.temperature
    )
    generation = {
        "temperature": temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "frequency_penalty": request.frequency_penalty,
        "enable_thinking": request.enable_thinking,
    }
    return GenerationPlan(
        messages=tuple(messages),
        generation=generation,
        prompt_policy_version=PROMPT_POLICY_VERSION if request.apply_prompt_policy else "",
        lora_name=request.lora_name,
        retrieval=request.retrieval,
    )


async def generate_character_response(
    request: GenerationRequest,
    generate: Callable[..., Awaitable[str]],
) -> GenerationResult:
    """Build and execute one request with an injected model adapter."""

    plan = build_generation_request(request)
    if not plan.should_generate:
        raise RuntimeError(
            plan.retrieval.reason or f"retrieval status is {plan.retrieval.status}"
        )
    reply = await generate(
        messages=[dict(message) for message in plan.messages],
        lora_name=plan.lora_name,
        **dict(plan.generation),
    )
    return GenerationResult(reply=reply, plan=plan)
