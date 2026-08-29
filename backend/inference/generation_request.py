"""Shared character-generation request construction.

Production and evaluation callers provide their own model and retrieval
adapters, while this module owns the prompt, conversation, grounding, and
generation-parameter contract seen by the model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from inference.prompt_policy import (
    PROMPT_POLICY_VERSION,
    build_grounded_user_message,
    compose_system_prompt,
    sanitize_speaker_label,
)

if TYPE_CHECKING:  # pragma: no cover - 仅类型注解使用，避免运行时反向依赖
    from character.models import CompiledCharacterContext


Message = dict[str, str]
RetrievalStatus = Literal["not_requested", "ok", "abstained", "error"]

DEFAULT_CONTEXT_WINDOW_TOKENS = 24576
CONTEXT_SAFETY_MARGIN_TOKENS = 512

MEMORY_ATTRIBUTION_POLICY = (
    "长期记忆参考中的‘用户’始终指当前对话者，不是角色自身。"
    "当对话者用第一人称询问自己的历史信息时，回答必须用第二人称‘你’归属这些事实，"
    "不得把用户的姓名、偏好、目标或经历说成角色的第一人称事实。"
)


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
    evidence_max_chars: int = 6000
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    apply_prompt_policy: bool = True
    # 可选的角色上下文：None 时生成行为与旧链路完全一致。
    character_context: "CompiledCharacterContext | None" = None


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


def _estimated_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate for mixed Chinese/ASCII text."""
    non_ascii = sum(1 for char in text if ord(char) > 127)
    ascii_chars = len(text) - non_ascii
    return non_ascii + (ascii_chars + 3) // 4


def _trim_history_to_budget(
    history: list[Message],
    *,
    fixed_messages: Sequence[Message],
    context_window_tokens: int,
    max_output_tokens: int,
) -> list[Message]:
    """Keep the newest complete history messages within the input budget."""
    fixed_tokens = sum(_estimated_tokens(item["content"]) + 4 for item in fixed_messages)
    available = max(
        0,
        int(context_window_tokens)
        - int(max_output_tokens)
        - CONTEXT_SAFETY_MARGIN_TOKENS
        - fixed_tokens,
    )
    kept: list[Message] = []
    used = 0
    for item in reversed(history):
        cost = _estimated_tokens(item["content"]) + 4
        if used + cost > available:
            break
        kept.append(item)
        used += cost
    kept.reverse()
    return kept


def _system_prompt(request: GenerationRequest) -> str:
    if request.apply_prompt_policy:
        persona = request.persona_prompt.strip()
        context = request.character_context
        dynamic_context = ""
        if context is not None:
            # 人物已有现成提示词（如月社妃 Prompt v3）时不再拼接
            # profile_context，避免人物规则重复；只有没有现成提示词的
            # 人物才使用结构化画像作为替代。
            if not persona:
                persona = context.profile_context
            dynamic_context = context.dynamic_context
            if context.reference_context:
                dynamic_context = "\n\n".join(
                    part for part in (dynamic_context, MEMORY_ATTRIBUTION_POLICY) if part
                )
        prompt = compose_system_prompt(
            persona,
            include_rag=request.retrieval.has_evidence,
            dynamic_context=dynamic_context,
        )
    else:
        prompt = request.persona_prompt.strip()
    # 对话者昵称（senderName，用户可控）不进入系统提示词：
    # 净化只能删除结构字符，语义级注入内容仍会以系统区权威出现。
    # 3.3.0 起改由 build_grounded_user_message 放入用户消息的
    # <speaker_label> 不可信参考区。
    return prompt


def build_generation_request(request: GenerationRequest) -> GenerationPlan:
    """Build the one canonical model-facing message and parameter contract."""

    system_prompt = _system_prompt(request)
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    current_user_message = {
        "role": "user",
        "content": build_grounded_user_message(
            request.message,
            request.retrieval.evidence if request.retrieval.has_evidence else "",
            max_chars=request.evidence_max_chars,
            # 长期记忆只进入用户消息的不可信参考区，绝不进入系统提示词。
            memory_context=(
                request.character_context.reference_context
                if request.character_context is not None
                else ""
            ),
            # 对话者昵称（用户可控）同样只进不可信参考区。
            speaker=sanitize_speaker_label(request.interlocutor),
        ),
    }
    history = _trim_history_to_budget(
        _conversation_history(request.history),
        fixed_messages=(*messages, current_user_message),
        context_window_tokens=request.context_window_tokens,
        max_output_tokens=request.max_tokens,
    )
    messages.extend(history)
    messages.append(current_user_message)

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
