"""Shared serving-budget text counting with provider-specific offline data."""

import os
from dataclasses import dataclass

CONTEXT_SAFETY_MARGIN_TOKENS = 512
MIN_CONTEXT_WINDOW_TOKENS = 1024


class InputBudgetError(ValueError):
    """Required input cannot fit the configured budget without losing data."""

    http_status = 503

    def public_detail(self):
        return {
            "kind": "context_budget_exceeded", "stage": "context_budget", "reason": "capacity_exceeded",
            "message": "完整输入或输出预留超出当前处理预算，本轮无法完成。请缩小问题范围或检查预算配置。",
        }


def configured_context_window(*, env=None) -> int:
    """Read the actual local serving limit; never expand invalid configuration."""
    source = os.environ if env is None else env
    message = f"VLLM_MAX_MODEL_LEN must be an integer of at least {MIN_CONTEXT_WINDOW_TOKENS}"
    try:
        window = int(source.get("VLLM_MAX_MODEL_LEN", "8192"))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if window < MIN_CONTEXT_WINDOW_TOKENS:
        raise ValueError(message)
    return window


def estimated_tokens(text: str) -> int:
    from inference.request_context import get_request_context
    from inference.token_counting import serving_text_tokens

    context = get_request_context()
    return context.count_tokens(text) if context is not None else serving_text_tokens(text)


@dataclass(frozen=True)
class ReviewContextBudget:
    window_tokens: int
    history_messages: int = 128

    def __post_init__(self):
        if self.window_tokens < MIN_CONTEXT_WINDOW_TOKENS or self.history_messages < 1:
            raise ValueError('Invalid review context budget')

    def fits(self, messages, output_tokens):
        return (sum(estimated_tokens(m['content']) + 4 for m in messages)
                + output_tokens + CONTEXT_SAFETY_MARGIN_TOKENS <= self.window_tokens)
