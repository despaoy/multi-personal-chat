"""Application working budgets selected from the actual inference provider."""

import os
from dataclasses import dataclass

from inference.context_budget import ReviewContextBudget


@dataclass(frozen=True)
class ProviderContextBudget:
    window_tokens: int
    history_limit: int = 24
    history_max_chars: int = 16000
    source_max_chars: int = 2400
    review: ReviewContextBudget | None = None
    reference_max_chars: int | None = None
    reference_observation_semantics: bool = False


def get_provider_context_budget(manager=None, *, env=None):
    """An explicit cloud working window never changes local serving limits.

    Without a configured cloud allowance use 8192, not an assumed theoretical
    vendor maximum. Local callers retain their existing history/review limits.
    """
    if manager is None:
        from inference.model_manager import get_model_manager

        manager = get_model_manager()
    source = os.environ if env is None else env
    if manager._current_provider.value == "openai_compat":
        window = int(source.get("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "8192"))
        if not 8192 <= window <= 1000000:
            raise ValueError("Cloud working context window must be between 8192 and 1000000")
        history_limit = max(24, min(256, window // 512))
        return ProviderContextBudget(
            window_tokens=window,
            history_limit=history_limit,
            history_max_chars=window,
            source_max_chars=window // 4,
            reference_max_chars=window // 2,
            reference_observation_semantics=True,
            review=ReviewContextBudget(window, history_messages=2 * history_limit),
        )
    window = max(1024, int(source.get("VLLM_MAX_MODEL_LEN", "8192")))
    return ProviderContextBudget(window_tokens=window)
