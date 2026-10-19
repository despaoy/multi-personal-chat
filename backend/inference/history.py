"""One text-message contract for external, stored and reviewed history."""

from collections.abc import Mapping, Sequence
from itertools import islice
import re

from inference.context_budget import InputBudgetError


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def is_message_text(value: object) -> bool:
    """Accept meaningful text unchanged, including indentation and line breaks."""
    return isinstance(value, str) and bool(value.strip()) and not _CONTROL_CHARS_RE.search(value)


class HistoryFormatError(ValueError):
    def __init__(self, field: str):
        self.field = field
        super().__init__("Invalid conversation history field: " + field)

    def public_detail(self) -> dict[str, str]:
        return {
            "stage": "conversation_history", "reason": "invalid_history", "field": self.field,
            "message": "对话历史格式错误，本轮未返回回复。请检查历史读取及消息结构后重试。",
        }


def validate_history(history: Sequence[Mapping[str, str]]) -> None:
    """Require valid role/text pairs without rewriting or discarding data."""
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise HistoryFormatError("history")
    for index, item in enumerate(history):
        field = f"history[{index}]"
        if not isinstance(item, Mapping):
            raise HistoryFormatError(field)
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant"):
            raise HistoryFormatError(field + ".role")
        if not is_message_text(content):
            raise HistoryFormatError(field + ".content")


def review_history(history: Sequence[Mapping[str, str]], *,
                  max_messages: int = 12,
                  max_chars: int = 6000) -> list[dict[str, str]]:
    """Keep a suffix of complete user-led turns or reject the review.

    A tail slice can detach a quotation or negation from its subject. The
    Message-count cuts must not leave an assistant claim without the user
    premise that qualified it. Legacy leading assistant-only history remains
    unchanged when it fits; this view cannot reconstruct missing upstream data.
    """
    validate_history(history)
    start = len(history)
    for index in reversed(range(len(history))):
        if index and history[index]["role"] != "user":
            continue
        if len(history) - index > max_messages:
            if start == len(history):
                raise InputBudgetError("latest turn exceeds history message budget")
            break
        start = index
    kept = [{"role": entry["role"], "content": entry["content"]} for entry in islice(history, start, None)]
    if sum(len(entry['content']) for entry in kept) > max_chars:
        raise InputBudgetError("recent history exceeds budget; must not be partially sliced")
    return kept
