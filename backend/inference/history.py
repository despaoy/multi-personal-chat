"""One text-message contract for external, stored and reviewed history."""

from collections.abc import Mapping, Sequence


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
        if not isinstance(content, str) or not content.strip():
            raise HistoryFormatError(field + ".content")
