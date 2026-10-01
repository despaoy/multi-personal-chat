"""Guarded final-response feedback on the exact server-returned archive row."""

from __future__ import annotations

from db.memory_source import source_scope


def reply_with_warning(reply: str, warning: str | None) -> str:
    return f"{reply}\n\n保存提示：{warning}" if warning else reply


def feedback_update_plan(receipt: dict, warning: str):
    """No lookup by client source ID/text and no cross-row fallback.

    The receipt comes from add_message, after authorized request normalization.
    Only a matching NULL-branch row whose original text is unchanged can move
    to the final public reply. Repeating the same patch cannot duplicate notice.
    """
    value = receipt.get("id")
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).isdigit() or int(value) < 1:
        raise ValueError("A trusted positive archive receipt is required")
    if receipt.get("branchId") is not None:
        raise ValueError("Branch feedback needs its authorized repository")
    fields = ("characterId", "platform", "adapter", "senderId", "conversationType", "conversationId")
    source_scope(*(receipt.get(name) for name in fields))
    match = (
        "platform",
        "adapter",
        "senderId",
        "conversationType",
        "conversationId",
        "characterId",
        "sessionId",
        "sessionType",
        "sourceMessageId",
        "traceId",
        "message",
        "reply",
    )
    if any(not isinstance(receipt.get(name), str) for name in match):
        raise ValueError("Exact original archive metadata is required")
    if not isinstance(warning, str) or not warning.strip() or len(warning) > 500:
        raise ValueError("A bounded server completion warning is required")
    params = {name: receipt[name] for name in match}
    params.update(id=int(value), final_reply=reply_with_warning(receipt["reply"], warning))
    conditions = ["id = :id", '"branchId" IS NULL']
    conditions += [f'"{name}" = :{name}' for name in match if name != "reply"]
    conditions.append("(reply = :reply OR reply = :final_reply)")
    return "UPDATE messages SET reply = :final_reply WHERE " + " AND ".join(conditions), params
