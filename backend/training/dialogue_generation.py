"""Validate a complete generated batch before converting its message format."""

import json


class DialogueBatchError(ValueError):
    """A safe model-output error; generated text is never part of the message."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"Invalid generated dialogue batch: {reason}")


def _unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DialogueBatchError("duplicate_field")
        value[key] = item
    return value


def parse_dialogue_batch(raw, expected_count, target_turns) -> list[list[dict[str, str]]]:
    """Require the requested count and turns; preserve every accepted content string."""
    if not isinstance(raw, str):
        raise DialogueBatchError("non_text")
    try:
        dialogues = json.loads(raw, object_pairs_hook=_unique_fields)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DialogueBatchError("invalid_json") from exc
    if not isinstance(dialogues, list) or len(dialogues) != expected_count:
        raise DialogueBatchError("wrong_batch_size")
    for dialogue in dialogues:
        if not isinstance(dialogue, dict) or set(dialogue) != {"conversations"}:
            raise DialogueBatchError("invalid_dialogue")
        messages = dialogue["conversations"]
        if not isinstance(messages, list) or len(messages) != 2 * target_turns:
            raise DialogueBatchError("wrong_turn_count")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise DialogueBatchError("invalid_message")
            if message["role"] != ("user" if index % 2 == 0 else "assistant"):
                raise DialogueBatchError("wrong_role_order")
            if not isinstance(message["content"], str) or not message["content"].strip():
                raise DialogueBatchError("invalid_content")
    return [
        [
            {"from": "human" if message["role"] == "user" else "gpt", "value": message["content"]}
            for message in dialogue["conversations"]
        ]
        for dialogue in dialogues
    ]
