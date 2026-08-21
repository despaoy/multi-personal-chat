"""Canonical chat-record normalization and response-only token labeling.

The training stack accepts several historical dataset shapes.  This module
converts them into one message contract and labels every assistant turn while
leaving system and user tokens out of the loss.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from inference.prompt_policy import (
    build_grounded_user_message,
    sanitize_speaker_label,
)
from training.speaker_contract import (
    parse_speaker_contract,
    resolve_message_speaker,
)


SUPPORTED_ROLES = {"system", "user", "assistant"}


def _attach_speaker_boundaries(messages: list[dict[str, str]]) -> None:
    """Use the same untrusted speaker boundary as production inference."""
    for message in messages:
        speaker = message.pop("speaker_label", None)
        if message["role"] != "user" or not speaker:
            continue
        sanitized = sanitize_speaker_label(speaker)
        if not sanitized:
            raise ValueError("speaker label contains no safe characters")
        message["content"] = build_grounded_user_message(
            message["content"],
            "",
            max_chars=0,
            speaker=sanitized,
        )


def normalize_chat_record(
    record: Mapping[str, Any],
    *,
    default_system_prompt: str,
    system_prompt_policy: str = "preserve",
) -> list[dict[str, str]]:
    """Return one validated OpenAI-style message sequence."""

    speaker_contract = parse_speaker_contract(record)

    if isinstance(record.get("messages"), list):
        raw_messages: Iterable[Mapping[str, Any]] = record["messages"]
    elif isinstance(record.get("conversations"), list):
        raw_messages = record["conversations"]
    elif "user_question" in record and "agent_response" in record:
        raw_messages = (
            {"role": "user", "content": record["user_question"]},
            {"role": "assistant", "content": record["agent_response"]},
        )
    else:
        raise ValueError(
            "chat record must contain messages, conversations, or "
            "user_question/agent_response"
        )

    messages: list[dict[str, str]] = []
    for index, raw in enumerate(raw_messages):
        role = raw.get("role") or raw.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(str(role), str(role))
        content = raw.get("content") if "content" in raw else raw.get("value")
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"unsupported role at message {index}: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"empty content at message {index}")
        message = {"role": role, "content": content}
        speaker = resolve_message_speaker(
            raw,
            role=role,
            contract=speaker_contract,
        )
        if speaker:
            message["speaker_label"] = speaker
        messages.append(message)

    system_positions = [index for index, message in enumerate(messages) if message["role"] == "system"]
    if any(index != 0 for index in system_positions):
        raise ValueError("system message is only allowed at the beginning of a conversation")

    if system_prompt_policy not in {"preserve", "replace", "require_match"}:
        raise ValueError(f"unsupported system prompt policy: {system_prompt_policy!r}")
    explicit_system = record.get("system")
    embedded_systems = [message["content"] for message in messages if message["role"] == "system"]
    if isinstance(explicit_system, str) and explicit_system.strip():
        embedded_systems.append(explicit_system)
    if len(embedded_systems) > 1:
        raise ValueError("chat record contains multiple system prompts")

    messages = [message for message in messages if message["role"] != "system"]
    configured_prompt = default_system_prompt.strip()
    if system_prompt_policy == "replace":
        if not configured_prompt:
            raise ValueError("replace system prompt policy requires a configured prompt")
        messages.insert(0, {"role": "system", "content": configured_prompt})
    elif system_prompt_policy == "require_match":
        if not configured_prompt:
            raise ValueError("require_match system prompt policy requires a configured prompt")
        if embedded_systems and embedded_systems[0].strip() != configured_prompt:
            raise ValueError("record system prompt does not match configured prompt")
        messages.insert(0, {"role": "system", "content": configured_prompt})
    elif embedded_systems:
        messages.insert(0, {"role": "system", "content": embedded_systems[0]})
    elif configured_prompt:
        messages.insert(0, {"role": "system", "content": configured_prompt})

    _attach_speaker_boundaries(messages)

    conversational = [message["role"] for message in messages if message["role"] != "system"]
    if not conversational or "assistant" not in conversational:
        raise ValueError("chat record must contain at least one assistant response")
    for previous, current in zip(conversational, conversational[1:]):
        if previous == current:
            raise ValueError(f"adjacent {current!r} messages must be merged before training")
    return messages


def _render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    use_chat_template: bool,
) -> str:
    if not use_chat_template:
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _content_spans(
    rendered: str,
    messages: Sequence[Mapping[str, str]],
) -> list[tuple[str, int, int]]:
    """Locate message bodies in rendered order without relying on template masks."""

    cursor = 0
    spans: list[tuple[str, int, int]] = []
    for index, message in enumerate(messages):
        content = message["content"]
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError(
                f"chat template changed message content at index {index}; "
                "cannot build trustworthy assistant labels"
            )
        end = start + len(content)
        spans.append((message["role"], start, end))
        cursor = end
    return spans


def _truncate_labeled_sequence(
    input_ids: list[int],
    labels: list[int],
    attention_mask: list[int],
    *,
    max_length: int,
    direction: str,
) -> tuple[list[int], list[int], list[int]]:
    if len(input_ids) <= max_length:
        return input_ids, labels, attention_mask
    if direction not in {"left", "right"}:
        raise ValueError("truncation direction must be 'left' or 'right'")

    start = len(input_ids) - max_length if direction == "left" else 0
    end = start + max_length
    sliced = (input_ids[start:end], labels[start:end], attention_mask[start:end])
    if any(label != -100 for label in sliced[1]):
        return sliced

    supervised = [index for index, label in enumerate(labels) if label != -100]
    if not supervised:
        raise ValueError("sample has no supervised assistant tokens")
    end = min(len(input_ids), supervised[-1] + 1)
    start = max(0, end - max_length)
    return input_ids[start:end], labels[start:end], attention_mask[start:end]


def tokenize_assistant_turns(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_length: int,
    truncation_direction: str = "left",
    use_chat_template: bool = True,
    assistant_supervision: str = "all",
) -> dict[str, list[int]]:
    """Tokenize once and supervise selected assistant bodies and end markers."""

    if assistant_supervision not in {"all", "last"}:
        raise ValueError(
            "assistant supervision must be 'all' or 'last', got "
            f"{assistant_supervision!r}"
        )

    rendered = _render_chat(
        tokenizer,
        messages,
        use_chat_template=use_chat_template,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    labels = [-100] * len(input_ids)
    spans = _content_spans(rendered, messages)

    assistant_spans = [span for span in spans if span[0] == "assistant"]
    if assistant_supervision == "last":
        assistant_spans = assistant_spans[-1:]

    assistant_token_ranges: list[tuple[int, int]] = []
    for _, char_start, char_end in assistant_spans:
        token_indexes = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end and end > start
        ]
        if not token_indexes:
            raise ValueError("assistant response produced no labelable tokens")
        for index in token_indexes:
            labels[index] = input_ids[index]
        assistant_token_ranges.append((token_indexes[0], token_indexes[-1]))

    end_token_ids = {
        token_id
        for token in ("<|im_end|>", "<|eot_id|>")
        if (token_id := tokenizer.convert_tokens_to_ids(token)) is not None
        and token_id != getattr(tokenizer, "unk_token_id", None)
    }
    for _, last_index in assistant_token_ranges:
        for index in range(last_index + 1, len(input_ids)):
            if input_ids[index] in end_token_ids:
                labels[index] = input_ids[index]
                break
            if offsets[index][1] > offsets[last_index][1] and offsets[index][0] > offsets[last_index][1]:
                break

    attention_mask = [1] * len(input_ids)
    input_ids, labels, attention_mask = _truncate_labeled_sequence(
        input_ids,
        labels,
        attention_mask,
        max_length=max_length,
        direction=truncation_direction,
    )
    if not any(label != -100 for label in labels):
        raise ValueError("sample has no supervised assistant tokens after truncation")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def pack_tokenized_records(
    records: Iterable[Mapping[str, Sequence[int]]],
    *,
    max_length: int,
) -> list[dict[str, list[int]]]:
    """Pack pre-tokenized samples without discarding response-only labels."""

    buffer = {"input_ids": [], "labels": [], "attention_mask": []}
    packed: list[dict[str, list[int]]] = []
    for record in records:
        lengths = {len(record[key]) for key in buffer}
        if len(lengths) != 1:
            raise ValueError("input_ids, labels, and attention_mask lengths must match")
        for key in buffer:
            buffer[key].extend(int(value) for value in record[key])
        while len(buffer["input_ids"]) >= max_length:
            chunk = {key: values[:max_length] for key, values in buffer.items()}
            if any(label != -100 for label in chunk["labels"]):
                packed.append(chunk)
            for key in buffer:
                del buffer[key][:max_length]
    if buffer["input_ids"] and any(label != -100 for label in buffer["labels"]):
        packed.append({key: list(values) for key, values in buffer.items()})
    if not packed:
        raise ValueError("packing produced no supervised chunks")
    return packed
