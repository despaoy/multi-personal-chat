import pytest

from training.chat_dataset import (
    normalize_chat_record,
    pack_tokenized_records,
    tokenize_assistant_turns,
)


class TinyChatTokenizer:
    """Small deterministic tokenizer for response-mask unit tests."""

    unk_token_id = -1
    end_token_id = 100_000

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, **_):
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<{message['role']}>{message['content']}<|im_end|>"
            for message in messages
        )

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        ids = []
        offsets = []
        cursor = 0
        marker = "<|im_end|>"
        while cursor < len(text):
            if text.startswith(marker, cursor):
                ids.append(self.end_token_id)
                offsets.append((cursor, cursor + len(marker)))
                cursor += len(marker)
            else:
                ids.append(ord(text[cursor]))
                offsets.append((cursor, cursor + 1))
                cursor += 1
        return {"input_ids": ids, "offset_mapping": offsets}

    def convert_tokens_to_ids(self, token):
        return self.end_token_id if token == "<|im_end|>" else self.unk_token_id

    def decode(self, token_ids, *, skip_special_tokens=True):
        return "".join(
            chr(token_id)
            for token_id in token_ids
            if not skip_special_tokens or token_id != self.end_token_id
        )


def test_normalize_supports_messages_sharegpt_and_legacy_records():
    messages = normalize_chat_record(
        {
            "messages": [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "回答"},
            ]
        },
        default_system_prompt="系统",
    )
    sharegpt = normalize_chat_record(
        {
            "system": "系统",
            "conversations": [
                {"from": "human", "value": "问题"},
                {"from": "gpt", "value": "回答"},
            ],
        },
        default_system_prompt="ignored",
    )
    legacy = normalize_chat_record(
        {"user_question": "问题", "agent_response": "回答"},
        default_system_prompt="系统",
    )
    assert messages == sharegpt == legacy


def test_normalize_rejects_adjacent_same_role_messages():
    with pytest.raises(ValueError, match="must be merged"):
        normalize_chat_record(
            {
                "messages": [
                    {"role": "user", "content": "甲"},
                    {"role": "user", "content": "乙"},
                    {"role": "assistant", "content": "丙"},
                ]
            },
            default_system_prompt="系统",
        )


def test_packing_preserves_response_only_labels():
    records = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "labels": [-100, -100, 6], "attention_mask": [1, 1, 1]},
    ]
    packed = pack_tokenized_records(records, max_length=4)
    assert packed == [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, -100], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 6], "labels": [-100, 6], "attention_mask": [1, 1]},
    ]


def test_qwen3_labels_every_assistant_turn_and_masks_user_tokens():
    tokenizer = TinyChatTokenizer()
    messages = [
        {"role": "system", "content": "系统规则"},
        {"role": "user", "content": "用户标记甲"},
        {"role": "assistant", "content": "助手标记甲"},
        {"role": "user", "content": "用户标记乙"},
        {"role": "assistant", "content": "助手标记乙"},
    ]
    tokenized = tokenize_assistant_turns(
        tokenizer,
        messages,
        max_length=512,
    )
    supervised_ids = [
        token_id
        for token_id, label in zip(tokenized["input_ids"], tokenized["labels"])
        if label != -100
    ]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=True)
    assert "助手标记甲" in supervised_text
    assert "助手标记乙" in supervised_text
    assert "用户标记甲" not in supervised_text
    assert "用户标记乙" not in supervised_text
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert sum(
        token_id == end_token_id
        for token_id, label in zip(tokenized["input_ids"], tokenized["labels"])
        if label != -100
    ) == 2


def test_qwen3_truncation_keeps_supervised_tokens():
    tokenizer = TinyChatTokenizer()
    tokenized = tokenize_assistant_turns(
        tokenizer,
        [
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "很长的问题" * 100},
            {"role": "assistant", "content": "最终回答"},
        ],
        max_length=32,
        truncation_direction="left",
    )
    assert len(tokenized["input_ids"]) <= 32
    assert any(label != -100 for label in tokenized["labels"])
