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


def test_normalize_replace_policy_uses_only_configured_system_prompt():
    messages = normalize_chat_record(
        {
            "messages": [
                {"role": "system", "content": "旧短提示词"},
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "回答"},
            ]
        },
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )

    assert messages[0] == {"role": "system", "content": "完整 prompt v3"}
    assert sum(message["role"] == "system" for message in messages) == 1


def test_normalize_require_match_rejects_prompt_drift():
    with pytest.raises(ValueError, match="does not match"):
        normalize_chat_record(
            {
                "system": "旧短提示词",
                "conversations": [
                    {"from": "human", "value": "问题"},
                    {"from": "gpt", "value": "回答"},
                ],
            },
            default_system_prompt="完整 prompt v3",
            system_prompt_policy="require_match",
        )


def test_normalize_rejects_system_message_after_conversation_starts():
    with pytest.raises(ValueError, match="only allowed at the beginning"):
        normalize_chat_record(
            {
                "messages": [
                    {"role": "user", "content": "问题"},
                    {"role": "system", "content": "迟到的系统消息"},
                    {"role": "assistant", "content": "回答"},
                ]
            },
            default_system_prompt="完整 prompt v3",
            system_prompt_policy="replace",
        )


def test_normalize_injects_context_speaker_from_game_metadata():
    messages = normalize_chat_record(
        {
            "messages": [
                {"role": "user", "content": "你怎么了？"},
                {"role": "assistant", "content": "没什么。"},
            ],
            "metadata": {"context_speaker_label": "夜子"},
        },
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )

    assert messages[0] == {"role": "system", "content": "完整 prompt v3"}
    assert messages[1:] == [
        {
            "role": "user",
            "content": (
                '<speaker_label trust="untrusted" purpose="addressing_reference">\n'
                "当前对话者：夜子。\n"
                "</speaker_label>\n\n"
                "<user_query>\n"
                "你怎么了？\n"
                "</user_query>"
            ),
        },
        {"role": "assistant", "content": "没什么。"},
    ]


def test_normalize_wraps_each_user_turn_with_the_runtime_speaker_boundary():
    messages = normalize_chat_record(
        {
            "messages": [
                {"role": "user", "content": "第一问"},
                {"role": "assistant", "content": "第一答"},
                {"role": "user", "content": "第二问"},
                {"role": "assistant", "content": "第二答"},
            ],
            "metadata": {"context_speaker_label": "夜子"},
        },
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )

    assert messages[0]["content"] == "完整 prompt v3"
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    assert len(user_messages) == 2
    assert all('<speaker_label trust="untrusted"' in content for content in user_messages)
    assert user_messages[0].endswith("<user_query>\n第一问\n</user_query>")
    assert user_messages[1].endswith("<user_query>\n第二问\n</user_query>")


def test_message_speaker_overrides_record_speaker_per_user_turn():
    messages = normalize_chat_record(
        {
            "messages": [
                {"role": "user", "content": "第一问", "speaker_label": "夜子"},
                {"role": "assistant", "content": "第一答"},
                {"role": "user", "content": "第二问", "speaker_label": "琉璃"},
                {"role": "assistant", "content": "第二答"},
            ],
            "metadata": {
                "interlocutor_kind": "canonical_character",
                "interlocutor_label": "夜子",
            },
        },
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )

    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    assert "当前对话者：夜子。" in user_messages[0]
    assert "当前对话者：琉璃。" in user_messages[1]
    assert messages[0] == {"role": "system", "content": "完整 prompt v3"}
    assert all(set(message) == {"role", "content"} for message in messages)


def test_generic_user_contract_does_not_invent_a_named_speaker():
    messages = normalize_chat_record(
        {
            "messages": [
                {"role": "user", "content": "夜子今天很安静。"},
                {"role": "assistant", "content": "是呢。"},
            ],
            "metadata": {"interlocutor_kind": "generic_user"},
        },
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )

    assert messages[1] == {"role": "user", "content": "夜子今天很安静。"}


def test_canonical_character_contract_requires_a_label():
    with pytest.raises(ValueError, match="require interlocutor_label"):
        normalize_chat_record(
            {
                "messages": [
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "回答"},
                ],
                "metadata": {"interlocutor_kind": "canonical_character"},
            },
            default_system_prompt="完整 prompt v3",
        )


@pytest.mark.parametrize(
    "messages, metadata, error",
    [
        (
            [
                {"role": "user", "content": "问题", "speaker_label": "夜子"},
                {"role": "assistant", "content": "回答"},
            ],
            {"interlocutor_kind": "generic_user"},
            "must not contain named user turns",
        ),
        (
            [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "回答", "speaker_label": "妃"},
            ],
            {},
            "only allowed on user messages",
        ),
    ],
)
def test_speaker_contract_rejects_identity_conflicts(messages, metadata, error):
    with pytest.raises(ValueError, match=error):
        normalize_chat_record(
            {"messages": messages, "metadata": metadata},
            default_system_prompt="完整 prompt v3",
        )


def test_normalize_does_not_invent_context_speaker():
    messages = normalize_chat_record(
        {
            "messages": [
                {"role": "user", "content": "你好。"},
                {"role": "assistant", "content": "晚上好。"},
            ]
        },
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )

    assert messages[0]["content"] == "完整 prompt v3"


def test_context_speaker_does_not_change_supervised_assistant_tokens():
    tokenizer = TinyChatTokenizer()
    record = {
        "messages": [
            {"role": "user", "content": "你怎么了？"},
            {"role": "assistant", "content": "没什么。"},
        ]
    }
    without_speaker = normalize_chat_record(
        record,
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )
    with_speaker = normalize_chat_record(
        {**record, "metadata": {"context_speaker_label": "夜子"}},
        default_system_prompt="完整 prompt v3",
        system_prompt_policy="replace",
    )

    supervised = []
    for messages in (without_speaker, with_speaker):
        tokenized = tokenize_assistant_turns(tokenizer, messages, max_length=512)
        supervised.append(
            [
                token_id
                for token_id, label in zip(tokenized["input_ids"], tokenized["labels"])
                if label != -100
            ]
        )
    assert supervised[0] == supervised[1]


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


def test_qwen3_can_supervise_only_the_final_assistant_turn():
    tokenizer = TinyChatTokenizer()
    messages = [
        {"role": "user", "content": "历史问题"},
        {"role": "assistant", "content": "历史回答"},
        {"role": "user", "content": "当前问题"},
        {"role": "assistant", "content": "最终回答"},
    ]

    tokenized = tokenize_assistant_turns(
        tokenizer,
        messages,
        max_length=512,
        assistant_supervision="last",
    )
    supervised_ids = [
        token_id
        for token_id, label in zip(tokenized["input_ids"], tokenized["labels"])
        if label != -100
    ]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=True)
    assert "历史回答" not in supervised_text
    assert "最终回答" in supervised_text
    assert supervised_ids.count(tokenizer.end_token_id) == 1


def test_qwen3_rejects_unknown_assistant_supervision_mode():
    with pytest.raises(ValueError, match="assistant supervision"):
        tokenize_assistant_turns(
            TinyChatTokenizer(),
            [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}],
            max_length=512,
            assistant_supervision="final-ish",
        )


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
