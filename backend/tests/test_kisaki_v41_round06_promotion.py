from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from training.chat_dataset import normalize_chat_record, tokenize_assistant_turns


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/promote_kisaki_v41_round06.py"
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"
ROUND06 = V4 / "augmentation_candidates/deepseek_user_simulation_round06"


def _module():
    spec = importlib.util.spec_from_file_location("promote_kisaki_v41_round06", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _CharacterTokenizer:
    unk_token_id = -1
    end_token_id = 100_000

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, **_):
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<{message['role']}>{message['content']}<|im_end|>" for message in messages
        )

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        ids, offsets = [], []
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


def test_round06_approval_builds_four_complete_sessions_not_prefixes():
    approval, records = _module().build_promoted_records()

    assert approval["promotion_contract"]["training_unit"] == "complete_five_turn_session"
    assert approval["promotion_contract"]["cumulative_prefix_records_allowed"] is False
    assert len(records) == 4
    assert len({record["id"] for record in records}) == 4
    assert all(len(record["messages"]) == 10 for record in records)
    assert all(
        [message["role"] for message in record["messages"]] == ["user", "assistant"] * 5
        for record in records
    )
    assert sum(
        message["role"] == "assistant"
        for record in records
        for message in record["messages"]
    ) == 20


def test_round06_promoted_records_are_the_only_new_training_units():
    train = _jsonl(V4 / "train.jsonl")
    auxiliary = _jsonl(V4 / "cleanup/candidate/technical_auxiliary.jsonl")
    formal = [
        record
        for record in train
        if record.get("metadata", {}).get("data_source")
        == "deepseek_user_simulation_v41_reviewed"
    ]
    moved = [
        record
        for record in auxiliary
        if record.get("metadata", {}).get("data_source")
        == "deepseek_user_simulation_v41_reviewed"
    ]
    promoted = formal + moved

    assert len(train) >= 730
    assert len(promoted) == 4
    assert len(formal) == 3
    assert [record["id"] for record in moved] == ["kisaki_v41_round06_coding_debug"]
    assert {record["metadata"]["turns"] for record in promoted} == {5}
    assert all("system" not in {message["role"] for message in record["messages"]} for record in promoted)
    assert not any("_turn_" in record["id"] for record in promoted)


def test_round06_has_no_protected_prompt_overlap():
    module = _module()
    _, records = module.build_promoted_records()
    validation = _jsonl(V4 / "validation.jsonl")
    gold = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3.json").read_text(encoding="utf-8")
    )["prompts"]
    candidates = [
        (record["id"], text) for record in records for text in module._user_texts(record)
    ]
    references = [
        (record["id"], text, "validation")
        for record in validation
        for text in module._user_texts(record)
    ] + [
        (record["id"], text, "gold_v3")
        for record in gold
        for text in module._user_texts(record)
    ]

    assert module._similarity_matches(candidates, references) == []


def test_round06_manifest_count_hash_and_current_reaudit_are_explicit():
    module = _module()
    manifest = json.loads((V4 / "canonical_dataset_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3_contamination_audit.json").read_text(
            encoding="utf-8"
        )
    )

    train = _jsonl(V4 / "train.jsonl")
    validation = _jsonl(V4 / "validation.jsonl")
    assert manifest["train"]["count"] == len(train)
    assert manifest["train"]["sha256"] == module._text_sha256(V4 / "train.jsonl")
    assert manifest["train"]["source_distribution"]["game_extraction_current_sft"] == 522
    assert manifest["train"]["source_distribution"]["llm_v4_reviewed_constructed"] == 150
    assert manifest["train"]["source_distribution"]["deepseek_user_simulation_v41_reviewed"] == 3
    assert audit["status"] == "clean"
    assert audit["frozen_reference_count"] == len(train) + len(validation)
    assert audit["frozen_train_count"] == len(train)
    assert audit["frozen_validation_count"] == len(validation)
    assert audit["frozen_train_sha256"] == manifest["train"]["sha256"]
    assert audit["frozen_validation_sha256"] == manifest["validation"]["sha256"]
    assert manifest["gold_v3"]["contamination_reaudit"]["status"] == "clean"


def test_round06_chat_normalization_supervises_all_assistant_turns():
    _, records = _module().build_promoted_records()
    tokenizer = _CharacterTokenizer()
    supervised_end_markers = 0

    for record in records:
        messages = normalize_chat_record(
            record,
            default_system_prompt="完整 prompt v3",
            system_prompt_policy="replace",
        )
        tokenized = tokenize_assistant_turns(tokenizer, messages, max_length=20_000)
        supervised_end_markers += sum(
            token_id == tokenizer.end_token_id and label != -100
            for token_id, label in zip(tokenized["input_ids"], tokenized["labels"])
        )

    assert supervised_end_markers == 20


def test_round06_promotion_is_idempotent_after_success():
    result = _module().promote()
    assert result["status"] == "already_promoted"
    manifest = json.loads((V4 / "canonical_dataset_manifest.json").read_text(encoding="utf-8"))
    assert result["train_count"] == manifest["train"]["count"]
    assert result["formal_record_count"] == 3
    assert result["auxiliary_record_count"] == 1
