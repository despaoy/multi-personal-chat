from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from training.chat_dataset import normalize_chat_record


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/apply_kisaki_v4_speaker_contract.py"
ACTIVE_TRAIN = (
    ROOT / "backend/data/character_dialogues/experiments/v4/train.jsonl"
)
ACTIVE_MANIFEST = (
    ROOT
    / "backend/data/character_dialogues/experiments/v4/canonical_dataset_manifest.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "apply_kisaki_v4_speaker_contract", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _game_record(*, with_context: bool = True):
    metadata = {
        "data_source": "game_extraction",
        "source_speaker_label": "妃",
    }
    if with_context:
        metadata["context_speaker_label"] = "夜子"
    return {
        "id": "game-1",
        "messages": [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ],
        "metadata": metadata,
    }


def _constructed_record():
    return {
        "id": "constructed-1",
        "messages": [
            {"role": "user", "content": "夜子今天很安静"},
            {"role": "assistant", "content": "是呢"},
        ],
        "metadata": {"data_source": "llm_v4_manual"},
    }


def _manifest(module, train_path: Path):
    return {
        "schema_version": 9,
        "status": "frozen_data_pending_gold",
        "train": {"count": 2, "sha256": module._text_sha256(train_path)},
        "validation": {"count": 70, "sha256": "unchanged-validation"},
        "freeze_blockers": [
            "gold_v3_refreeze_after_train_cleanup",
            "prompt_policy_alignment_pending",
        ],
        "gold_v3": {
            "contamination_reaudit": {"expected_train_sha256": "old-train"}
        },
    }


def test_speaker_contract_migration_is_explicit_and_idempotent(tmp_path):
    module = _module()
    train_path = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(train_path, [_game_record(), _constructed_record()])
    _write_json(manifest_path, _manifest(module, train_path))

    first = module.apply_contract(train_path=train_path, manifest_path=manifest_path)
    first_train_text = train_path.read_text(encoding="utf-8")
    first_manifest_text = manifest_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in first_train_text.splitlines()]
    manifest = json.loads(first_manifest_text)

    assert first["status"] == "applied_pending_existing_freeze_blockers"
    assert records[0]["metadata"]["interlocutor_kind"] == "canonical_character"
    assert records[0]["metadata"]["interlocutor_label"] == "夜子"
    assert records[1]["metadata"]["interlocutor_kind"] == "generic_user"
    assert "interlocutor_label" not in records[1]["metadata"]
    assert manifest["validation"] == {
        "count": 70,
        "sha256": "unchanged-validation",
    }
    assert manifest["freeze_blockers"] == [
        "gold_v3_refreeze_after_train_cleanup",
        "prompt_policy_alignment_pending",
    ]
    assert manifest["train"]["sha256"] == module._text_sha256(train_path)
    assert manifest["gold_v3"]["contamination_reaudit"][
        "expected_train_sha256"
    ] == manifest["train"]["sha256"]

    second = module.apply_contract(train_path=train_path, manifest_path=manifest_path)
    assert second["status"] == "already_applied"
    assert train_path.read_text(encoding="utf-8") == first_train_text
    assert manifest_path.read_text(encoding="utf-8") == first_manifest_text


def test_speaker_contract_migration_fails_before_writing_unproven_game_identity(
    tmp_path,
):
    module = _module()
    train_path = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records = [_game_record(with_context=False), _constructed_record()]
    _write_jsonl(train_path, records)
    _write_json(manifest_path, _manifest(module, train_path))
    original_train = train_path.read_text(encoding="utf-8")
    original_manifest = manifest_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="has no context speaker"):
        module.apply_contract(train_path=train_path, manifest_path=manifest_path)

    assert train_path.read_text(encoding="utf-8") == original_train
    assert manifest_path.read_text(encoding="utf-8") == original_manifest


def test_active_kisaki_train_uses_the_complete_speaker_contract():
    module = _module()
    manifest = json.loads(ACTIVE_MANIFEST.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in ACTIVE_TRAIN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    game = [
        record
        for record in records
        if record["metadata"]["data_source"] == "game_extraction"
    ]
    constructed = [
        record
        for record in records
        if record["metadata"]["data_source"] != "game_extraction"
    ]

    assert len(records) == manifest["train"]["count"] == 926
    assert len(game) == 522
    assert len(constructed) == 404
    assert all(
        record["metadata"]["interlocutor_kind"] == "canonical_character"
        and record["metadata"]["interlocutor_label"]
        == record["metadata"]["context_speaker_label"]
        for record in game
    )
    assert all(
        record["metadata"]["interlocutor_kind"] == "generic_user"
        and "interlocutor_label" not in record["metadata"]
        for record in constructed
    )
    assert manifest["train"]["sha256"] == module._text_sha256(ACTIVE_TRAIN)
    assert manifest["speaker_contract_revision"]["status"] == "applied"

    for record in records:
        normalized = normalize_chat_record(
            record,
            default_system_prompt="月社妃契约测试",
            system_prompt_policy="replace",
        )
        user_messages = [
            message["content"]
            for message in normalized
            if message["role"] == "user"
        ]
        if record["metadata"]["interlocutor_kind"] == "canonical_character":
            expected = f"当前对话者：{record['metadata']['interlocutor_label']}。"
            assert all(expected in message for message in user_messages)
        else:
            assert all("<speaker_label" not in message for message in user_messages)
