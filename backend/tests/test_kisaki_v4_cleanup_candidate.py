from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_kisaki_v4_cleanup_candidate.py"
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"


def _module():
    spec = importlib.util.spec_from_file_location("build_kisaki_v4_cleanup_candidate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_cleanup_partitions_records_and_repairs_metadata():
    records = [
        {"id": "keep", "messages": [], "metadata": {"scene": "\ufffd"}},
        {"id": "exclude", "messages": [], "metadata": {"scene": "old"}},
    ]
    policy = {
        "formal_training_exclusion_ids": ["exclude"],
        "truncation_damaged_ids": ["exclude"],
        "scene_metadata_repairs": {"keep": "修复后的场景"},
    }

    kept, auxiliary, repaired = _module().apply_cleanup(records, policy)

    assert [record["id"] for record in kept] == ["keep"]
    assert kept[0]["metadata"]["scene"] == "修复后的场景"
    assert [record["id"] for record in auxiliary] == ["exclude"]
    assert auxiliary[0]["metadata"]["formal_training_status"] == "auxiliary_only"
    assert (
        auxiliary[0]["metadata"]["formal_training_exclusion_reason"]
        == "truncation_damaged_and_code_dominant"
    )
    assert repaired == ["keep"]


def test_cleanup_rejects_unknown_policy_ids():
    with pytest.raises(ValueError, match="references missing records"):
        _module().apply_cleanup(
            [{"id": "keep", "messages": [], "metadata": {}}],
            {
                "formal_training_exclusion_ids": ["missing"],
                "truncation_damaged_ids": [],
                "scene_metadata_repairs": {},
            },
        )


def test_supervision_count_respects_last_turn_records():
    record = {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
        "metadata": {"assistant_supervision": "last"},
    }

    assert _module().supervised_assistant_turns(record) == 1
    record["metadata"]["assistant_supervision"] = "all"
    assert _module().supervised_assistant_turns(record) == 2


def test_generated_cleanup_candidate_is_complete_and_non_destructive():
    source = _jsonl(V4 / "train.jsonl")
    train = _jsonl(V4 / "cleanup/candidate/train.jsonl")
    auxiliary = _jsonl(V4 / "cleanup/candidate/technical_auxiliary.jsonl")
    manifest = _json(V4 / "cleanup/candidate/manifest.json")
    policy = _json(V4 / "cleanup/length_cleanup_policy.json")

    assert len(source) == 926
    assert len(train) == 926
    assert len(auxiliary) == 22
    assert {record["id"] for record in train}.isdisjoint(
        {record["id"] for record in auxiliary}
    )
    assert [record["id"] for record in train] == [record["id"] for record in source]
    assert manifest["source"]["canonical_train_count"] == 948
    assert {record["id"] for record in auxiliary} == set(
        policy["formal_training_exclusion_ids"]
    )
    assert not any(
        "\ufffd" in str(record.get("metadata", {}).get("scene", ""))
        for record in train + auxiliary
    )
    assert manifest["status"] == "ready_for_review"
    assert manifest["train"]["count"] == len(train)
    assert manifest["formal_training_exclusions"]["count"] == len(auxiliary)
    assert manifest["formal_training_exclusions"]["sha256"] == _module().text_sha256(
        V4 / "cleanup/candidate/technical_auxiliary.jsonl"
    )
    assert manifest["formal_training_exclusions"]["reason_counts"] == {
        "code_dominant_auxiliary_capability_sample": 18,
        "truncation_damaged_and_code_dominant": 4,
    }
    assert manifest["training_contract"]["token_audit"]["records_over_limit"] == 0
    assert manifest["training_contract"]["token_audit"]["maximum_tokens"] <= 1280
