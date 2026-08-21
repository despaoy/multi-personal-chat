from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"
BATCH_ROOT = V4 / "augmentation_candidates"
SCRIPT = ROOT / "scripts/promote_kisaki_v41_round06.py"
DATA_SOURCE = "codex_user_simulation_v41_reviewed"


def _module():
    spec = importlib.util.spec_from_file_location("promote_kisaki_v41_batch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _batch_dirs():
    return sorted(path for path in BATCH_ROOT.glob("automation_batch_*") if path.is_dir())


def test_automation_batches_have_complete_review_and_promotion_provenance():
    batches = _batch_dirs()
    assert batches
    expected_records = 0
    expected_turns = 0

    for batch in batches:
        approval, records = _module().build_promoted_records(
            batch / "candidates.json", batch / "approved_sessions.json"
        )
        review = _json(batch / "review.json")
        result = _json(batch / "promotion_result.json")
        assert approval["status"] in {"approved_after_review", "approved_after_revision"}
        assert review["status"] == approval["status"]
        assert review["summary"]["rejected"] == 0
        assert result["status"] == "promoted"
        assert result["approval_id"] == approval["approval_id"]
        assert result["promoted"]["record_ids"] == [record["id"] for record in records]
        expected_records += len(records)
        expected_turns += sum(
            message["role"] == "assistant"
            for record in records
            for message in record["messages"]
        )

    assert expected_records == len(batches) * 4
    assert expected_turns == len(batches) * 20


def test_automation_records_are_complete_sessions_with_balanced_task_types():
    train = _jsonl(V4 / "train.jsonl")
    auxiliary = _jsonl(V4 / "cleanup/candidate/technical_auxiliary.jsonl")
    formal_records = [
        row for row in train if row.get("metadata", {}).get("data_source") == DATA_SOURCE
    ]
    auxiliary_records = [
        row for row in auxiliary if row.get("metadata", {}).get("data_source") == DATA_SOURCE
    ]
    records = formal_records + auxiliary_records
    batches = _batch_dirs()

    assert len(records) == len(batches) * 4
    assert len({record["id"] for record in records}) == len(records)
    assert len(formal_records) == 251
    assert len(auxiliary_records) == 21
    assert all(record["metadata"]["turns"] == 5 for record in records)
    assert all(
        [message["role"] for message in record["messages"]] == ["user", "assistant"] * 5
        for record in records
    )
    assert all(
        record["metadata"]["human_review"]["status"]
        in {"approved_after_review", "approved_after_revision"}
        for record in records
    )

    task_counts = Counter(record["metadata"]["task_type"] for record in records)
    assert max(task_counts.values()) <= math.ceil(len(records) * 0.25)
    social = task_counts["casual_chat"] + task_counts["emotional_relationship"]
    assert social >= math.ceil(len(records) * 0.35)


def test_automation_source_count_matches_canonical_and_gold_reaudit_is_current():
    module = _module()
    train = _jsonl(V4 / "train.jsonl")
    validation = _jsonl(V4 / "validation.jsonl")
    records = [row for row in train if row.get("metadata", {}).get("data_source") == DATA_SOURCE]
    manifest = _json(V4 / "canonical_dataset_manifest.json")
    audit = _json(ROOT / "backend/evaluation/kisaki_gold_set_v3_contamination_audit.json")

    assert manifest["train"]["count"] == len(train)
    assert manifest["train"]["sha256"] == module._text_sha256(V4 / "train.jsonl")
    assert manifest["train"]["source_distribution"][DATA_SOURCE] == len(records)
    assert audit["status"] == "clean"
    assert audit["text_overlap_matches"] == []
    assert audit["frozen_reference_count"] == len(train) + len(validation)
    assert audit["frozen_train_count"] == len(train)
    assert audit["frozen_validation_count"] == len(validation)
    assert audit["frozen_train_sha256"] == manifest["train"]["sha256"]
    assert manifest["gold_v3"]["status"] == "frozen"
    assert manifest["gold_v3"]["formal_use_allowed"] is True


def test_gold_contamination_helper_returns_a_complete_clean_audit():
    module = _module()
    train = _jsonl(V4 / "train.jsonl")
    validation = _jsonl(V4 / "validation.jsonl")
    manifest = _json(V4 / "canonical_dataset_manifest.json")
    audit = module._gold_contamination_audit(
        train=train,
        validation=validation,
        gold_v3=_json(ROOT / "backend/evaluation/kisaki_gold_set_v3.json"),
        gold_v21=_json(ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json"),
        train_sha256=manifest["train"]["sha256"],
        validation_sha256=manifest["validation"]["sha256"],
    )

    assert audit["status"] == "clean"
    assert audit["text_overlap_matches"] == []
    assert audit["frozen_reference_count"] == len(train) + len(validation)
