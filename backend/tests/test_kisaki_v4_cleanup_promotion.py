from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/promote_kisaki_v4_cleanup_candidate.py"


def _module():
    spec = importlib.util.spec_from_file_location("promote_kisaki_v4_cleanup_candidate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def test_cleanup_promotion_updates_data_and_blocks_formal_training(tmp_path):
    module = _module()
    active_train = tmp_path / "train.jsonl"
    active_train.write_text('{"id":"old"}\n', encoding="utf-8")
    candidate_train = tmp_path / "candidate.jsonl"
    candidate_train.write_text('{"id":"new"}\n', encoding="utf-8")
    auxiliary = tmp_path / "auxiliary.jsonl"
    auxiliary.write_text('{"id":"old-aux"}\n', encoding="utf-8")
    active_manifest = tmp_path / "canonical.json"
    _write_json(
        active_manifest,
        {
            "schema_version": 8,
            "status": "frozen",
            "train": {"count": 1, "sha256": module.text_sha256(active_train)},
            "validation": {"count": 1},
            "gold_v3": {
                "status": "frozen",
                "contamination_reaudit": {"train_sha256": module.text_sha256(active_train)},
            },
            "freeze_blockers": [],
        },
    )
    candidate_manifest = tmp_path / "candidate-manifest.json"
    _write_json(
        candidate_manifest,
        {
            "status": "ready_for_review",
            "source": {
                "canonical_train_count": 1,
                "canonical_train_sha256": module.text_sha256(active_train),
            },
            "train": {
                "path": str(candidate_train),
                "count": 1,
                "sha256": module.text_sha256(candidate_train),
                "assistant_supervision_targets": 1,
                "source_distribution": {"game_extraction": 1},
            },
            "formal_training_exclusions": {
                "path": str(auxiliary),
                "count": 1,
                "sha256": module.text_sha256(auxiliary),
            },
            "scene_metadata_repairs": {"count": 0},
            "training_contract": {
                "token_audit": {"max_sequence_length": 1280, "maximum_tokens": 100}
            },
        },
    )
    approval = tmp_path / "approval.json"
    _write_json(
        approval,
        {
            "status": "approved_for_promotion",
            "manual_record_by_record_review_claimed": False,
            "candidate_manifest_sha256": module.text_sha256(candidate_manifest),
            "candidate_train_sha256": module.text_sha256(candidate_train),
        },
    )
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "schema_version": 5,
            "status": "ready",
            "active_assets": {
                "persona_prompt": {"status": "approved", "formal_use_allowed": True},
                "prompt_policy": {"version": "3.1.0"},
                "canonical_dataset": {},
                "gold_v3": {"status": "frozen", "formal_use_allowed": True},
            },
            "research": [{"id": "R0V4"}, {"id": "R1V4"}],
        },
    )

    result = module.promote(
        candidate_manifest_path=candidate_manifest,
        approval_path=approval,
        canonical_manifest_path=active_manifest,
        canonical_train_path=active_train,
        registry_path=registry,
    )

    promoted = json.loads(active_manifest.read_text(encoding="utf-8"))
    registered = json.loads(registry.read_text(encoding="utf-8"))
    assert result["status"] == "promoted_pending_refreeze"
    assert active_train.read_text(encoding="utf-8") == candidate_train.read_text(encoding="utf-8")
    assert promoted["status"] == "frozen_data_pending_gold"
    assert promoted["freeze_blockers"] == [
        "gold_v3_refreeze_after_train_cleanup",
        "prompt_policy_alignment_pending",
    ]
    assert promoted["gold_v3"]["formal_use_allowed"] is False
    assert registered["status"] == "r0v4_cleanup_pending_refreeze"
    assert registered["active_assets"]["gold_v3"]["formal_use_allowed"] is False
    assert next(item for item in registered["research"] if item["id"] == "R1V4")[
        "status"
    ] == "blocked_until_r0v4_refrozen"
