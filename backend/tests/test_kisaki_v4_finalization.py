import importlib.util
import json
from pathlib import Path

import pytest

from evaluation.experiment_contracts import canonical_json_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/finalize_kisaki_v4_dataset.py"


def _module():
    spec = importlib.util.spec_from_file_location("finalize_kisaki_v4_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path):
    module = _module()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"id":"train"}\n', encoding="utf-8")
    validation.write_text('{"id":"validation"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "dataset_id": "KISAKI-CANONICAL-V4",
            "status": "frozen_data_pending_gold",
            "train": {
                "status": "frozen",
                "count": 1,
                "path": str(train),
                "sha256": module._text_hash(train),
            },
            "validation": {
                "status": "frozen",
                "count": 1,
                "path": str(validation),
                "sha256": module._text_hash(validation),
            },
            "freeze_blockers": ["gold_v21_human_review_pending", "gold_v3_missing"],
        },
    )
    review = tmp_path / "review.json"
    _write(
        review,
        {
            "approval": {
                "items": {
                    "gold_v21": {"status": "approved"},
                    "gold_v3": {"status": "approved"},
                }
            }
        },
    )
    prompts = [{"id": f"gold-{index:03d}", "prompt": "test"} for index in range(150)]
    content_sha256 = canonical_json_hash(prompts)
    review_payload = json.loads(review.read_text(encoding="utf-8"))
    review_payload["approval"]["items"]["gold_v3"]["content_sha256"] = content_sha256
    _write(review, review_payload)
    gold = tmp_path / "gold.json"
    _write(
        gold,
        {
            "dataset_id": "KISAKI-GOLD-V3",
            "status": "frozen",
            "evaluation_role": "final_held_out",
            "formal_use_allowed": True,
            "total_prompts": len(prompts),
            "prompts": prompts,
            "content_sha256": content_sha256,
        },
    )
    approval = tmp_path / "approval.json"
    _write(
        approval,
        {
            "status": "approved",
            "approved_count": 150,
            "content_sha256": content_sha256,
        },
    )
    audit = tmp_path / "audit.json"
    _write(
        audit,
        {
            "schema_version": 2,
            "status": "clean",
            "candidate_count": 150,
            "candidate_content_sha256": content_sha256,
            "frozen_train_count": 1,
            "frozen_validation_count": 1,
            "frozen_reference_count": 2,
            "frozen_train_sha256": module._text_hash(train),
            "frozen_validation_sha256": module._text_hash(validation),
            "duplicate_ids": [],
            "duplicate_normalized_prompts": [],
            "text_overlap_matches": [],
            "rag_evidence_event_overlaps": [],
        },
    )
    registry = tmp_path / "registry.json"
    _write(
        registry,
        {
            "status": "r0v4_gold_refreeze_pending",
            "active_assets": {
                "canonical_dataset": {"status": "frozen_data_pending_gold"},
                "gold_v3": {
                    "status": "stale_after_canonical_cleanup",
                    "formal_use_allowed": False,
                },
            },
            "research": [
                {"id": "R0V4", "status": "gold_refreeze_pending"},
                {"id": "R1V4", "status": "blocked_until_r0v4_refrozen"},
            ],
        },
    )
    return module, manifest, gold, review, audit, approval, registry


def test_finalizer_attaches_approved_gold_and_clears_blockers(tmp_path):
    module, manifest, gold, review, audit, approval, registry = _fixture(tmp_path)
    result = module.finalize(manifest, gold, review, audit, approval, registry)
    assert result["status"] == "frozen"
    assert result["freeze_blockers"] == []
    assert result["gold_v3"]["id"] == "KISAKI-GOLD-V3"
    assert result["gold_v3"]["count"] == 150
    assert result["gold_v3"]["sha256"] == module._text_hash(gold)
    assert result["gold_v3"]["formal_use_allowed"] is True
    assert result["gold_v3"]["contamination_reaudit"]["status"] == "clean"
    updated_registry = json.loads(registry.read_text(encoding="utf-8"))
    assert updated_registry["active_assets"]["canonical_dataset"]["status"] == "frozen"
    assert updated_registry["active_assets"]["gold_v3"]["formal_use_allowed"] is True
    assert updated_registry["research"] == [
        {"id": "R0V4", "status": "complete"},
        {"id": "R1V4", "status": "config_regeneration_pending"},
    ]


def test_finalizer_refuses_pending_gold_review(tmp_path):
    module, manifest, gold, review, audit, approval, registry = _fixture(tmp_path)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["approval"]["items"]["gold_v3"]["status"] = "pending_human_review"
    _write(review, payload)
    with pytest.raises(ValueError, match="Gold v3 human review is not approved"):
        module.finalize(manifest, gold, review, audit, approval, registry)


def test_finalizer_refuses_stale_train_audit(tmp_path):
    module, manifest, gold, review, audit, approval, registry = _fixture(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["frozen_train_sha256"] = "stale"
    _write(audit, payload)
    with pytest.raises(ValueError, match="stale train split"):
        module.finalize(manifest, gold, review, audit, approval, registry)


def test_finalizer_refuses_audit_for_different_gold_content(tmp_path):
    module, manifest, gold, review, audit, approval, registry = _fixture(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["candidate_content_sha256"] = "different-gold"
    _write(audit, payload)
    with pytest.raises(ValueError, match="does not match frozen Gold content"):
        module.finalize(manifest, gold, review, audit, approval, registry)
