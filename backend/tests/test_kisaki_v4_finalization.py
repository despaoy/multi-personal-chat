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
                "path": str(train),
                "sha256": module._text_hash(train),
            },
            "validation": {
                "status": "frozen",
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
            "content_sha256": canonical_json_hash(prompts),
        },
    )
    return module, manifest, gold, review


def test_finalizer_attaches_approved_gold_and_clears_blockers(tmp_path):
    module, manifest, gold, review = _fixture(tmp_path)
    result = module.finalize(manifest, gold, review)
    assert result["status"] == "frozen"
    assert result["freeze_blockers"] == []
    assert result["gold_v3"]["id"] == "KISAKI-GOLD-V3"
    assert result["gold_v3"]["count"] == 150
    assert result["gold_v3"]["sha256"] == module._text_hash(gold)


def test_finalizer_refuses_pending_gold_review(tmp_path):
    module, manifest, gold, review = _fixture(tmp_path)
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["approval"]["items"]["gold_v3"]["status"] = "pending_human_review"
    _write(review, payload)
    with pytest.raises(ValueError, match="Gold v3 human review is not approved"):
        module.finalize(manifest, gold, review)
