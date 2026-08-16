import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_kisaki_v4_final_review.py"


def test_final_review_package_tracks_only_current_frozen_objects(tmp_path):
    spec = importlib.util.spec_from_file_location("build_kisaki_v4_final_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = tmp_path / "review"
    train = [
        json.loads(line)
        for line in (ROOT / "backend/data/character_dialogues/experiments/v4/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    augmentation_count = sum(
        row.get("metadata", {}).get("data_source") in module.REVIEWED_AUGMENTATION_SOURCES
        for row in train
    )
    augmentation_batches = (augmentation_count + module.BATCH_SIZE - 1) // module.BATCH_SIZE
    manifest = module.build(output)
    assert manifest == {
        "schema_version": 1,
        "status": "pending_final_human_review",
        "game_train": 576,
        "constructed_train": 150,
        "approved_multiturn_augmentation": augmentation_count,
        "validation": 70,
        "game_batches": 12,
        "constructed_batches": 3,
        "approved_multiturn_augmentation_batches": augmentation_batches,
        "validation_batches": 2,
    }
    assert len(list((output / "02_GAME_CONTEXT").glob("batch_*.md"))) == 12
    assert len(list((output / "03_CONSTRUCTED").glob("batch_*.md"))) == 3
    assert len(list((output / "04_VALIDATION").glob("batch_*.md"))) == 2
    assert (
        len(list((output / "05_APPROVED_AUGMENTATION").glob("batch_*.md")))
        == augmentation_batches
    )
    index = (output / "00_INDEX.md").read_text(encoding="utf-8")
    assert "不要再使用旧" in index
    assert "仅供追溯，不重复审核" in index
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored == manifest
