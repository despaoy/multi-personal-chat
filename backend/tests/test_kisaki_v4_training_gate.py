import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "validate_kisaki_v4_training_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("validate_kisaki_v4_training_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_gate_blocks_current_unreviewed_state(tmp_path):
    result = _module().validate_gate(
        review_path=PROJECT_ROOT / "docs/research/review_packets/kisaki_v4/review_manifest.json",
        dataset_path=tmp_path / "missing-dataset.json",
        gold_path=tmp_path / "missing-gold.json",
    )
    assert result["passed"] is False
    assert any("categories are pending" in item for item in result["blockers"])
    assert "canonical V4 dataset manifest is missing" in result["blockers"]
    assert "Gold v3 is missing" in result["blockers"]


def test_gate_passes_only_complete_frozen_contracts(tmp_path):
    review = tmp_path / "review.json"
    dataset = tmp_path / "dataset.json"
    gold = tmp_path / "gold.json"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("approved prompt", encoding="utf-8")
    required = sorted(_module().REQUIRED_CATEGORIES)
    import hashlib

    _write(
        review,
        {
            "source_hashes": {"prompt_v3": hashlib.sha256(prompt.read_bytes()).hexdigest()},
            "approval": {
                "approved_categories": required,
                "all_required_approved": True,
                "items": {"system_prompt_v3": {"status": "approved"}},
            },
        },
    )
    _write(
        dataset,
        {
            "status": "frozen",
            "train": {"sha256": "train"},
            "validation": {"sha256": "validation"},
        },
    )
    _write(gold, {"status": "frozen", "total_prompts": 150, "content_sha256": "gold"})
    result = _module().validate_gate(
        review_path=review,
        dataset_path=dataset,
        gold_path=gold,
        prompt_path=prompt,
    )
    assert result["passed"] is True
    assert result["blockers"] == []
