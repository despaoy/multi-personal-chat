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


def test_gate_blocks_when_frozen_dataset_and_gold_are_missing(tmp_path):
    review = tmp_path / "review.json"
    _write(
        review,
        {
            "approval": {
                "approved_categories": sorted(_module().REQUIRED_CATEGORIES),
                "all_required_approved": True,
                "items": {
                    "system_prompt_v3": {
                        "status": "approved",
                        "path": str(_module().DEFAULT_PROMPT),
                        "prompt_policy_version": _module().PROMPT_POLICY_VERSION,
                    }
                },
            }
        },
    )
    result = _module().validate_gate(
        review_path=review,
        dataset_path=tmp_path / "missing-dataset.json",
        gold_path=tmp_path / "missing-gold.json",
    )
    assert result["passed"] is False
    assert "canonical V4 dataset manifest is missing" in result["blockers"]
    assert "Gold v3 is missing" in result["blockers"]
    assert "human review has not been explicitly finalized" not in result["blockers"]


def test_gate_passes_only_complete_frozen_contracts(tmp_path):
    review = tmp_path / "review.json"
    dataset = tmp_path / "dataset.json"
    gold = tmp_path / "gold.json"
    prompt = tmp_path / "prompt.txt"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    prompt.write_text("approved prompt", encoding="utf-8")
    train.write_text('{"id":"train"}\n', encoding="utf-8")
    validation.write_text('{"id":"validation"}\n', encoding="utf-8")
    required = sorted(_module().REQUIRED_CATEGORIES)

    _write(
        review,
        {
            "approval": {
                "approved_categories": required,
                "all_required_approved": True,
                "items": {
                    "system_prompt_v3": {
                        "status": "approved",
                        "path": str(prompt),
                        "prompt_policy_version": _module().PROMPT_POLICY_VERSION,
                    }
                },
            },
        },
    )
    _write(
        dataset,
        {
            "status": "frozen",
            "freeze_blockers": [],
            "prompt_policy": {"version": _module().PROMPT_POLICY_VERSION},
            "train": {"path": str(train), "sha256": _module()._sha256_text(train)},
            "validation": {
                "path": str(validation),
                "sha256": _module()._sha256_text(validation),
            },
        },
    )
    prompts = [{"id": str(index), "prompt": "test"} for index in range(150)]
    from evaluation.experiment_contracts import canonical_json_hash

    _write(
        gold,
        {
            "status": "frozen",
            "evaluation_role": "final_held_out",
            "formal_use_allowed": True,
            "total_prompts": 150,
            "prompts": prompts,
            "content_sha256": canonical_json_hash(prompts),
        },
    )
    result = _module().validate_gate(
        review_path=review,
        dataset_path=dataset,
        gold_path=gold,
        prompt_path=prompt,
    )
    assert result["passed"] is True
    assert result["blockers"] == []
