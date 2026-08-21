import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "validate_kisaki_v4_training_gate.py"
CONFIG_SCRIPT = PROJECT_ROOT / "scripts" / "build_kisaki_r1v4_configs.py"


def _module():
    spec = importlib.util.spec_from_file_location("validate_kisaki_v4_training_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config_builder():
    spec = importlib.util.spec_from_file_location(
        "build_kisaki_r1v4_configs_for_gate", CONFIG_SCRIPT
    )
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
        config_manifest_path=tmp_path / "missing-config-manifest.json",
    )
    assert result["passed"] is False
    assert "canonical V4 dataset manifest is missing" in result["blockers"]
    assert "Gold v3 is missing" in result["blockers"]
    assert "R1V4 config manifest is missing" in result["blockers"]
    assert "human review has not been explicitly finalized" not in result["blockers"]


def test_gate_passes_only_complete_frozen_contracts(tmp_path, monkeypatch):
    review = tmp_path / "review.json"
    dataset = tmp_path / "dataset.json"
    gold = tmp_path / "gold.json"
    prompt = tmp_path / "prompt.txt"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config_dir = tmp_path / "configs"
    config_manifest = config_dir / "config_manifest.json"
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
            "dataset_id": "KISAKI-CANONICAL-V4",
            "status": "frozen",
            "freeze_blockers": [],
            "prompt_policy": {
                "version": _module().PROMPT_POLICY_VERSION,
                "required_training_policy": "replace",
            },
            "train": {
                "path": str(train),
                "count": 1,
                "sha256": _module()._sha256_text(train),
            },
            "validation": {
                "path": str(validation),
                "count": 1,
                "sha256": _module()._sha256_text(validation),
            },
            "cleanup_revision": {"maximum_observed_tokens": 1},
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
    builder = _config_builder()
    monkeypatch.setattr(builder, "PROMPT_PATH", prompt)
    builder.write_configs(dataset, config_dir, builder.DEFAULT_TEMPLATE)
    result = _module().validate_gate(
        review_path=review,
        dataset_path=dataset,
        gold_path=gold,
        prompt_path=prompt,
        config_manifest_path=config_manifest,
    )
    assert result["passed"] is True
    assert result["blockers"] == []


def test_gate_refuses_stale_r1_configs(tmp_path):
    config_manifest = tmp_path / "config-manifest.json"
    _write(
        config_manifest,
        {
            "status": "stale_after_canonical_cleanup",
            "formal_use_allowed": False,
            "prompt_policy_version": "3.1.0",
            "experiments": _module().EXPECTED_EXPERIMENTS,
        },
    )
    result = _module().validate_gate(
        review_path=tmp_path / "missing-review.json",
        dataset_path=tmp_path / "missing-dataset.json",
        gold_path=tmp_path / "missing-gold.json",
        config_manifest_path=config_manifest,
    )
    assert "R1V4 config manifest is stale" in result["blockers"]
    assert "R1V4 configs are not approved for formal use" in result["blockers"]
    assert "R1V4 config prompt policy version is stale" in result["blockers"]
