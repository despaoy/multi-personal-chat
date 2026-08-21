from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/align_kisaki_v4_prompt_policy.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "align_kisaki_v4_prompt_policy", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_prompt_policy_alignment_preserves_persona_and_gold_blocker(tmp_path):
    module = _module()
    prompt = tmp_path / "prompt.txt"
    manifest = tmp_path / "manifest.json"
    review = tmp_path / "review.json"
    registry = tmp_path / "registry.json"
    config_manifest = tmp_path / "configs.json"
    prompt.write_text(
        "你是月社妃。琉璃是你的亲生哥哥。保持人物身份。\n", encoding="utf-8"
    )
    _write(
        manifest,
        {
            "schema_version": 10,
            "prompt_policy": {"path": module.PROMPT_PATH, "version": "3.1.0"},
            "freeze_blockers": [
                "gold_v3_refreeze_after_train_cleanup",
                "prompt_policy_alignment_pending",
            ],
        },
    )
    _write(
        review,
        {
            "approval": {
                "items": {
                    "system_prompt_v3": {
                        "status": "approved",
                        "path": module.PROMPT_PATH,
                        "prompt_policy_version": "3.1.0",
                    },
                    "experiment_configs": {"status": "approved"},
                }
            }
        },
    )
    _write(
        registry,
        {
            "schema_version": 6,
            "status": "pending",
            "active_assets": {
                "persona_prompt": {},
                "prompt_policy": {},
                "canonical_dataset": {},
            },
            "research": [{"id": "R0V4"}, {"id": "R1V4"}],
        },
    )
    _write(
        config_manifest,
        {"schema_version": 1, "prompt_policy_version": "3.1.0"},
    )
    original_prompt = prompt.read_text(encoding="utf-8")

    result = module.align(
        prompt_path=prompt,
        manifest_path=manifest,
        review_path=review,
        registry_path=registry,
        config_manifest_path=config_manifest,
    )

    aligned_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    aligned_review = json.loads(review.read_text(encoding="utf-8"))
    aligned_registry = json.loads(registry.read_text(encoding="utf-8"))
    stale_configs = json.loads(config_manifest.read_text(encoding="utf-8"))
    assert result["status"] == "aligned"
    assert prompt.read_text(encoding="utf-8") == original_prompt
    assert aligned_manifest["prompt_policy"]["version"] == "3.3.0"
    assert aligned_manifest["freeze_blockers"] == [
        "gold_v3_refreeze_after_train_cleanup"
    ]
    assert aligned_review["approval"]["items"]["system_prompt_v3"][
        "policy_alignment"
    ]["persona_content_changed"] is False
    assert aligned_registry["active_assets"]["prompt_policy"]["status"] == "active"
    assert stale_configs["formal_use_allowed"] is False
    assert stale_configs["required_prompt_policy_version"] == "3.3.0"

    second = module.align(
        prompt_path=prompt,
        manifest_path=manifest,
        review_path=review,
        registry_path=registry,
        config_manifest_path=config_manifest,
    )
    assert second["status"] == "already_aligned"


def test_active_prompt_contract_is_frozen_while_r1_configs_remain_stale():
    module = _module()
    manifest = json.loads(module.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    review = json.loads(module.DEFAULT_REVIEW.read_text(encoding="utf-8"))
    registry = json.loads(module.DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    config_manifest = json.loads(
        module.DEFAULT_CONFIG_MANIFEST.read_text(encoding="utf-8")
    )

    assert manifest["prompt_policy"]["version"] == module.PROMPT_POLICY_VERSION
    assert manifest["prompt_policy"]["stable_persona_content_changed"] is False
    assert manifest["status"] == "frozen"
    assert manifest["freeze_blockers"] == []
    assert review["approval"]["items"]["system_prompt_v3"][
        "prompt_policy_version"
    ] == module.PROMPT_POLICY_VERSION
    assert registry["active_assets"]["prompt_policy"]["status"] == "active"
    assert next(item for item in registry["research"] if item["id"] == "R0V4")[
        "status"
    ] == "complete"
    assert config_manifest["status"] == "generated_for_frozen_dataset"
    assert config_manifest["formal_use_allowed"] is True
    assert config_manifest["prompt_policy_version"] == module.PROMPT_POLICY_VERSION
    assert config_manifest["single_variable_contract"]["status"] == "validated"
