import importlib.util
import json
from pathlib import Path

import pytest

from evaluation.experiment_contracts import validate_r1_variant_set


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_r1v4_config_builder_changes_one_registered_factor():
    builder = _module("build_kisaki_r1v4_configs", "scripts/build_kisaki_r1v4_configs.py")
    manifest = {
        "dataset_id": "KISAKI-CANONICAL-V4",
        "train": {"path": "train.jsonl", "sha256": "train-hash"},
        "validation": {"path": "validation.jsonl", "sha256": "validation-hash"},
        "prompt_policy": {"required_training_policy": "replace"},
    }
    template = json.loads(builder.DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    configs = builder.build_configs(manifest, template, "persona prompt")
    assert validate_r1_variant_set(configs) == []
    assert [configs[name]["_experiment_id"] for name in sorted(configs)] == [
        "R1-E1",
        "R1-E2",
        "R1-E3",
        "R1-E4",
        "R1-E5",
    ]
    assert {config["_train_data_sha256"] for config in configs.values()} == {
        "train-hash"
    }
    assert {config["_validation_data_sha256"] for config in configs.values()} == {
        "validation-hash"
    }
    assert {config["max_seq_length"] for config in configs.values()} == {1280}
    assert {config["learning_rate"] for config in configs.values()} == {1e-4}
    assert {config["num_train_epochs"] for config in configs.values()} == {2}
    assert {config["save_total_limit"] for config in configs.values()} == {4}


def test_r1v4_config_writer_refuses_a_draft_dataset(tmp_path):
    builder = _module("build_kisaki_r1v4_configs_draft", "scripts/build_kisaki_r1v4_configs.py")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"status": "draft", "freeze_blockers": ["review_pending"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be frozen"):
        builder.write_configs(manifest, tmp_path / "configs")


def test_active_r1v4_configs_bind_current_data_prompt_and_single_variables():
    builder = _module(
        "build_kisaki_r1v4_configs_active", "scripts/build_kisaki_r1v4_configs.py"
    )
    canonical = json.loads(builder.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    config_manifest = json.loads(
        (builder.DEFAULT_OUTPUT / "config_manifest.json").read_text(encoding="utf-8")
    )
    configs = {
        name: json.loads(
            (builder.DEFAULT_OUTPUT / f"kisaki_r1v4_{name}.json").read_text(
                encoding="utf-8"
            )
        )
        for name in builder.VARIANTS
    }

    assert config_manifest["schema_version"] == 3
    assert config_manifest["status"] == "generated_for_frozen_dataset"
    assert config_manifest["formal_use_allowed"] is True
    assert config_manifest["train"]["sha256"] == canonical["train"]["sha256"]
    assert config_manifest["validation"]["sha256"] == canonical["validation"]["sha256"]
    assert config_manifest["prompt_policy_version"] == builder.PROMPT_POLICY_VERSION
    assert config_manifest["single_variable_contract"]["status"] == "validated"
    assert config_manifest["training_contract"] == {
        "revision": "r1v4_stability_v2",
        "reason": "Reduce update strength after the first E1 pilot showed free-generation collapse.",
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "save_total_limit": 4,
        "data_changed": False,
    }
    assert validate_r1_variant_set(configs) == []
    assert {config["max_seq_length"] for config in configs.values()} == {1280}
    assert {config["learning_rate"] for config in configs.values()} == {1e-4}
    assert {config["num_train_epochs"] for config in configs.values()} == {2}
    assert {config["save_total_limit"] for config in configs.values()} == {4}
    for name, config in configs.items():
        assert config["_train_data_sha256"] == canonical["train"]["sha256"]
        assert config["_validation_data_sha256"] == canonical["validation"]["sha256"]
        assert config["_prompt_policy_version"] == builder.PROMPT_POLICY_VERSION
        assert config_manifest["config_files"][name]["sha256"] == builder._text_sha256(
            builder.DEFAULT_OUTPUT / f"kisaki_r1v4_{name}.json"
        )


def test_r1v4_config_writer_refuses_a_template_that_would_truncate_canonical_data(
    tmp_path, monkeypatch
):
    builder = _module(
        "build_kisaki_r1v4_configs_length", "scripts/build_kisaki_r1v4_configs.py"
    )
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    prompt = tmp_path / "prompt.txt"
    train.write_text('{"id":"train"}\n', encoding="utf-8")
    validation.write_text('{"id":"validation"}\n', encoding="utf-8")
    prompt.write_text("persona prompt\n", encoding="utf-8")
    monkeypatch.setattr(builder, "PROMPT_PATH", prompt)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "KISAKI-CANONICAL-V4",
                "status": "frozen",
                "freeze_blockers": [],
                "train": {
                    "path": str(train),
                    "count": 1,
                    "sha256": builder._text_sha256(train),
                },
                "validation": {
                    "path": str(validation),
                    "count": 1,
                    "sha256": builder._text_sha256(validation),
                },
                "prompt_policy": {"required_training_policy": "replace"},
                "cleanup_revision": {"maximum_observed_tokens": 1165},
            }
        ),
        encoding="utf-8",
    )
    template = tmp_path / "template.json"
    payload = json.loads(builder.DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    payload["max_seq_length"] = 1024
    template.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="max_seq_length"):
        builder.write_configs(manifest, tmp_path / "configs", template)


def test_checkpoint_steps_are_sorted_numerically():
    runner = _module("run_kisaki_experiment_numeric", "scripts/run_kisaki_experiment.py")
    checkpoints = [Path("checkpoint-900"), Path("checkpoint-1000"), Path("checkpoint-50")]
    assert sorted(checkpoints, key=runner._checkpoint_step)[-1].name == "checkpoint-1000"


def test_dataset_freeze_refuses_pending_human_approval(tmp_path, monkeypatch):
    freezer = _module("freeze_kisaki_v4_dataset", "scripts/freeze_kisaki_v4_dataset.py")
    v4_dir = tmp_path / "v4"
    v4_dir.mkdir()
    (v4_dir / "canonical_dataset_manifest.json").write_text(
        json.dumps({"status": "draft_rebuilt_pending_review"}), encoding="utf-8"
    )
    (v4_dir / "train_candidate.jsonl").write_text(
        json.dumps(
            {
                "id": "game-pending",
                "messages": [],
                "metadata": {"data_source": "game_extraction"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (v4_dir / "validation_candidate.jsonl").write_text("", encoding="utf-8")
    pending_approval = tmp_path / "game-approval.json"
    pending_approval.write_text(
        json.dumps(
            {
                "status": "pending_human_review",
                "candidate_count": 1,
                "default_decision": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(freezer, "GAME_APPROVAL", pending_approval)
    with pytest.raises(ValueError, match="game train review is not approved"):
        freezer.freeze(v4_dir)


def test_dataset_freeze_uses_the_current_validation_review(tmp_path, monkeypatch):
    freezer = _module("freeze_kisaki_v4_dataset_current", "scripts/freeze_kisaki_v4_dataset.py")
    v4_dir = tmp_path / "v4"
    v4_dir.mkdir()

    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def write_jsonl(path, values):
        path.write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
            encoding="utf-8",
        )

    game = {
        "id": "game-1",
        "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        "metadata": {"data_source": "game_extraction"},
    }
    constructed = {
        "id": "constructed-1",
        "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        "metadata": {"data_source": "constructed"},
    }
    validation = [
        {
            "id": sample_id,
            "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
            "metadata": {"data_source": "game_extraction"},
        }
        for sample_id in ("validation-blocked", "validation-kept")
    ]
    write_jsonl(v4_dir / "train_candidate.jsonl", [game, constructed])
    write_jsonl(v4_dir / "validation_candidate.jsonl", validation)
    write_json(
        v4_dir / "canonical_dataset_manifest.json",
        {
            "status": "draft_rebuilt_pending_review",
            "train": {},
            "validation": {},
            "checks": {"train_validation_blocker_pairs": 1},
        },
    )
    write_json(
        v4_dir / "validation_exclusions.json",
        {
            "exclusions": [
                {
                    "validation_id": "validation-blocked",
                    "paired_train_id": "game-1",
                    "reason": "near_duplicate_user_question",
                }
            ]
        },
    )

    game_approval = tmp_path / "game.json"
    constructed_approval = tmp_path / "constructed.json"
    validation_approval = tmp_path / "validation.json"
    write_json(
        game_approval,
        {
            "status": "approved",
            "default_decision": "approve",
            "candidate_count": 1,
            "excluded_ids": [],
        },
    )
    write_json(
        constructed_approval,
        {"status": "approved", "approved_by": "owner", "approved_count": 1},
    )
    write_json(
        validation_approval,
        {
            "review_id": "current-validation",
            "status": "pending_human_review",
            "candidate_count": 2,
            "default_decision": None,
            "excluded_ids": [],
        },
    )
    monkeypatch.setattr(freezer, "GAME_APPROVAL", game_approval)
    monkeypatch.setattr(freezer, "CONSTRUCTED_APPROVAL", constructed_approval)
    monkeypatch.setattr(freezer, "VALIDATION_APPROVAL", validation_approval)

    with pytest.raises(ValueError, match="current validation review is not approved"):
        freezer.freeze(v4_dir)

    write_json(
        validation_approval,
        {
            "review_id": "current-validation",
            "status": "approved",
            "reviewed_by": "owner",
            "reviewed_at": "2026-08-10",
            "candidate_count": 2,
            "default_decision": "approve",
            "excluded_ids": ["validation-blocked"],
        },
    )
    frozen = freezer.freeze(v4_dir)
    assert frozen["status"] == "frozen_data_pending_gold"
    assert frozen["train"]["count"] == 2
    assert frozen["validation"]["count"] == 1
    assert frozen["validation"]["applied_exclusions"] == ["validation-blocked"]
