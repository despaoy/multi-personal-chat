#!/usr/bin/env python3
"""Generate the five single-variable R1V4 training configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import (  # noqa: E402
    R1_VARIANT_DIFFS,
    validate_r1_variant_set,
)
from inference.prompt_policy import PROMPT_POLICY_VERSION  # noqa: E402


V4_DIR = BACKEND / "data/character_dialogues/experiments/v4"
DEFAULT_MANIFEST = V4_DIR / "canonical_dataset_manifest.json"
DEFAULT_OUTPUT = V4_DIR / "configs"
DEFAULT_TEMPLATE = V4_DIR / "r1v4_base_config.json"
PROMPT_PATH = BACKEND / "data/character_dialogues/kisaki_system_prompt_v3.txt"

VARIANTS = {
    "e1": {
        "neftune_noise_alpha": 0.0,
        "use_dora": False,
        "use_rslora": False,
        "packing": False,
        "role": "baseline",
    },
    "e2": {
        "neftune_noise_alpha": 5.0,
        "use_dora": False,
        "use_rslora": False,
        "packing": False,
        "role": "neftune_only",
    },
    "e3": {
        "neftune_noise_alpha": 0.0,
        "use_dora": True,
        "use_rslora": False,
        "packing": False,
        "role": "dora_only",
    },
    "e4": {
        "neftune_noise_alpha": 0.0,
        "use_dora": False,
        "use_rslora": True,
        "packing": False,
        "role": "rslora_only",
    },
    "e5": {
        "neftune_noise_alpha": 0.0,
        "use_dora": False,
        "use_rslora": False,
        "packing": True,
        "role": "packing_only",
    },
}

CALIBRATION_NAME = "e1_calibration_lr2e5"
CALIBRATION_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_train_epochs": 1,
    "eval_steps": 25,
    "save_steps": 25,
    "save_total_limit": 5,
}
CALIBRATION_ALLOWED_DIFFS = {
    "_experiment_id",
    "_comparison_role",
    "_calibration_parent",
    "_calibration_reason",
    "_formal_variant",
    "output_dir",
    *CALIBRATION_OVERRIDES,
}
ALPHA32_CALIBRATION_NAME = "e1_calibration_lr2e5_alpha32"
ALPHA32_CALIBRATION_OVERRIDES = {
    **CALIBRATION_OVERRIDES,
    "lora_alpha": 32,
}
ALPHA32_CALIBRATION_ALLOWED_DIFFS = {
    *CALIBRATION_ALLOWED_DIFFS,
    "lora_alpha",
}
ALPHA16_CALIBRATION_NAME = "e1_calibration_lr2e5_alpha16"
ALPHA16_CALIBRATION_OVERRIDES = {
    **CALIBRATION_OVERRIDES,
    "eval_steps": 20,
    "save_steps": 20,
    "save_total_limit": 6,
    "lora_alpha": 16,
}
ALPHA16_CALIBRATION_ALLOWED_DIFFS = {
    *CALIBRATION_ALLOWED_DIFFS,
    "lora_alpha",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_sha256_value(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _text_sha256(path: Path) -> str:
    return _text_sha256_value(path.read_text(encoding="utf-8"))


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_configs(
    manifest: dict[str, Any],
    template: dict[str, Any],
    system_prompt: str,
) -> dict[str, dict[str, Any]]:
    """Build all configs in memory and verify the one-variable contract."""

    configs: dict[str, dict[str, Any]] = {}
    for name, variant in VARIANTS.items():
        config = copy.deepcopy(template)
        config.update(
            {
                "_experiment_id": f"R1-E{name[-1]}",
                "_comparison_role": variant["role"],
                "_dataset_manifest_path": "backend/data/character_dialogues/experiments/v4/canonical_dataset_manifest.json",
                "_dataset_version": manifest["dataset_id"],
                "_train_data_sha256": manifest["train"]["sha256"],
                "_validation_data_sha256": manifest["validation"]["sha256"],
                "_prompt_policy_version": PROMPT_POLICY_VERSION,
                "_prompt_content_sha256": _text_sha256_value(system_prompt.strip()),
                "train_data_path": manifest["train"]["path"],
                "eval_data_path": manifest["validation"]["path"],
                "system_prompt": system_prompt.strip(),
                "system_prompt_policy": manifest["prompt_policy"]["required_training_policy"],
                # The runner resolves this against MULTIPERSONAL_LAB_ROOT at runtime.
                "output_dir": f"runtime/loras/kisaki/r1v4/{name}/seed42",
                "neftune_noise_alpha": variant["neftune_noise_alpha"],
                "use_dora": variant["use_dora"],
                "use_rslora": variant["use_rslora"],
                "packing": variant["packing"],
            }
        )
        config.pop("_v2_config_source", None)
        configs[name] = config

    errors = validate_r1_variant_set(configs)
    if errors:
        raise ValueError("; ".join(errors))
    return configs


def build_e1_calibration_config(e1_config: dict[str, Any]) -> dict[str, Any]:
    """Build a non-formal E1 run that only lowers optimization intensity."""

    config = copy.deepcopy(e1_config)
    config.update(CALIBRATION_OVERRIDES)
    config.update(
        {
            "_experiment_id": "R1-E1-CAL-LR2E5",
            "_comparison_role": "stability_calibration_not_formal_variant",
            "_calibration_parent": "R1-E1",
            "_calibration_reason": "All LR=1e-4 checkpoints failed the free-generation gate.",
            "_formal_variant": False,
            "output_dir": f"runtime/loras/kisaki/r1v4/{CALIBRATION_NAME}/seed42",
        }
    )
    differences = {
        key
        for key in set(e1_config) | set(config)
        if e1_config.get(key) != config.get(key)
    }
    unexpected = differences - CALIBRATION_ALLOWED_DIFFS
    if unexpected:
        raise ValueError(
            "E1 calibration changes non-calibration fields: "
            + ", ".join(sorted(unexpected))
        )
    return config


def _build_e1_followup_calibration_config(
    e1_config: dict[str, Any],
    *,
    overrides: dict[str, Any],
    allowed_diffs: set[str],
    experiment_id: str,
    comparison_role: str,
    parent: str,
    reason: str,
    output_name: str,
) -> dict[str, Any]:
    """Build one non-formal stability follow-up without changing its data contract."""

    config = copy.deepcopy(e1_config)
    config.update(overrides)
    config.update(
        {
            "_experiment_id": experiment_id,
            "_comparison_role": comparison_role,
            "_calibration_parent": parent,
            "_calibration_reason": reason,
            "_formal_variant": False,
            "output_dir": f"runtime/loras/kisaki/r1v4/{output_name}/seed42",
        }
    )
    differences = {
        key
        for key in set(e1_config) | set(config)
        if e1_config.get(key) != config.get(key)
    }
    unexpected = differences - allowed_diffs
    if unexpected:
        raise ValueError(
            "E1 follow-up calibration changes non-calibration fields: "
            + ", ".join(sorted(unexpected))
        )
    return config


def build_e1_alpha32_calibration_config(
    e1_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the follow-up run selected by the first scale diagnostic."""

    return _build_e1_followup_calibration_config(
        e1_config,
        overrides=ALPHA32_CALIBRATION_OVERRIDES,
        allowed_diffs=ALPHA32_CALIBRATION_ALLOWED_DIFFS,
        experiment_id="R1-E1-CAL-LR2E5-A32",
        comparison_role="alpha32_stability_calibration_not_formal_variant",
        parent="R1-E1-CAL-LR2E5",
        reason="Post-training alpha scaling isolated excessive adapter update strength.",
        output_name=ALPHA32_CALIBRATION_NAME,
    )


def build_e1_alpha16_calibration_config(
    e1_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the lower-strength run selected after alpha32 failed semantic review."""

    return _build_e1_followup_calibration_config(
        e1_config,
        overrides=ALPHA16_CALIBRATION_OVERRIDES,
        allowed_diffs=ALPHA16_CALIBRATION_ALLOWED_DIFFS,
        experiment_id="R1-E1-CAL-LR2E5-A16",
        comparison_role="alpha16_stability_calibration_not_formal_variant",
        parent="R1-E1-CAL-LR2E5-A32",
        reason=(
            "Alpha32 checkpoint 25 passed length gates but failed semantic review; "
            "inference scaling localized the stable range near alpha16."
        ),
        output_name=ALPHA16_CALIBRATION_NAME,
    )


def write_configs(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
    template_path: Path = DEFAULT_TEMPLATE,
) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if manifest.get("status") != "frozen" or manifest.get("freeze_blockers"):
        raise ValueError("V4 dataset must be frozen without blockers before configs are generated")
    maximum_observed_tokens = int(
        manifest.get("cleanup_revision", {}).get("maximum_observed_tokens", 0)
    )
    template = _load(template_path)
    if int(template.get("max_seq_length", 0)) < maximum_observed_tokens:
        raise ValueError(
            "max_seq_length is smaller than the canonical maximum observed tokens"
        )
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    configs = build_configs(
        manifest,
        template,
        system_prompt,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_files: dict[str, dict[str, Any]] = {}
    for name, config in configs.items():
        path = output_dir / f"kisaki_r1v4_{name}.json"
        _write_json_atomic(path, config)
        config_files[name] = {
            "experiment_id": config["_experiment_id"],
            "comparison_role": config["_comparison_role"],
            "path": _path_label(path),
            "sha256": _text_sha256(path),
        }
    calibration = build_e1_calibration_config(configs["e1"])
    calibration_path = output_dir / f"kisaki_r1v4_{CALIBRATION_NAME}.json"
    _write_json_atomic(calibration_path, calibration)
    alpha32_calibration = build_e1_alpha32_calibration_config(configs["e1"])
    alpha32_calibration_path = (
        output_dir / f"kisaki_r1v4_{ALPHA32_CALIBRATION_NAME}.json"
    )
    _write_json_atomic(alpha32_calibration_path, alpha32_calibration)
    alpha16_calibration = build_e1_alpha16_calibration_config(configs["e1"])
    alpha16_calibration_path = (
        output_dir / f"kisaki_r1v4_{ALPHA16_CALIBRATION_NAME}.json"
    )
    _write_json_atomic(alpha16_calibration_path, alpha16_calibration)
    summary = {
        "schema_version": 3,
        "status": "generated_for_frozen_dataset",
        "formal_use_allowed": True,
        "training_contract": {
            "revision": "r1v4_stability_v2",
            "reason": "Reduce update strength after the first E1 pilot showed free-generation collapse.",
            "learning_rate": template["learning_rate"],
            "num_train_epochs": template["num_train_epochs"],
            "save_total_limit": template["save_total_limit"],
            "data_changed": False,
        },
        "dataset_id": manifest["dataset_id"],
        "dataset_status": manifest["status"],
        "dataset_manifest_path": _path_label(manifest_path),
        "dataset_manifest_sha256": _text_sha256(manifest_path),
        "train": {
            "path": manifest["train"]["path"],
            "count": manifest["train"]["count"],
            "sha256": manifest["train"]["sha256"],
        },
        "validation": {
            "path": manifest["validation"]["path"],
            "count": manifest["validation"]["count"],
            "sha256": manifest["validation"]["sha256"],
        },
        "prompt": {
            "path": _path_label(PROMPT_PATH),
            "sha256": _text_sha256(PROMPT_PATH),
        },
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "experiments": [configs[name]["_experiment_id"] for name in sorted(configs)],
        "config_files": config_files,
        "calibration_config": {
            "experiment_id": calibration["_experiment_id"],
            "formal_use_allowed": False,
            "parent": calibration["_calibration_parent"],
            "path": _path_label(calibration_path),
            "sha256": _text_sha256(calibration_path),
            "overrides": CALIBRATION_OVERRIDES,
        },
        "followup_calibration_config": {
            "experiment_id": alpha32_calibration["_experiment_id"],
            "formal_use_allowed": False,
            "parent": alpha32_calibration["_calibration_parent"],
            "path": _path_label(alpha32_calibration_path),
            "sha256": _text_sha256(alpha32_calibration_path),
            "overrides": ALPHA32_CALIBRATION_OVERRIDES,
        },
        "stability_calibration_config": {
            "experiment_id": alpha16_calibration["_experiment_id"],
            "formal_use_allowed": False,
            "parent": alpha16_calibration["_calibration_parent"],
            "path": _path_label(alpha16_calibration_path),
            "sha256": _text_sha256(alpha16_calibration_path),
            "overrides": ALPHA16_CALIBRATION_OVERRIDES,
        },
        "single_variable_contract": {
            "status": "validated",
            "baseline": "R1-E1",
            "differences": {
                configs[name]["_experiment_id"]: differences
                for name, differences in R1_VARIANT_DIFFS.items()
            },
        },
    }
    _write_json_atomic(output_dir / "config_manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()
    try:
        summary = write_configs(args.manifest, args.output, args.template)
    except ValueError as exc:
        print(f"config_generation_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
