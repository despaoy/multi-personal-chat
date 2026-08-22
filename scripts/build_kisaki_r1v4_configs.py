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
