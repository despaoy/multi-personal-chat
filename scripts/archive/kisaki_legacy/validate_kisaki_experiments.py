"""Historical V3 R1 validator; retained only to reproduce archived results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import (  # noqa: E402
    canonical_json_hash,
    compare_experiment_configs,
    environment_snapshot,
    sha256_text_file,
    validate_frozen_gold,
    validate_r1_variant_set,
)

EXPERIMENT_DIR = BACKEND / "data" / "character_dialogues" / "experiments"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
CONFIG_PATHS = {
    name: CONFIG_DIR / f"kisaki_{name}_canonical.json"
    for name in ("e1", "e2", "e3", "e4", "e5")
}
DATASET_MANIFEST = EXPERIMENT_DIR / "canonical_dataset_manifest.json"
REGISTRY_PATH = EXPERIMENT_DIR / "canonical_experiment_registry.json"
PROGRAM_REGISTRY = EXPERIMENT_DIR / "research" / "research_program_registry_v3.json"
GOLD_V2_PATH = BACKEND / "evaluation" / "kisaki_gold_set_v2.json"
PROMPT_V3_PATH = BACKEND / "data" / "character_dialogues" / "kisaki_system_prompt_v3.txt"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_registry(*, require_model: bool, formal_eval: bool) -> tuple[dict[str, Any], list[str]]:
    configs = {name: _load(path) for name, path in CONFIG_PATHS.items()}
    dataset = _load(DATASET_MANIFEST)
    errors = validate_r1_variant_set(configs)

    for name, config in configs.items():
        label = f"R1-{name.upper()}"
        for field, manifest_key in (("train_data_path", "train"), ("eval_data_path", "validation")):
            path = _resolve_project_path(config[field])
            expected = dataset[manifest_key]
            if not path.exists():
                errors.append(f"{label} {field} does not exist: {path}")
            elif sha256_text_file(path) != expected["sha256"]:
                errors.append(f"{label} {field} hash does not match canonical dataset manifest")
        model_path = Path(config["base_model_path"])
        if require_model and not model_path.exists():
            errors.append(f"{label} base model does not exist: {model_path}")
        if model_path.exists():
            model_config_path = model_path / "config.json"
            if not model_config_path.exists():
                errors.append(f"{label} base model has no config.json")
            elif _load(model_config_path).get("quantization_config"):
                errors.append(f"{label} base model is pre-quantized and cannot be used for training")

    if not PROMPT_V3_PATH.exists():
        errors.append(f"evaluation prompt v3 is missing: {PROMPT_V3_PATH}")
    prompt = {
        "path": PROMPT_V3_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "version": 3,
        "sha256": sha256_text_file(PROMPT_V3_PATH) if PROMPT_V3_PATH.exists() else None,
    }

    gold: dict[str, Any] = {
        "path": GOLD_V2_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "status": "missing" if not GOLD_V2_PATH.exists() else "unknown",
    }
    if GOLD_V2_PATH.exists():
        gold_data = _load(GOLD_V2_PATH)
        gold.update(status=gold_data.get("status"), sha256=sha256_text_file(GOLD_V2_PATH))
        if formal_eval:
            errors.extend(validate_frozen_gold(gold_data))
    elif formal_eval:
        errors.append("Frozen Gold v2 is required for formal evaluation")

    experiments = []
    for name, config in configs.items():
        experiments.append(
            {
                "id": f"R1-{name.upper()}",
                "run_id": config.get("_experiment_id"),
                "config": CONFIG_PATHS[name].relative_to(PROJECT_ROOT).as_posix(),
                "config_sha256": sha256_text_file(CONFIG_PATHS[name]),
                "config_contract_sha256": canonical_json_hash(config),
                "baseline_differences": compare_experiment_configs(configs["e1"], config),
            }
        )

    registry = {
        "schema_version": 3,
        "report_type": "runtime_preflight",
        "program_registry": PROGRAM_REGISTRY.relative_to(PROJECT_ROOT).as_posix(),
        "series_id": "KISAKI-R1-CONTROLLED-PEFT",
        "status": "ready_for_training" if not errors else "blocked",
        "mock": False,
        "fixed_contract": {
            "dataset_manifest": DATASET_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "train_sha256": dataset["train"]["sha256"],
            "validation_sha256": dataset["validation"]["sha256"],
            "base_model_repo_id": "Qwen/Qwen3-8B",
            "base_model_local_alias": "Qwen3-8B-Instruct",
            "seeds": [42, 43, 44],
            "pilot_seed": 42,
            "evaluation_prompt": prompt,
            "generation": {
                "temperature": 0.0,
                "max_tokens": 256,
                "enable_thinking": False,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
            },
        },
        "experiments": experiments,
        "gold_v2": gold,
        "legacy": {
            "status": "legacy_exploratory_non_comparable",
            "series": ["E1 historical", "E2 historical", "E2' Safety++", "E2'' RAG"],
        },
        "environment": environment_snapshot(PROJECT_ROOT),
        "errors": errors,
    }
    return registry, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical Kisaki R1 experiments")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--formal-eval", action="store_true")
    parser.add_argument("--write-registry", action="store_true")
    parser.add_argument("--registry-output", type=Path)
    args = parser.parse_args()
    registry, errors = build_registry(require_model=args.require_model, formal_eval=args.formal_eval)
    if args.write_registry:
        registry_output = args.registry_output or REGISTRY_PATH
        registry_output.parent.mkdir(parents=True, exist_ok=True)
        registry_output.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(registry, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
