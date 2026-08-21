"""Run one gate-approved Kisaki V4 training experiment with immutable provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import (  # noqa: E402
    environment_snapshot,
    sha256_text_file,
)

EXPERIMENT_DIR = BACKEND / "data" / "character_dialogues" / "experiments"
V4_DIR = EXPERIMENT_DIR / "v4"
CONFIGS = {
    name: V4_DIR / "configs" / f"kisaki_r1v4_{name}.json"
    for name in ("e1", "e2", "e3", "e4", "e5")
}
DATASET_MANIFEST = V4_DIR / "canonical_dataset_manifest.json"
TRAINING_GATE = PROJECT_ROOT / "scripts" / "validate_kisaki_v4_training_gate.py"
SERVER_ROOT = Path(os.getenv("MULTIPERSONAL_LAB_ROOT") or os.getenv("QQCHAT_LAB_ROOT") or str(PROJECT_ROOT / "runtime"))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_snapshot() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    rows = []
    for line in output.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == 6:
            rows.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "driver": values[2],
                    "memory_used_mb": int(values[3]),
                    "memory_total_mb": int(values[4]),
                    "utilization_percent": int(values[5]),
                }
            )
    return rows


def _best_checkpoint(output_dir: Path) -> str | None:
    state_files = sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: _checkpoint_step(path.parent),
    )
    if not state_files:
        return None
    state = _load(state_files[-1])
    value = state.get("best_model_checkpoint")
    return str(value) if value else None


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def prompt_contract_error(config: dict[str, Any], dataset: dict[str, Any]) -> str | None:
    contract = dataset.get("prompt_policy", {})
    required_policy = contract.get("required_training_policy")
    if config.get("system_prompt_policy") != required_policy:
        return "system_prompt_policy"
    prompt_path_value = contract.get("path")
    if not prompt_path_value:
        return "prompt_path"
    prompt_path = Path(prompt_path_value)
    prompt_path = prompt_path if prompt_path.is_absolute() else PROJECT_ROOT / prompt_path
    if not prompt_path.exists():
        return "prompt_path"
    if config.get("system_prompt", "").strip() != prompt_path.read_text(encoding="utf-8").strip():
        return "system_prompt_content"
    prompt_hash = hashlib.sha256(
        config.get("system_prompt", "").strip().encode("utf-8")
    ).hexdigest()
    if config.get("_prompt_content_sha256") != prompt_hash:
        return "prompt_content_hash"
    if config.get("_prompt_policy_version") != contract.get("version"):
        return "prompt_policy_version"
    return None


def dataset_contract_error(config: dict[str, Any], dataset: dict[str, Any]) -> str | None:
    if dataset.get("status") != "frozen" or dataset.get("freeze_blockers"):
        return "dataset_status"
    if config.get("_dataset_version") != dataset.get("dataset_id"):
        return "dataset_version"
    bindings = (
        ("train", "train_data_path", "_train_data_sha256"),
        ("validation", "eval_data_path", "_validation_data_sha256"),
    )
    for split, path_key, hash_key in bindings:
        contract = dataset.get(split, {})
        if config.get(path_key) != contract.get("path"):
            return f"{split}_path"
        if config.get(hash_key) != contract.get("sha256"):
            return f"{split}_hash"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one canonical Kisaki R1 PEFT experiment")
    parser.add_argument("--experiment", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gate = subprocess.run(
        [sys.executable, str(TRAINING_GATE)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if gate.returncode != 0:
        print(gate.stdout.strip() or gate.stderr.strip(), file=sys.stderr)
        return 2
    if not CONFIGS[args.experiment].exists():
        print(f"v4_experiment_config_missing={CONFIGS[args.experiment]}", file=sys.stderr)
        return 2

    config = _load(CONFIGS[args.experiment])
    dataset = _load(DATASET_MANIFEST)
    if dataset_error := dataset_contract_error(config, dataset):
        print(f"canonical_dataset_contract_failed={dataset_error}", file=sys.stderr)
        return 2
    if prompt_error := prompt_contract_error(config, dataset):
        print(f"canonical_prompt_contract_failed={prompt_error}", file=sys.stderr)
        return 2
    experiment_id = str(config["_experiment_id"])
    output_dir = SERVER_ROOT / "runtime" / "loras" / "kisaki" / "r1v4" / args.experiment / f"seed{args.seed}"
    run_dir = SERVER_ROOT / "runtime" / "experiments" / "kisaki" / "r1v4" / args.experiment / f"seed{args.seed}"
    resolved_config_path = run_dir / "resolved_training_config.json"
    run_manifest_path = run_dir / "run_manifest.json"
    config["seed"] = args.seed
    config["output_dir"] = str(output_dir)
    # 运行时覆盖 base_model_path（保持配置文件 frozen，支持跨服务器迁移）
    model_name = Path(config["base_model_path"]).name
    resolved_model = SERVER_ROOT / "runtime" / "models" / model_name
    if resolved_model.exists():
        config["base_model_path"] = str(resolved_model)
    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=_checkpoint_step)
        config["resume_from_checkpoint"] = str(checkpoints[-1]) if checkpoints else None
    elif output_dir.exists() and any(output_dir.iterdir()):
        print(f"refusing_to_overwrite_nonempty_output={output_dir}", file=sys.stderr)
        return 2

    for field, key in (("train_data_path", "train"), ("eval_data_path", "validation")):
        path = Path(config[field])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.exists() or sha256_text_file(path) != dataset[key]["sha256"]:
            print(f"canonical_dataset_contract_failed={field}", file=sys.stderr)
            return 2
    model_config_path = Path(config["base_model_path"]) / "config.json"
    if not model_config_path.exists():
        print(f"base_model_config_missing={model_config_path}", file=sys.stderr)
        return 2
    if _load(model_config_path).get("quantization_config"):
        print("refusing_prequantized_training_model=true", file=sys.stderr)
        return 2

    _write(resolved_config_path, config)
    started_at = datetime.now(timezone.utc).isoformat()
    run_manifest: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "status": "running",
        "seed": args.seed,
        "started_at": started_at,
        "resolved_config": str(resolved_config_path),
        "dataset": {
            "train_count": dataset["train"]["count"],
            "validation_count": dataset["validation"]["count"],
            "train_sha256": dataset["train"]["sha256"],
            "validation_sha256": dataset["validation"]["sha256"],
            "used_fixed_validation": True,
        },
        "environment": {
            **environment_snapshot(PROJECT_ROOT),
            "packages": package_versions(),
            "gpus_before": gpu_snapshot(),
        },
        "adapter": {},
        "error": "",
    }
    _write(run_manifest_path, run_manifest)

    started = time.perf_counter()
    command = [sys.executable, "-m", "training.trainer", "--config", str(resolved_config_path)]
    completed = subprocess.run(command, cwd=BACKEND, check=False)
    run_manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
    run_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    run_manifest["environment"]["gpus_after"] = gpu_snapshot()
    run_manifest["exit_code"] = completed.returncode
    if completed.returncode == 0:
        final_adapter = output_dir / "final"
        run_manifest["status"] = "training_complete"
        run_manifest["adapter"] = {
            "path": str(final_adapter),
            "best_checkpoint": _best_checkpoint(output_dir),
            "training_evaluation": str(output_dir / "training_evaluation.json"),
        }
    else:
        run_manifest["status"] = "failed"
        run_manifest["error"] = f"training process exited with {completed.returncode}"
    _write(run_manifest_path, run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
