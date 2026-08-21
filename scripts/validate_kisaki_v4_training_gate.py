#!/usr/bin/env python3
"""Refuse formal Kisaki V4 training until human-review contracts are complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.experiment_contracts import (  # noqa: E402
    validate_frozen_gold,
    validate_r1_variant_set,
)
from inference.prompt_policy import PROMPT_POLICY_VERSION  # noqa: E402
DEFAULT_REVIEW = PROJECT_ROOT / "docs" / "research" / "review_packets" / "kisaki_v4" / "review_manifest.json"
DEFAULT_DATASET = PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "experiments" / "v4" / "canonical_dataset_manifest.json"
DEFAULT_GOLD = PROJECT_ROOT / "backend" / "evaluation" / "kisaki_gold_set_v3.json"
DEFAULT_PROMPT = PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "kisaki_system_prompt_v3.txt"
DEFAULT_CONFIG_MANIFEST = (
    PROJECT_ROOT
    / "backend/data/character_dialogues/experiments/v4/configs/config_manifest.json"
)
EXPECTED_EXPERIMENTS = [f"R1-E{index}" for index in range(1, 6)]
REQUIRED_CATEGORIES = {
    "profile_prompt",
    "source_coverage",
    "game_train",
    "constructed_train",
    "validation",
    "gold_v21",
    "exclusions",
    "experiment_configs",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _config_blockers(
    config_manifest_path: Path,
    dataset_path: Path,
    dataset: dict[str, Any] | None,
    prompt_path: Path,
) -> list[str]:
    blockers: list[str] = []
    if not config_manifest_path.exists():
        return ["R1V4 config manifest is missing"]

    manifest = _load(config_manifest_path)
    if manifest.get("status") != "generated_for_frozen_dataset":
        blockers.append("R1V4 config manifest is stale")
    if manifest.get("formal_use_allowed") is not True:
        blockers.append("R1V4 configs are not approved for formal use")
    if manifest.get("prompt_policy_version") != PROMPT_POLICY_VERSION:
        blockers.append("R1V4 config prompt policy version is stale")
    if manifest.get("experiments") != EXPECTED_EXPERIMENTS:
        blockers.append("R1V4 config manifest does not contain the canonical E1-E5 set")

    if dataset is not None:
        if manifest.get("dataset_id") != dataset.get("dataset_id"):
            blockers.append("R1V4 config dataset ID is stale")
        if manifest.get("dataset_manifest_sha256") != _sha256_text(dataset_path):
            blockers.append("R1V4 config dataset manifest hash is stale")
        for split in ("train", "validation"):
            if manifest.get(split) != {
                "path": dataset.get(split, {}).get("path"),
                "count": dataset.get(split, {}).get("count"),
                "sha256": dataset.get(split, {}).get("sha256"),
            }:
                blockers.append(f"R1V4 config {split} binding is stale")

    if prompt_path.exists():
        prompt_contract = manifest.get("prompt", {})
        if prompt_contract.get("sha256") != _sha256_text(prompt_path):
            blockers.append("R1V4 config prompt hash is stale")

    files = manifest.get("config_files")
    expected_names = {f"e{index}" for index in range(1, 6)}
    if not isinstance(files, dict) or set(files) != expected_names:
        blockers.append("R1V4 config file registry is incomplete")
        return blockers

    configs: dict[str, dict[str, Any]] = {}
    prompt_text = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""
    prompt_content_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    for name in sorted(expected_names):
        entry = files[name]
        path = _resolve(str(entry.get("path", "")))
        if not path.exists() or entry.get("sha256") != _sha256_text(path):
            blockers.append(f"R1V4 {name.upper()} config file hash mismatch")
            continue
        config = _load(path)
        configs[name] = config
        if dataset is not None:
            if config.get("_dataset_version") != dataset.get("dataset_id"):
                blockers.append(f"R1V4 {name.upper()} dataset version is stale")
            if config.get("_train_data_sha256") != dataset["train"]["sha256"]:
                blockers.append(f"R1V4 {name.upper()} train hash is stale")
            if config.get("_validation_data_sha256") != dataset["validation"]["sha256"]:
                blockers.append(f"R1V4 {name.upper()} validation hash is stale")
            if config.get("train_data_path") != dataset["train"]["path"]:
                blockers.append(f"R1V4 {name.upper()} train path is stale")
            if config.get("eval_data_path") != dataset["validation"]["path"]:
                blockers.append(f"R1V4 {name.upper()} validation path is stale")
        if config.get("_prompt_policy_version") != PROMPT_POLICY_VERSION:
            blockers.append(f"R1V4 {name.upper()} prompt policy is stale")
        if config.get("system_prompt", "").strip() != prompt_text:
            blockers.append(f"R1V4 {name.upper()} prompt content is stale")
        if config.get("_prompt_content_sha256") != prompt_content_sha256:
            blockers.append(f"R1V4 {name.upper()} prompt content hash is stale")

    if len(configs) == len(expected_names):
        blockers.extend(
            f"R1V4 single-variable contract failed: {error}"
            for error in validate_r1_variant_set(configs)
        )
    return blockers


def validate_gate(
    *,
    review_path: Path,
    dataset_path: Path,
    gold_path: Path,
    prompt_path: Path = DEFAULT_PROMPT,
    config_manifest_path: Path = DEFAULT_CONFIG_MANIFEST,
    disk_path: Path | None = None,
    minimum_free_gb: float = 15.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    approved: set[str] = set()
    dataset: dict[str, Any] | None = None
    if not review_path.exists():
        blockers.append("human review manifest is missing")
    else:
        review = _load(review_path)
        approved = set(review.get("approval", {}).get("approved_categories", []))
        missing = sorted(REQUIRED_CATEGORIES - approved)
        if missing:
            blockers.append("human review categories are pending: " + ", ".join(missing))
        if review.get("approval", {}).get("all_required_approved") is not True:
            blockers.append("human review has not been explicitly finalized")
        prompt_review = review.get("approval", {}).get("items", {}).get("system_prompt_v3", {})
        if prompt_review.get("status") != "approved":
            blockers.append("system prompt v3 has not been explicitly approved")
        if not prompt_path.exists():
            blockers.append("system prompt v3 is missing")
        reviewed_prompt_path = _resolve(str(prompt_review.get("path", "")))
        if reviewed_prompt_path.resolve() != prompt_path.resolve():
            blockers.append("system prompt v3 path does not match the reviewed prompt")
        if prompt_review.get("prompt_policy_version") != PROMPT_POLICY_VERSION:
            blockers.append("reviewed prompt policy version is stale")

    if not dataset_path.exists():
        blockers.append("canonical V4 dataset manifest is missing")
    else:
        dataset = _load(dataset_path)
        if dataset.get("status") != "frozen":
            blockers.append("canonical V4 dataset is not frozen")
        if dataset.get("freeze_blockers"):
            blockers.append("canonical V4 dataset still has freeze blockers")
        prompt_contract = dataset.get("prompt_policy", {})
        if prompt_contract.get("version") != PROMPT_POLICY_VERSION:
            blockers.append("canonical V4 prompt policy version is stale")
        for key in ("train", "validation"):
            contract = dataset.get(key, {})
            expected_hash = contract.get("sha256")
            if not expected_hash:
                blockers.append(f"canonical V4 {key} hash is missing")
                continue
            data_file = _resolve(str(contract.get("path", "")))
            if not data_file.exists() or _sha256_text(data_file) != expected_hash:
                blockers.append(f"canonical V4 {key} file does not match its frozen hash")

    if not gold_path.exists():
        blockers.append("Gold v3 is missing")
    else:
        gold = _load(gold_path)
        blockers.extend(validate_frozen_gold(gold, require_final_held_out=True))
        if gold.get("total_prompts") != 150:
            blockers.append("Gold v3 must contain exactly 150 prompts")

    blockers.extend(
        _config_blockers(config_manifest_path, dataset_path, dataset, prompt_path)
    )

    free_gb = None
    if disk_path is not None:
        free_gb = shutil.disk_usage(disk_path).free / (1024**3)
        if free_gb < minimum_free_gb:
            blockers.append(
                f"free disk space {free_gb:.2f}GB is below {minimum_free_gb:.2f}GB"
            )
    return {
        "schema_version": 1,
        "gate": "KISAKI-V4-FORMAL-TRAINING",
        "passed": not blockers,
        "blockers": blockers,
        "approved_categories": sorted(approved),
        "free_disk_gb": None if free_gb is None else round(free_gb, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--config-manifest", type=Path, default=DEFAULT_CONFIG_MANIFEST)
    parser.add_argument("--disk-path", type=Path)
    parser.add_argument("--minimum-free-gb", type=float, default=15.0)
    args = parser.parse_args()
    result = validate_gate(
        review_path=args.review.resolve(),
        dataset_path=args.dataset.resolve(),
        gold_path=args.gold.resolve(),
        prompt_path=args.prompt.resolve(),
        config_manifest_path=args.config_manifest.resolve(),
        disk_path=args.disk_path.resolve() if args.disk_path else None,
        minimum_free_gb=args.minimum_free_gb,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
