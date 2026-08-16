"""Shared contracts for reproducible character-model experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_EXPERIMENT_IDS = tuple(f"R1-E{index}" for index in range(1, 6))
CONFIG_COMPARISON_IGNORES = {"output_dir", "logging_dir", "resume_from_checkpoint"}
ALLOWED_E1_E2_TRAINING_DIFFS = {"neftune_noise_alpha"}


def normalized_text(value: str) -> str:
    """Normalize text conservatively for leakage and duplicate checks."""
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    value = value.replace("……", "…").replace("--", "—")
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value, flags=re.UNICODE)


def text_similarity(left: str, right: str) -> float:
    left_key = normalized_text(left)
    right_key = normalized_text(right)
    if not left_key or not right_key:
        return 0.0
    return SequenceMatcher(None, left_key, right_key, autojunk=False).ratio()


def sha256_file(path: Path) -> str:
    """Hash exact bytes; use this for binary model artifacts."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Hash UTF-8 text after LF normalization for cross-platform provenance."""
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dialogue_prompts(item: Mapping[str, Any]) -> list[str]:
    conversations = item.get("conversations", [])
    return [
        str(message.get("value", ""))
        for message in conversations
        if isinstance(message, Mapping) and message.get("from") in {"human", "user"}
    ]


def primary_prompt(item: Mapping[str, Any]) -> str:
    prompts = dialogue_prompts(item)
    return prompts[0] if prompts else ""


@dataclass(frozen=True)
class LeakageMatch:
    candidate_id: str
    candidate_prompt: str
    reference_id: str
    reference_prompt: str
    similarity: float
    match_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_prompt": self.candidate_prompt,
            "reference_id": self.reference_id,
            "reference_prompt": self.reference_prompt,
            "similarity": round(self.similarity, 6),
            "match_type": self.match_type,
        }


def audit_prompt_leakage(
    candidates: Iterable[Mapping[str, Any]],
    references: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 0.90,
) -> list[LeakageMatch]:
    """Return the strongest exact/near match for each leaking candidate."""
    reference_rows = [
        (
            str(reference.get("id", "unknown")),
            str(reference.get("prompt", primary_prompt(reference))),
        )
        for reference in references
    ]
    matches: list[LeakageMatch] = []
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("id", f"candidate-{index}"))
        candidate_prompt = str(candidate.get("prompt", primary_prompt(candidate)))
        candidate_key = normalized_text(candidate_prompt)
        best: LeakageMatch | None = None
        for reference_id, reference_prompt in reference_rows:
            reference_key = normalized_text(reference_prompt)
            if not candidate_key or not reference_key:
                continue
            exact = candidate_key == reference_key
            similarity = 1.0 if exact else text_similarity(candidate_prompt, reference_prompt)
            if exact or similarity >= threshold:
                current = LeakageMatch(
                    candidate_id=candidate_id,
                    candidate_prompt=candidate_prompt,
                    reference_id=reference_id,
                    reference_prompt=reference_prompt,
                    similarity=similarity,
                    match_type="exact" if exact else "near_duplicate",
                )
                if best is None or current.similarity > best.similarity:
                    best = current
        if best is not None:
            matches.append(best)
    return matches


def compare_experiment_configs(
    e1: Mapping[str, Any],
    e2: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Compare training-relevant fields while ignoring run-specific paths."""
    keys = (set(e1) | set(e2)) - CONFIG_COMPARISON_IGNORES
    keys = {key for key in keys if not key.startswith("_")}
    return {
        key: (e1.get(key), e2.get(key))
        for key in sorted(keys)
        if e1.get(key) != e2.get(key)
    }


def validate_e1_e2_pair(e1: Mapping[str, Any], e2: Mapping[str, Any]) -> list[str]:
    differences = compare_experiment_configs(e1, e2)
    errors: list[str] = []
    if set(differences) != ALLOWED_E1_E2_TRAINING_DIFFS:
        errors.append(
            "E1/E2 training differences must be exactly neftune_noise_alpha; "
            f"found {sorted(differences)}"
        )
    if float(e1.get("neftune_noise_alpha", -1)) != 0.0:
        errors.append("KISAKI-E1 must use neftune_noise_alpha=0.0")
    if float(e2.get("neftune_noise_alpha", -1)) != 5.0:
        errors.append("KISAKI-E2 must use neftune_noise_alpha=5.0")
    if not e1.get("eval_data_path") or not e2.get("eval_data_path"):
        errors.append("Both experiments must use an explicit fixed eval_data_path")
    return errors


R1_VARIANT_DIFFS: dict[str, dict[str, tuple[Any, Any]]] = {
    "e1": {},
    "e2": {"neftune_noise_alpha": (0.0, 5.0)},
    "e3": {"use_dora": (False, True)},
    "e4": {"use_rslora": (False, True)},
    "e5": {"packing": (False, True)},
}


def validate_r1_variant_set(configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Require every R1 variant to differ from E1 by exactly one declared factor."""
    errors: list[str] = []
    missing = set(R1_VARIANT_DIFFS) - set(configs)
    if missing:
        return [f"R1 configs missing: {sorted(missing)}"]
    baseline = configs["e1"]
    for name, expected in R1_VARIANT_DIFFS.items():
        config = configs[name]
        differences = compare_experiment_configs(baseline, config)
        if differences != expected:
            errors.append(
                f"R1-{name.upper()} differences must be {expected}; found {differences}"
            )
        if not config.get("eval_data_path"):
            errors.append(f"R1-{name.upper()} must use an explicit fixed eval_data_path")
    if configs["e3"].get("use_rslora") or configs["e4"].get("use_dora"):
        errors.append("DoRA and RSLoRA must remain separate R1 variants")
    return errors


def validate_frozen_gold(
    data: Mapping[str, Any], *, require_final_held_out: bool = False
) -> list[str]:
    errors: list[str] = []
    if data.get("status") != "frozen":
        errors.append("Gold set status must be 'frozen' for a formal evaluation")
    if require_final_held_out and data.get("evaluation_role") != "final_held_out":
        errors.append("Formal evaluation requires evaluation_role='final_held_out'")
    if require_final_held_out and data.get("formal_use_allowed") is not True:
        errors.append("Formal evaluation requires formal_use_allowed=true")
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        errors.append("Gold set must contain a non-empty prompts list")
    expected_hash = data.get("content_sha256")
    if expected_hash and isinstance(prompts, list):
        actual_hash = canonical_json_hash(prompts)
        if expected_hash != actual_hash:
            errors.append("Gold set content_sha256 does not match prompts")
    elif data.get("status") == "frozen":
        errors.append("Frozen Gold set must include content_sha256")
    return errors


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return None


def environment_snapshot(project_root: Path) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "vllm", "trl", "datasets", "sentence-transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_version = cuda_version = None
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        pass
    gpu = []
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        gpu = [line.strip() for line in output.splitlines() if line.strip()]
    except Exception:
        pass
    return {
        "git_commit": git_commit(project_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": torch_version,
        "cuda": cuda_version,
        "gpus": gpu,
        "packages": packages,
    }
