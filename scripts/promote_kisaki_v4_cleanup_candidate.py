#!/usr/bin/env python3
"""Promote the reviewed Kisaki V4 cleanup candidate without enabling training."""

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
V4_DIR = PROJECT_ROOT / "backend/data/character_dialogues/experiments/v4"
DEFAULT_CANDIDATE_MANIFEST = V4_DIR / "cleanup/candidate/manifest.json"
DEFAULT_APPROVAL = V4_DIR / "cleanup/promotion_approval.json"
DEFAULT_CANONICAL_MANIFEST = V4_DIR / "canonical_dataset_manifest.json"
DEFAULT_CANONICAL_TRAIN = V4_DIR / "train.jsonl"
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "backend/data/character_dialogues/experiments/research/research_program_registry_v4.json"
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def canonical_source_distribution(raw: dict[str, int]) -> dict[str, int]:
    constructed = sum(
        count for source, count in raw.items() if source.startswith("llm_v4_")
    )
    grouped = {
        "game_extraction_current_sft": raw.get("game_extraction", 0),
        "llm_v4_reviewed_constructed": constructed,
        "deepseek_user_simulation_v41_reviewed": raw.get(
            "deepseek_user_simulation_v41_reviewed", 0
        ),
        "codex_user_simulation_v41_reviewed": raw.get(
            "codex_user_simulation_v41_reviewed", 0
        ),
    }
    if sum(grouped.values()) != sum(raw.values()):
        raise ValueError(f"unrecognized candidate data sources: {raw}")
    return grouped


def updated_registry(
    registry: dict[str, Any],
    *,
    train_count: int,
    source_distribution: dict[str, int],
    auxiliary_count: int,
) -> dict[str, Any]:
    output = copy.deepcopy(registry)
    output["schema_version"] = max(int(output.get("schema_version", 0)), 6)
    output["status"] = "r0v4_cleanup_pending_refreeze"
    assets = output["active_assets"]
    assets["persona_prompt"]["status"] = "approved_content_policy_alignment_pending"
    assets["persona_prompt"]["formal_use_allowed"] = False
    assets["prompt_policy"].update(
        version="3.3.0", status="dataset_and_review_contract_alignment_pending"
    )
    canonical = assets["canonical_dataset"]
    canonical.update(
        status="frozen_data_pending_gold_and_prompt_alignment",
        train_count=train_count,
        game_train_count=source_distribution["game_extraction_current_sft"],
        constructed_train_count=source_distribution["llm_v4_reviewed_constructed"],
        reviewed_multiturn_augmentation_count=(
            source_distribution["deepseek_user_simulation_v41_reviewed"]
            + source_distribution["codex_user_simulation_v41_reviewed"]
        ),
        technical_auxiliary_count=auxiliary_count,
    )
    assets["gold_v3"].update(
        status="stale_after_canonical_cleanup", formal_use_allowed=False
    )
    for item in output.get("research", []):
        if item.get("id") == "R0V4":
            item["status"] = "reopened_pending_refreeze"
        elif item.get("id") == "R1V4":
            item["status"] = "blocked_until_r0v4_refrozen"
    return output


def promote(
    *,
    candidate_manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    approval_path: Path = DEFAULT_APPROVAL,
    canonical_manifest_path: Path = DEFAULT_CANONICAL_MANIFEST,
    canonical_train_path: Path = DEFAULT_CANONICAL_TRAIN,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    candidate = load_json(candidate_manifest_path)
    approval = load_json(approval_path)
    canonical = load_json(canonical_manifest_path)
    registry = load_json(registry_path)
    candidate_manifest_hash = text_sha256(candidate_manifest_path)

    if approval.get("status") != "approved_for_promotion":
        raise ValueError("cleanup candidate promotion is not authorized")
    if approval.get("manual_record_by_record_review_claimed") is not False:
        raise ValueError("cleanup approval must not overclaim manual row review")
    if approval.get("candidate_manifest_sha256") != candidate_manifest_hash:
        raise ValueError("cleanup approval does not match the candidate manifest")
    if candidate.get("status") != "ready_for_review":
        raise ValueError("cleanup candidate is not ready for promotion")

    candidate_train_path = resolve_project_path(candidate["train"]["path"])
    auxiliary_path = resolve_project_path(
        candidate["formal_training_exclusions"]["path"]
    )
    if text_sha256(candidate_train_path) != candidate["train"]["sha256"]:
        raise ValueError("cleanup candidate train hash mismatch")
    if text_sha256(auxiliary_path) != candidate["formal_training_exclusions"]["sha256"]:
        raise ValueError("cleanup auxiliary data hash mismatch")
    if approval.get("candidate_train_sha256") != candidate["train"]["sha256"]:
        raise ValueError("cleanup approval does not match the candidate train data")

    current_train_hash = text_sha256(canonical_train_path)
    if (
        current_train_hash == candidate["train"]["sha256"]
        and canonical.get("cleanup_revision", {}).get("candidate_manifest_sha256")
        == candidate_manifest_hash
    ):
        return {
            "status": "already_promoted",
            "train_count": canonical["train"]["count"],
            "train_sha256": current_train_hash,
        }
    if current_train_hash != candidate["source"]["canonical_train_sha256"]:
        raise ValueError("active canonical train changed after the cleanup candidate was built")
    if canonical.get("train", {}).get("sha256") != current_train_hash:
        raise ValueError("active canonical manifest does not match its train file")

    source_distribution = canonical_source_distribution(
        candidate["train"]["source_distribution"]
    )
    updated = copy.deepcopy(canonical)
    updated["schema_version"] = max(int(updated.get("schema_version", 0)) + 1, 9)
    updated["status"] = "frozen_data_pending_gold"
    updated["train"].update(
        status="frozen",
        count=candidate["train"]["count"],
        path=project_path(canonical_train_path),
        sha256=candidate["train"]["sha256"],
        assistant_supervision_targets=candidate["train"]["assistant_supervision_targets"],
        source_distribution=source_distribution,
    )
    updated["cleanup_revision"] = {
        "policy_id": "KISAKI-V4-LENGTH-CLEANUP-20260821",
        "status": "promoted",
        "approval_path": project_path(approval_path),
        "candidate_manifest_path": project_path(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_manifest_hash,
        "previous_train_count": candidate["source"]["canonical_train_count"],
        "previous_train_sha256": candidate["source"]["canonical_train_sha256"],
        "formal_train_count": candidate["train"]["count"],
        "formal_train_sha256": candidate["train"]["sha256"],
        "technical_auxiliary_count": candidate["formal_training_exclusions"]["count"],
        "technical_auxiliary_path": project_path(auxiliary_path),
        "technical_auxiliary_sha256": candidate["formal_training_exclusions"]["sha256"],
        "scene_metadata_repair_count": candidate["scene_metadata_repairs"]["count"],
        "target_max_sequence_length": candidate["training_contract"]["token_audit"][
            "max_sequence_length"
        ],
        "maximum_observed_tokens": candidate["training_contract"]["token_audit"][
            "maximum_tokens"
        ],
        "manual_record_by_record_review_claimed": False,
    }
    updated["freeze_blockers"] = [
        "gold_v3_refreeze_after_train_cleanup",
        "prompt_policy_alignment_pending",
    ]
    gold = updated.setdefault("gold_v3", {})
    gold["status"] = "stale_after_training_cleanup"
    gold["formal_use_allowed"] = False
    reaudit = gold.setdefault("contamination_reaudit", {})
    reaudit.update(
        status="stale_after_training_cleanup",
        previous_train_sha256=reaudit.get("train_sha256"),
        expected_train_sha256=candidate["train"]["sha256"],
    )

    new_registry = updated_registry(
        registry,
        train_count=candidate["train"]["count"],
        source_distribution=source_distribution,
        auxiliary_count=candidate["formal_training_exclusions"]["count"],
    )
    candidate_text = candidate_train_path.read_text(encoding="utf-8")
    write_text_atomic(canonical_train_path, candidate_text)
    write_json_atomic(canonical_manifest_path, updated)
    write_json_atomic(registry_path, new_registry)
    return {
        "status": "promoted_pending_refreeze",
        "train_count": updated["train"]["count"],
        "train_sha256": updated["train"]["sha256"],
        "freeze_blockers": updated["freeze_blockers"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--canonical-train", type=Path, default=DEFAULT_CANONICAL_TRAIN)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    try:
        result = promote(
            candidate_manifest_path=args.candidate_manifest.resolve(),
            approval_path=args.approval.resolve(),
            canonical_manifest_path=args.canonical_manifest.resolve(),
            canonical_train_path=args.canonical_train.resolve(),
            registry_path=args.registry.resolve(),
        )
    except (KeyError, OSError, ValueError) as exc:
        print(f"cleanup_promotion_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
