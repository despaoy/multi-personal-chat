#!/usr/bin/env python3
"""Align the unchanged Kisaki persona prompt with runtime policy 3.3."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from inference.prompt_policy import PROMPT_POLICY_VERSION  # noqa: E402


V4_DIR = BACKEND_ROOT / "data/character_dialogues/experiments/v4"
DEFAULT_PROMPT = BACKEND_ROOT / "data/character_dialogues/kisaki_system_prompt_v3.txt"
DEFAULT_MANIFEST = V4_DIR / "canonical_dataset_manifest.json"
DEFAULT_REVIEW = (
    PROJECT_ROOT / "docs/research/review_packets/kisaki_v4/review_manifest.json"
)
DEFAULT_REGISTRY = (
    BACKEND_ROOT
    / "data/character_dialogues/experiments/research/research_program_registry_v4.json"
)
DEFAULT_CONFIG_MANIFEST = V4_DIR / "configs/config_manifest.json"
ALIGNMENT_ID = "KISAKI-PROMPT-POLICY-ALIGNMENT-20260821"
COMPOSITION_CONTRACT = (
    "backend/data/character_dialogues/PROMPT_COMPOSITION_CONTRACT.md"
)
PROMPT_PATH = "backend/data/character_dialogues/kisaki_system_prompt_v3.txt"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_atomic(path: Path, value: dict[str, Any]) -> bool:
    rendered = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.read_text(encoding="utf-8") == rendered:
        return False
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return True


def _validate_stable_prompt(prompt_path: Path) -> None:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt or "你是月社妃" not in prompt or "亲生哥哥" not in prompt:
        raise ValueError("canonical Kisaki persona prompt is incomplete")
    forbidden_dynamic_sections = (
        "【当前关系】",
        "【当前情景】",
        "【本轮行为决策】",
        "<speaker_label",
        "<retrieved_evidence",
    )
    found = [section for section in forbidden_dynamic_sections if section in prompt]
    if found:
        raise ValueError(f"stable persona prompt contains dynamic sections: {found}")


def align(
    *,
    prompt_path: Path = DEFAULT_PROMPT,
    manifest_path: Path = DEFAULT_MANIFEST,
    review_path: Path = DEFAULT_REVIEW,
    registry_path: Path = DEFAULT_REGISTRY,
    config_manifest_path: Path = DEFAULT_CONFIG_MANIFEST,
) -> dict[str, Any]:
    _validate_stable_prompt(prompt_path)
    manifest = _load(manifest_path)
    review = _load(review_path)
    registry = _load(registry_path)
    config_manifest = _load(config_manifest_path)

    prompt_contract = manifest.get("prompt_policy")
    if not isinstance(prompt_contract, dict) or prompt_contract.get("path") != PROMPT_PATH:
        raise ValueError("canonical dataset does not reference the reviewed persona prompt")
    review_item = review.get("approval", {}).get("items", {}).get("system_prompt_v3")
    if not isinstance(review_item, dict) or review_item.get("status") != "approved":
        raise ValueError("persona prompt content is not approved")
    if review_item.get("path") != PROMPT_PATH:
        raise ValueError("reviewed persona prompt path does not match the dataset")

    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["schema_version"] = max(
        int(updated_manifest.get("schema_version", 0)), 11
    )
    updated_manifest["prompt_policy"].update(
        version=PROMPT_POLICY_VERSION,
        persona_revision="v3",
        composition_contract_path=COMPOSITION_CONTRACT,
        stable_persona_content_changed=False,
        dynamic_context_source="backend/character/context_builder.py",
        speaker_context_location="untrusted_user_region",
        alignment_id=ALIGNMENT_ID,
    )
    updated_manifest["freeze_blockers"] = [
        blocker
        for blocker in updated_manifest.get("freeze_blockers", [])
        if blocker != "prompt_policy_alignment_pending"
    ]

    updated_review = copy.deepcopy(review)
    updated_review_item = updated_review["approval"]["items"]["system_prompt_v3"]
    updated_review_item["prompt_policy_version"] = PROMPT_POLICY_VERSION
    updated_review_item["policy_alignment"] = {
        "id": ALIGNMENT_ID,
        "status": "aligned_without_persona_content_change",
        "composition_contract_path": COMPOSITION_CONTRACT,
        "speaker_boundary": "untrusted_user_region",
        "persona_content_changed": False,
    }

    updated_registry = copy.deepcopy(registry)
    updated_registry["schema_version"] = max(
        int(updated_registry.get("schema_version", 0)), 7
    )
    updated_registry["status"] = "r0v4_gold_refreeze_pending"
    assets = updated_registry["active_assets"]
    assets["persona_prompt"].update(
        policy_version=PROMPT_POLICY_VERSION,
        status="approved_policy_aligned",
        formal_use_allowed=True,
        composition_contract_path=COMPOSITION_CONTRACT,
    )
    assets["prompt_policy"].update(
        version=PROMPT_POLICY_VERSION,
        status="active",
        alignment_id=ALIGNMENT_ID,
    )
    assets["canonical_dataset"]["status"] = "frozen_data_pending_gold"
    for research in updated_registry.get("research", []):
        if research.get("id") == "R0V4":
            research["status"] = "gold_refreeze_pending"
        elif research.get("id") == "R1V4":
            research["status"] = "blocked_until_r0v4_refrozen"

    updated_config_manifest = copy.deepcopy(config_manifest)
    updated_config_manifest["schema_version"] = max(
        int(updated_config_manifest.get("schema_version", 0)), 2
    )
    updated_config_manifest.update(
        status="stale_after_canonical_cleanup",
        formal_use_allowed=False,
        required_prompt_policy_version=PROMPT_POLICY_VERSION,
        regeneration_condition="canonical_dataset_frozen_without_blockers",
    )
    config_review = updated_review["approval"]["items"].get("experiment_configs")
    if isinstance(config_review, dict):
        config_review["formal_use_allowed"] = False
        config_review["regeneration_status"] = "pending_after_r0v4_refreeze"

    changed = []
    for path, value in (
        (manifest_path, updated_manifest),
        (review_path, updated_review),
        (registry_path, updated_registry),
        (config_manifest_path, updated_config_manifest),
    ):
        if _write_atomic(path, value):
            changed.append(path.name)
    return {
        "status": "aligned" if changed else "already_aligned",
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "persona_content_changed": False,
        "remaining_freeze_blockers": updated_manifest["freeze_blockers"],
        "stale_configs_marked": True,
        "changed_files": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--config-manifest", type=Path, default=DEFAULT_CONFIG_MANIFEST
    )
    args = parser.parse_args()
    try:
        result = align(
            prompt_path=args.prompt.resolve(),
            manifest_path=args.manifest.resolve(),
            review_path=args.review.resolve(),
            registry_path=args.registry.resolve(),
            config_manifest_path=args.config_manifest.resolve(),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"prompt_policy_alignment_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
