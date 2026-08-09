"""Build the canonical, leakage-audited Kisaki train/validation split.

Source files are never modified. Outputs are deterministic for a fixed source set,
seed and script version. Gold v1 is a development set and is excluded from SFT.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import (  # noqa: E402
    audit_prompt_leakage,
    canonical_json_hash,
    dialogue_prompts,
    normalized_text,
    sha256_text_file,
)

DATA_DIR = BACKEND / "data" / "character_dialogues"
EXPERIMENT_DIR = DATA_DIR / "experiments"
DEFAULT_SOURCES = (
    (DATA_DIR / "tsukiyashiro_kisaki_sft.json", "game_extraction"),
    (DATA_DIR / "kisaki_llm_generated_v3.json", "llm_v3_deepseek"),
)
DEV_GOLD_PATH = BACKEND / "evaluation" / "kisaki_gold_set_v1.json"
TRAIN_PATH = EXPERIMENT_DIR / "tsukiyashiro_kisaki_train.json"
EVAL_PATH = EXPERIMENT_DIR / "tsukiyashiro_kisaki_eval.json"
MANIFEST_PATH = EXPERIMENT_DIR / "canonical_dataset_manifest.json"
EXCLUSIONS_PATH = EXPERIMENT_DIR / "canonical_dataset_exclusions.json"
SEED = 42
EVAL_RATIO = 0.10
NEAR_DUPLICATE_THRESHOLD = 0.90


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _normalize_item(raw: dict[str, Any], source: str) -> dict[str, Any]:
    item = deepcopy(raw)
    role_map = {"gpt": "assistant", "user": "human"}
    for message in item.get("conversations", []):
        message["from"] = role_map.get(message.get("from"), message.get("from"))
    metadata = dict(item.get("metadata") or {})
    metadata["data_source"] = source
    item["metadata"] = metadata
    return item


def _dialogue_key(item: dict[str, Any]) -> str:
    messages = [
        [message.get("from"), normalized_text(message.get("value", ""))]
        for message in item.get("conversations", [])
    ]
    return canonical_json_hash(messages)


def _valid_item(item: dict[str, Any]) -> bool:
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return False
    roles = [message.get("from") for message in conversations]
    if not any(role == "human" for role in roles) or not any(role == "assistant" for role in roles):
        return False
    return all(str(message.get("value", "")).strip() for message in conversations)


def load_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for path, source in DEFAULT_SOURCES:
        data = _read_json(path)
        source_records.append(
            {"path": path.relative_to(PROJECT_ROOT).as_posix(), "count": len(data), "sha256": sha256_text_file(path)}
        )
        for raw in data:
            item = _normalize_item(raw, source)
            if _valid_item(item):
                rows.append(item)
            else:
                exclusions.append({"id": item.get("id"), "reason": "invalid_schema", "source": source})
    return rows, source_records, exclusions


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for item in sorted(rows, key=lambda row: str(row.get("id", ""))):
        key = _dialogue_key(item)
        if key in seen:
            exclusions.append(
                {"id": item.get("id"), "reason": "duplicate_dialogue", "duplicate_of": seen[key]}
            )
            continue
        seen[key] = str(item.get("id", ""))
        kept.append(item)
    return kept, exclusions


def exclude_development_gold(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold = _read_json(DEV_GOLD_PATH)
    prompts = gold.get("prompts", gold)
    kept: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for item in rows:
        probes = [
            {"id": f"{item.get('id')}#turn-{index}", "prompt": prompt}
            for index, prompt in enumerate(dialogue_prompts(item))
        ]
        matches = audit_prompt_leakage(probes, prompts, threshold=NEAR_DUPLICATE_THRESHOLD)
        if matches:
            exclusions.append(
                {
                    "id": item.get("id"),
                    "reason": "development_gold_leakage",
                    "matches": [match.to_dict() for match in matches],
                }
            )
        else:
            kept.append(item)
    return kept, exclusions


def prompt_connected_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows transitively when any normalized user turn is shared."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    prompt_owner: dict[str, int] = {}
    for index, item in enumerate(rows):
        for prompt in dialogue_prompts(item):
            key = normalized_text(prompt)
            if not key:
                continue
            if key in prompt_owner:
                union(index, prompt_owner[key])
            else:
                prompt_owner[key] = index

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(rows):
        groups[find(index)].append(item)
    return sorted(
        groups.values(),
        key=lambda group: min(str(item.get("id", "")) for item in group),
    )


def group_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    eval_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = prompt_connected_groups(rows)
    random.Random(seed).shuffle(groups)
    target_count = max(1, round(len(rows) * eval_ratio))
    selected = 0
    evaluation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for group in groups:
        if selected < target_count:
            evaluation.extend(group)
            selected += len(group)
        else:
            train.extend(group)
    train.sort(key=lambda item: str(item.get("id", "")))
    evaluation.sort(key=lambda item: str(item.get("id", "")))
    return train, evaluation


def build(seed: int = SEED, eval_ratio: float = EVAL_RATIO) -> dict[str, Any]:
    rows, sources, exclusions = load_sources()
    rows, duplicate_exclusions = deduplicate(rows)
    exclusions.extend(duplicate_exclusions)
    rows, gold_exclusions = exclude_development_gold(rows)
    exclusions.extend(gold_exclusions)
    train, evaluation = group_split(rows, seed=seed, eval_ratio=eval_ratio)

    train_prompt_keys = {normalized_text(prompt) for item in train for prompt in dialogue_prompts(item)}
    eval_prompt_keys = {normalized_text(prompt) for item in evaluation for prompt in dialogue_prompts(item)}
    overlap = sorted(key for key in train_prompt_keys & eval_prompt_keys if key)
    if overlap:
        raise RuntimeError(f"prompt groups crossed train/eval split: {len(overlap)}")

    _write_json(TRAIN_PATH, train)
    _write_json(EVAL_PATH, evaluation)
    _write_json(EXCLUSIONS_PATH, {"schema_version": 1, "items": exclusions})

    manifest = {
        "schema_version": 2,
        "dataset_id": "KISAKI-CANONICAL-V2",
        "status": "frozen_for_e1_e2",
        "seed": seed,
        "eval_ratio": eval_ratio,
        "hash_mode": "sha256_utf8_lf_v1",
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "sources": sources,
        "development_gold": {
            "path": DEV_GOLD_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "status": "development_only",
            "sha256": sha256_text_file(DEV_GOLD_PATH),
        },
        "train": {
            "path": TRAIN_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "count": len(train),
            "sha256": sha256_text_file(TRAIN_PATH),
            "source_distribution": dict(Counter(item["metadata"]["data_source"] for item in train)),
        },
        "validation": {
            "path": EVAL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "count": len(evaluation),
            "sha256": sha256_text_file(EVAL_PATH),
            "source_distribution": dict(Counter(item["metadata"]["data_source"] for item in evaluation)),
        },
        "exclusions": {
            "path": EXCLUSIONS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "count": len(exclusions),
            "reason_counts": dict(Counter(item["reason"] for item in exclusions)),
        },
        "checks": {
            "train_validation_prompt_overlap": len(overlap),
            "semantic_gold_v2_audit": "required_before_gold_v2_freeze",
        },
    }
    _write_json(MANIFEST_PATH, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical Kisaki E1/E2 data")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    args = parser.parse_args()
    if not 0 < args.eval_ratio < 0.5:
        parser.error("--eval-ratio must be between 0 and 0.5")
    print(json.dumps(build(args.seed, args.eval_ratio), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
