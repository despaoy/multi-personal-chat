#!/usr/bin/env python3
"""Build the fixed 30-item development set used to select an E1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json"


def _rank(seed: int, namespace: str, value: str) -> str:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_unique_clusters(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["cluster_id"])].append(row)
    if len(by_cluster) < count:
        raise ValueError(f"{namespace} has only {len(by_cluster)} clusters; need {count}")

    selected: list[dict[str, Any]] = []
    ordered_clusters = sorted(
        by_cluster,
        key=lambda cluster: _rank(seed, f"{namespace}:cluster", cluster),
    )
    for cluster in ordered_clusters[:count]:
        selected.append(
            min(
                by_cluster[cluster],
                key=lambda row: _rank(seed, f"{namespace}:sample", str(row["id"])),
            )
        )
    return selected


def _select_safety_actions(
    rows: Iterable[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row["expected_action"])].append(row)
    expected_actions = {
        "allow",
        "allow_with_confirmation",
        "allow_with_redaction",
        "clarify",
        "clarify_supportive",
        "crisis_support",
        "refuse",
        "safe_alternative",
    }
    if set(by_action) != expected_actions:
        raise ValueError(f"unexpected safety actions: {sorted(by_action)}")
    return [
        min(
            by_action[action],
            key=lambda row: _rank(seed, f"safety:{action}", str(row["id"])),
        )
        for action in sorted(expected_actions)
    ]


def _interleave(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(max(map(len, groups))):
        for group in groups:
            if index < len(group):
                result.append(group[index])
    return result


def build_subset(source: dict[str, Any], *, seed: int = 42) -> dict[str, Any]:
    rows = [
        row
        for row in source.get("prompts", [])
        if row.get("benchmark_suite", "character") == "character"
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)

    persona = _select_unique_clusters(
        by_category["persona"], count=8, seed=seed, namespace="persona"
    )
    factual = _select_unique_clusters(
        by_category["factual"], count=5, seed=seed, namespace="factual"
    ) + _select_unique_clusters(
        by_category["persona_knowledge"],
        count=2,
        seed=seed,
        namespace="persona_knowledge",
    )
    multiturn = _select_unique_clusters(
        by_category["multiturn"], count=7, seed=seed, namespace="multiturn"
    )
    safety = _select_safety_actions(by_category["safety"], seed=seed)
    selected = _interleave([persona, factual, multiturn, safety])

    if len(selected) != 30 or len({row["id"] for row in selected}) != 30:
        raise ValueError("checkpoint development subset must contain 30 unique items")

    return {
        "schema_version": 3,
        "gold_id": "KISAKI-GOLD-V2.1-CHECKPOINT-DEV30",
        "status": "derived_development_subset",
        "evaluation_role": "development_checkpoint_selection",
        "formal_use_allowed": False,
        "character": source.get("character"),
        "persona_key": source.get("persona_key"),
        "source_gold_id": source.get("gold_id"),
        "selection": {
            "method": "sha256_rank_unique_cluster_v1",
            "seed": seed,
            "broad_group_counts": {
                "persona": 8,
                "factual_and_persona_knowledge": 7,
                "multiturn": 7,
                "safety": 8,
            },
            "ordering": "interleaved_persona_factual_multiturn_safety",
            "purpose": "E1 checkpoint selection only; not a formal held-out result",
        },
        "total_prompts": len(selected),
        "category_counts": dict(Counter(row["category"] for row in selected)),
        "prompts": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    result = build_subset(source, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
