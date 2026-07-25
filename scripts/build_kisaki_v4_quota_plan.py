"""Build a stratified quota_plan.json for the Kisaki V4 calibration run.

Reads v3_negative_pool.jsonl, distributes a target sample count across all
scenes proportionally (with a minimum of 2 per scene that has >=2 samples),
and deterministically picks sample_spec_ids by even-spacing sampling so the
plan is reproducible.

Output: quota_plan.json next to the negative pool, with:
  - per-scene quota and selected sample_spec_ids
  - SHA256 of the source pool (for run_manifest provenance)
  - plan_version and created_at

Usage:
  python scripts/build_kisaki_v4_quota_plan.py --target 30
  python scripts/build_kisaki_v4_quota_plan.py --target 50 --min-per-scene 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = (
    PROJECT_ROOT / "backend" / "data" / "character_dialogues"
    / "experiments" / "v3" / "llm_v4_judged" / "v3_negative_pool.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "backend" / "data" / "character_dialogues"
    / "experiments" / "v3" / "llm_v4_judged" / "quota_plan.json"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _even_spaced_pick(items: list[str], k: int) -> list[str]:
    """Pick k items from a list by even spacing (deterministic, reproducible).

    For k=1 picks the middle; for k>=len picks all. Otherwise picks indices
    at i * len / k for i in [0, k) — gives uniform coverage of the list.
    """
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 0:
        return []
    return [items[int(i * n / k)] for i in range(k)]


def build_plan(target: int, min_per_scene: int) -> dict[str, Any]:
    scene_counter: Counter[str] = Counter()
    scene_ids: dict[str, list[str]] = {}
    with POOL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scene = rec.get("scene", "未知")
            sid = rec.get("sample_spec_id", "")
            scene_counter[scene] += 1
            scene_ids.setdefault(scene, []).append(sid)

    # Sort ids for determinism
    for scene in scene_ids:
        scene_ids[scene].sort()
    total = sum(scene_counter.values())

    # Initial proportional allocation with floor of min_per_scene
    quota: dict[str, int] = {}
    for scene, count in scene_counter.items():
        alloc = max(min_per_scene, round(count / total * target))
        quota[scene] = min(alloc, count)

    # Adjust to hit target exactly
    current = sum(quota.values())
    if current > target:
        for scene in sorted(quota, key=lambda s: -quota[s]):
            if current <= target:
                break
            trim = min(current - target, quota[scene] - min_per_scene)
            quota[scene] -= trim
            current -= trim
    elif current < target:
        for scene in sorted(quota, key=lambda s: -scene_counter[s]):
            if current >= target:
                break
            room = scene_counter[scene] - quota[scene]
            add = min(target - current, room)
            quota[scene] += add
            current += add

    scenes_block: dict[str, dict[str, Any]] = {}
    selected_total = 0
    for scene in sorted(quota):
        q = quota[scene]
        ids = _even_spaced_pick(scene_ids[scene], q)
        scenes_block[scene] = {
            "available": scene_counter[scene],
            "quota": q,
            "sample_spec_ids": ids,
        }
        selected_total += len(ids)

    plan = {
        "plan_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_pool_path": str(POOL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_pool_sha256": _sha256_file(POOL_PATH),
        "source_pool_total": total,
        "target_total": target,
        "min_per_scene": min_per_scene,
        "scenes_count": len(scenes_block),
        "selected_total": selected_total,
        "scenes": scenes_block,
    }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=30,
                        help="Target total samples (default 30 for calibration)")
    parser.add_argument("--min-per-scene", type=int, default=2,
                        help="Minimum samples per scene (default 2)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    plan = build_plan(args.target, args.min_per_scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"=== Wrote quota_plan to {args.output} ===")
    print(f"source_pool_total: {plan['source_pool_total']}")
    print(f"target_total: {plan['target_total']}")
    print(f"selected_total: {plan['selected_total']}")
    print(f"scenes_count: {plan['scenes_count']}")
    print("per-scene quota:")
    for scene, block in plan["scenes"].items():
        print(f"  {scene}: {block['quota']} / {block['available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
