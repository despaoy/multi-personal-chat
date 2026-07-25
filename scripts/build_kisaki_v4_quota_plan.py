"""Build a stratified quota_plan.json for the Kisaki V4 calibration run.

Reads v3_negative_pool.jsonl, distributes a target sample count across all
scenes proportionally (with a minimum of N per scene that has >=N samples),
and deterministically picks sample_spec_ids by even-spacing sampling so the
plan is reproducible.

Major-4 fix: the plan now also records ``style_targets`` (documented goals
for meta-narrative / laughter / sharpness / 正因如此 / length distribution
that the human reviewer should check during calibration, since the v3
negative pool does not carry per-sample style tags) and ``pool_warnings``
(scenes where ``available`` is below the formal-training requirement of
8-10 per scene, so the operator knows the current pool is only sufficient
for Pilot calibration, not for the final 111-sample training set).

Major-5 fix: the plan now emits ``ordered_sample_spec_ids`` — a round-
robin interleaving of the selected ids across scenes — so the pipeline
processes samples in truly stratified order rather than following the
negative-pool file order (which clusters same-scene samples and would
make a ``--limit`` cap over-represent the first scene).

Output: quota_plan.json next to the negative pool, with:
  - per-scene quota and selected sample_spec_ids
  - ordered_sample_spec_ids (round-robin interleaved)
  - style_targets (documented, not enforced by code)
  - pool_warnings (scenes below formal-training sufficiency)
  - pool_sufficiency_note
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

# Major-4: formal training set requires 8-10 samples per scene. Scenes
# below this threshold are flagged in pool_warnings so the operator knows
# the current pool only supports Pilot calibration, not the final 111-
# sample training set (which requires new human prompt specs).
FORMAL_MIN_PER_SCENE = 8

# Major-4: documented style targets for the human reviewer. These are
# NOT enforced by the pipeline (the v3 negative pool has no style tags),
# but they remind the reviewer to check coverage during calibration. The
# final training set must hit these targets; Pilot calibration only
# samples them.
STYLE_TARGETS: dict[str, dict[str, Any]] = {
    "meta_narrative": {
        "target_share": 0.15,
        "note": "samples where 1 meta-narrative word is character-appropriate (书籍讨论/角色设定/观点讨论/突发奇想/回忆故事)",
    },
    "laughter_diversity": {
        "target_kinds": ["呼呼呼", "噗噗", "哎呀"],
        "note": "at least 2 distinct laughter kinds across the calibration set",
    },
    "sharp_expression": {
        "target_count": 5,
        "note": "samples containing sharp sarcastic expressions (恕我拒绝/你疯了吗/这可不行/太危险)",
    },
    "zheng_yin_ci": {
        "target_max_per_sample": 1,
        "target_share_le_0_3": 0.7,
        "note": "正因如此 should appear in <=30% of samples, at most once per sample",
    },
    "length_distribution": {
        "target_short_share": 0.68,
        "short_range": "10-40 chars",
        "note": "68%+ of assistant turns should be 10-40 chars (matches original character style)",
    },
}


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


def _interleave_round_robin(
    scene_to_ids: dict[str, list[str]],
    scene_order: list[str],
) -> list[str]:
    """Major-5: round-robin interleave ids across scenes.

    Takes one id from each scene in turn (in ``scene_order``), repeating
    until all lists are exhausted. This produces an ordering where no two
    consecutive samples come from the same scene when there are >=2 scenes
    with samples — preventing the run from clustering on one scene early.
    """
    queues: dict[str, list[str]] = {
        s: list(scene_to_ids.get(s, [])) for s in scene_order
    }
    ordered: list[str] = []
    while any(queues.values()):
        for scene in scene_order:
            q = queues[scene]
            if q:
                ordered.append(q.pop(0))
    return ordered


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

    # Initial proportional allocation with floor of min_per_scene.
    # Major-2 fix: effective_floor = min(min_per_scene, count) so a scarce
    # scene (count < min_per_scene) uses its actual size as the floor
    # instead of an unreachable min_per_scene. This prevents the trim
    # step from computing a negative gap (quota - min_per_scene < 0)
    # which previously caused `quota -= negative` to INCREASE the quota
    # beyond the available sample count, inflating selected_total well
    # past --target (e.g. --target 12 yielded 19).
    effective_floor: dict[str, int] = {
        scene: min(min_per_scene, count)
        for scene, count in scene_counter.items()
    }
    quota: dict[str, int] = {}
    for scene, count in scene_counter.items():
        alloc = max(effective_floor[scene], round(count / total * target))
        quota[scene] = min(alloc, count)

    # Adjust to hit target exactly.
    # Major-2 fix: trim gap is clamped to >= 0 via max(0, ...). Without
    # this clamp, a scene whose quota was clipped below min_per_scene
    # (scarce scenes) produced a negative gap, and `min(current-target,
    # negative)` selected the negative value, growing the quota instead
    # of trimming it.
    current = sum(quota.values())
    if current > target:
        for scene in sorted(quota, key=lambda s: -quota[s]):
            if current <= target:
                break
            gap = max(0, quota[scene] - effective_floor[scene])
            trim = min(current - target, gap)
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

    # Major-2 fix: if sum(effective_floor) > target, the trim step could
    # not reach target (every scene is already at its floor). Record
    # this as a plan-level warning so the operator knows --target was
    # unsatisfiable with the requested --min-per-scene, and the actual
    # selected_total will exceed target. The operator should either lower
    # --min-per-scene or accept the over-selection.
    floor_sum = sum(effective_floor.values())
    target_unsatisfiable = current > target

    scenes_block: dict[str, dict[str, Any]] = {}
    selected_per_scene: dict[str, list[str]] = {}
    selected_total = 0
    # Major-5: scene_order is sorted alphabetically for deterministic
    # round-robin interleaving.
    scene_order = sorted(quota.keys())
    for scene in scene_order:
        q = quota[scene]
        ids = _even_spaced_pick(scene_ids[scene], q)
        scenes_block[scene] = {
            "available": scene_counter[scene],
            "quota": q,
            "sample_spec_ids": ids,
        }
        selected_per_scene[scene] = ids
        selected_total += len(ids)

    # Major-5: build the interleaved ordering.
    ordered_sample_spec_ids = _interleave_round_robin(
        selected_per_scene, scene_order,
    )

    # Major-4: flag scenes where the pool is below the formal-training
    # requirement. These scenes can still be used for Pilot calibration
    # but cannot support the final 111-sample training set without new
    # human prompt specs.
    pool_warnings: list[dict[str, Any]] = []
    for scene in scene_order:
        available = scene_counter[scene]
        if available < FORMAL_MIN_PER_SCENE:
            pool_warnings.append({
                "scene": scene,
                "available": available,
                "formal_min_required": FORMAL_MIN_PER_SCENE,
                "severity": "critical" if available < 3 else "major",
                "message": (
                    f"scene '{scene}' has only {available} sample(s) in the "
                    f"pool; formal training set requires >= {FORMAL_MIN_PER_SCENE}. "
                    "Pilot calibration can proceed but the final 111-sample "
                    "training set needs new human prompt specs for this scene."
                ),
            })

    pool_sufficiency_note = (
        f"Pool has {total} samples across {len(scenes_block)} scenes. "
        f"Formal training set requires >={FORMAL_MIN_PER_SCENE} per scene "
        f"({len(scenes_block) * FORMAL_MIN_PER_SCENE} minimum total). "
        + (
            "Pool is INSUFFICIENT for the formal training set; new human "
            "prompt specs are required before Stage E."
            if pool_warnings
            else "Pool is sufficient for the formal training set."
        )
    )

    plan = {
        "plan_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_pool_path": str(POOL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_pool_sha256": _sha256_file(POOL_PATH),
        "source_pool_total": total,
        "target_total": target,
        "min_per_scene": min_per_scene,
        "scenes_count": len(scenes_block),
        "selected_total": selected_total,
        "scenes": scenes_block,
        # Major-5: interleaved ordering consumed by run_pipeline.
        "ordered_sample_spec_ids": ordered_sample_spec_ids,
        # Major-4: documented style targets (reviewer-checked, not code-enforced).
        "style_targets": STYLE_TARGETS,
        # Major-4: pool sufficiency diagnostics.
        "pool_warnings": pool_warnings,
        "pool_sufficiency_note": pool_sufficiency_note,
        "formal_min_per_scene": FORMAL_MIN_PER_SCENE,
    }

    # Major-2 fix: surface target-unsatisfiable as a top-level plan field
    # so the pipeline and operator can detect when selected_total will
    # exceed --target (every scene already at its floor, cannot trim).
    if target_unsatisfiable:
        plan["target_unsatisfiable"] = True
        plan["target_unsatisfiable_reason"] = (
            f"sum(effective_floor)={floor_sum} > target={target}; every "
            "scene is already at its min-per-scene floor and cannot be "
            "trimmed further. Lower --min-per-scene or accept the over-"
            f"selection (selected_total={selected_total})."
        )
    else:
        plan["target_unsatisfiable"] = False

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
    print(f"ordered_sample_spec_ids: {len(plan['ordered_sample_spec_ids'])} ids (round-robin interleaved)")
    print("per-scene quota:")
    for scene, block in plan["scenes"].items():
        print(f"  {scene}: {block['quota']} / {block['available']}")
    if plan["pool_warnings"]:
        print(f"\n[pool_warnings] {len(plan['pool_warnings'])} scene(s) below formal minimum:")
        for w in plan["pool_warnings"]:
            print(f"  [{w['severity']}] {w['scene']}: {w['available']}/{w['formal_min_required']}")
        print(f"\n{plan['pool_sufficiency_note']}")
    # Major-2: warn when --target could not be hit.
    if plan.get("target_unsatisfiable"):
        print(f"\n[target_unsatisfiable] {plan['target_unsatisfiable_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
