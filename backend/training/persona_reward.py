"""Auditable reward aggregation/calibration for a future PPO rollout scorer.

Consumes externally judged dimensions, never infers personality from keywords.
This module does not train PPO or claim a judge has been calibrated by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DIMENSIONS = (
    "persona_decision",
    "relationship",
    "emotion",
    "grounding",
    "continuity",
    "naturalness",
    "responsiveness",
)
VIOLATIONS = ("fabricated_fact", "relationship_violation", "state_contradiction")


def aggregate_reward(judgment: dict, weights: dict | None = None) -> dict:
    """Score in [-1, 1]; any declared hard violation receives -1.

    Equal weights are a transparent baseline, not an empirically optimal policy.
    Missing checks and nonfinite scores fail closed instead of becoming zero.
    """
    if not isinstance(judgment.get("judge_id"), str) or not judgment["judge_id"].strip():
        raise ValueError("judge_id is required for reward provenance")
    scores, violations = judgment.get("scores"), judgment.get("violations")
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise ValueError("all seven persona score dimensions are required")
    if not isinstance(violations, dict) or set(violations) != set(VIOLATIONS):
        raise ValueError("all hard-violation checks are required")
    if any(type(value) is not bool for value in violations.values()):
        raise ValueError("hard-violation checks must be explicit booleans")
    for value in scores.values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("dimension scores must be finite numbers in [0, 1]")
    weights = dict.fromkeys(DIMENSIONS, 1.0) if weights is None else weights
    if (
        set(weights) != set(DIMENSIONS)
        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("weights must cover all dimensions and be nonnegative with positive total")
    total = sum(weights.values())
    contributions = {key: scores[key] * weights[key] / total for key in DIMENSIONS}
    failed = [key for key in VIOLATIONS if violations[key]]
    return {
        "reward": -1.0 if failed else 2 * sum(contributions.values()) - 1,
        "hard_failures": failed,
        "weighted_scores": contributions,
        "judge_id": judgment["judge_id"],
        "note": "external judgment aggregation, not independently verified character quality",
    }


def calibrate_pairs(rows: list[dict], weights: dict | None = None) -> dict:
    """Compare judge ranking to held-out human pairwise labels, including ties."""
    if not rows:
        raise ValueError("calibration dataset is empty")
    seen, judges, comparisons = set(), set(), []
    for row in rows:
        if not row.get("id") or row["id"] in seen:
            raise ValueError("calibration pair ids must be nonempty and unique")
        seen.add(row["id"])
        if row.get("split") != "validation" or row.get("human_approved") is not True:
            raise ValueError("calibration requires human-approved validation pairs")
        if not isinstance(row.get("source_group"), str) or not row["source_group"].strip():
            raise ValueError("calibration source_group is required")
        label = row.get("human_preference")
        if label not in {"a", "b", "tie"}:
            raise ValueError("human_preference must be a, b or tie")
        a, b = aggregate_reward(row["a"], weights), aggregate_reward(row["b"], weights)
        if a["judge_id"] != b["judge_id"]:
            raise ValueError("a calibration pair must use the same judge")
        judges.add(a["judge_id"])
        delta = a["reward"] - b["reward"]
        predicted = "tie" if abs(delta) < 1e-8 else ("a" if delta > 0 else "b")
        comparisons.append({"id": row["id"], "human": label, "predicted": predicted, "reward_margin": delta})
    if len(judges) != 1:
        raise ValueError("calibrate each judge version separately")
    return {
        "status": "measured_not_approved_for_ppo",
        "pair_count": len(rows),
        "source_group_count": len({row["source_group"] for row in rows}),
        "judge_id": next(iter(judges)),
        "agreement": sum(item["human"] == item["predicted"] for item in comparisons) / len(rows),
        "judge_ties": sum(item["predicted"] == "tie" for item in comparisons),
        "comparisons": comparisons,
        "note": "Agreement alone does not establish generalization or resistance to reward hacking.",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate externally judged persona rewards against human labels")
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="JSON mapping of all seven dimension weights")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.calibration_data.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    weights = None if args.weights is None else json.loads(args.weights.read_text(encoding="utf-8"))
    report = calibrate_pairs(rows, weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
