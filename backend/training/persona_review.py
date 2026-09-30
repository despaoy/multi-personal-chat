"""Bind blinded human decisions to exact candidates; never promote AI scores to approval."""

from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict

from training.persona_reward import DIMENSIONS, VIOLATIONS
from training.persona_sampling import digest, require_text, validate_candidate, validate_scenes
from training.preference_validation import fingerprint, reviewed_pair_digest, validate_pairs


def make_packet(records, *, purpose, seed=42, baseline=None, candidate=None):
    if purpose not in {"preferences", "evaluation"} or not records:
        raise ValueError("nonempty candidates and explicit review purpose are required")
    by_scene, identities, ids = defaultdict(list), {}, set()
    for row in records:
        validate_candidate(row)
        if purpose == "evaluation" and row.get("status") == "source_excerpt":
            raise ValueError("source excerpts are preference anchors, not model evaluation generations")
        if row["candidate_id"] in ids:
            raise ValueError("duplicate candidate id")
        ids.add(row["candidate_id"])
        scene = row["scene"]
        if purpose == "preferences" and scene["split"] != "train":
            raise ValueError("preference construction may only use train scenes, never validation/test")
        if purpose == "evaluation" and scene["split"] not in {"validation", "test"}:
            raise ValueError("blind evaluation requires validation/test scenes")
        if identities.setdefault(row["model"]["name"], row["model"]) != row["model"]:
            raise ValueError("model name has inconsistent endpoint/revision identity")
        by_scene[scene["id"]].append(row)
    validate_scenes([rows[0]["scene"] for rows in by_scene.values()])
    rng, items, key_rows = random.Random(seed), [], {}
    for scene_id in sorted(by_scene):
        rows = sorted(by_scene[scene_id], key=lambda row: row["candidate_id"])
        if len({row["context_sha256"] for row in rows}) != 1:
            raise ValueError("paired candidates must have identical scene/history/evidence")
        if purpose == "evaluation":
            if not baseline or not candidate or baseline == candidate:
                raise ValueError("evaluation needs distinct baseline and candidate names")
            if len(rows) != 2 or {row["model"]["name"] for row in rows} != {baseline, candidate}:
                raise ValueError("each evaluation scene needs exactly one generation per compared model")
            if rows[0]["generation"] != rows[1]["generation"]:
                raise ValueError("evaluation decoding parameters must match")
            pairs = [rows]
        else:
            pairs = itertools.combinations(rows, 2)
        for first, second in pairs:
            if purpose == "preferences" and fingerprint(first["response"]) == fingerprint(second["response"]):
                continue
            a, b = (first, second) if rng.getrandbits(1) else (second, first)
            item_id = digest(sorted([a["candidate_id"], b["candidate_id"]]))
            item = {
                "id": item_id,
                "scene": a["scene"],
                "messages": a["messages"],
                "A": a["response"],
                "B": b["response"],
            }
            item["binding_sha256"] = digest(item)
            items.append(item)
            key_rows[item_id] = {"A": a, "B": b}
    if not items:
        raise ValueError("no distinct candidate pairs available for review")
    packet = {
        "schema": "persona-blind-v1",
        "purpose": purpose,
        "items": items,
        "instructions": "Blind review. Complete dimensions and hard errors before overall choice. Never infer quality from catchphrases alone.",
    }
    packet["packet_sha256"] = digest(packet)
    key = {
        "packet_sha256": packet["packet_sha256"],
        "seed": seed,
        "baseline": baseline,
        "candidate": candidate,
        "rows": key_rows,
    }
    template = {
        "packet_sha256": packet["packet_sha256"],
        "reviewer": "",
        "human_confirmed": False,
        "review_method": "human_only",
        "decisions": [
            {
                "id": item["id"],
                "binding_sha256": item["binding_sha256"],
                "winner": "",
                "dimensions": dict.fromkeys(DIMENSIONS, ""),
                "hard_errors": {"A": None, "B": None},
                "reason": "",
            }
            for item in items
        ],
    }
    return packet, key, template


def validate_packet(packet):
    content = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != digest(content):
        raise ValueError("blind packet hash mismatch")
    ids = set()
    for item in packet["items"]:
        if item["id"] in ids or item["binding_sha256"] != digest(
            {k: v for k, v in item.items() if k != "binding_sha256"}
        ):
            raise ValueError("blind item binding mismatch or duplicate")
        ids.add(item["id"])


def lock_decisions(packet, decisions):
    """Pure validation used by the lock command, which does not read a blind key."""
    validate_packet(packet)
    if decisions.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("decisions belong to another packet")
    require_text(decisions.get("reviewer"), "reviewer")
    if decisions.get("human_confirmed") is not True:
        raise ValueError("human-confirmed decisions are required; AI scores cannot auto-approve")
    if decisions.get("review_method") not in {"human_only", "human_confirmed_ai_assisted"}:
        raise ValueError("explicit human/AI-assisted review method is required")
    rows = decisions.get("decisions")
    if not isinstance(rows, list) or len(rows) != len(packet["items"]):
        raise ValueError("all blind items must have decisions (use invalid to exclude)")
    expected = {item["id"]: item for item in packet["items"]}
    seen = set()
    for row in rows:
        if row.get("id") in seen or row.get("id") not in expected:
            raise ValueError("unknown or duplicate decision id")
        seen.add(row["id"])
        if row.get("binding_sha256") != expected[row["id"]]["binding_sha256"]:
            raise ValueError("decision is not bound to the reviewed responses")
        if row.get("winner") not in {"A", "B", "tie", "invalid"}:
            raise ValueError("invalid overall winner")
        item = expected[row["id"]]
        if fingerprint(item["A"]) == fingerprint(item["B"]) and row["winner"] in {"A", "B"}:
            raise ValueError("identical responses cannot have a strict preference")
        require_text(row.get("reason"), "review reason")
        dimensions = row.get("dimensions")
        if (
            not isinstance(dimensions, dict)
            or set(dimensions) != set(DIMENSIONS)
            or any(value not in {"A", "B", "tie", "not_applicable"} for value in dimensions.values())
        ):
            raise ValueError("all seven dimension comparisons must be completed")
        errors = row.get("hard_errors")
        if not isinstance(errors, dict) or set(errors) != {"A", "B"}:
            raise ValueError("explicit A/B hard-error lists required")
        for values in errors.values():
            if (
                not isinstance(values, list)
                or any(value not in VIOLATIONS for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError("invalid hard-error labels")
    locked = {"schema": "persona-decisions-locked-v1", "review": decisions}
    locked["locked_sha256"] = digest(locked)
    return locked


def reviewed_pairs(packet, locked, key):
    if locked.get("locked_sha256") != digest({k: v for k, v in locked.items() if k != "locked_sha256"}):
        raise ValueError("locked decisions hash mismatch")
    if lock_decisions(packet, locked["review"]) != locked:
        raise ValueError("invalid locked review")
    if key.get("packet_sha256") != packet["packet_sha256"] or set(key["rows"]) != {
        item["id"] for item in packet["items"]
    }:
        raise ValueError("blind key belongs to another packet")
    decisions = {row["id"]: row for row in locked["review"]["decisions"]}
    pairs = []
    for item in packet["items"]:
        mapped = key["rows"][item["id"]]
        if set(mapped) != {"A", "B"}:
            raise ValueError("invalid key sides")
        for side in ("A", "B"):
            row = mapped[side]
            validate_candidate(row)
            if row["response"] != item[side] or row["messages"] != item["messages"] or row["scene"] != item["scene"]:
                raise ValueError("blind key response/context mismatch")
        if digest(sorted([mapped["A"]["candidate_id"], mapped["B"]["candidate_id"]])) != item["id"]:
            raise ValueError("blind key candidate identity mismatch")
        pairs.append((item, decisions[item["id"]], mapped))
    return pairs


def export_preferences(packet, locked, key):
    if packet["purpose"] != "preferences":
        raise ValueError("evaluation packets cannot become preference training data")
    output, seen_prompts, exclusions = [], set(), Counter()
    for item, decision, mapped in reviewed_pairs(packet, locked, key):
        scene, winner = item["scene"], decision["winner"]
        if scene["split"] != "train":
            raise ValueError("only train scenes may be exported")
        if winner not in {"A", "B"}:
            exclusions[winner] += 1
            continue
        if decision["hard_errors"][winner]:
            exclusions["winner_has_hard_error"] += 1
            continue
        if (
            any(row.get("status") == "source_excerpt" for row in mapped.values())
            and mapped[winner].get("status") != "source_excerpt"
        ):
            exclusions["original_not_preferred"] += 1
            continue
        # One pair per prompt preserves the existing trainer's duplicate policy.
        prompt_key = fingerprint(item["messages"])
        if prompt_key in seen_prompts:
            exclusions["additional_pair_same_prompt"] += 1
            continue
        seen_prompts.add(prompt_key)
        loser = "B" if winner == "A" else "A"
        row = {
            "id": item["id"],
            "prompt": item["messages"],
            "chosen": [{"role": "assistant", "content": item[winner]}],
            "rejected": [{"role": "assistant", "content": item[loser]}],
            "review_status": "approved",
            "annotator": "human_blind_review",
            "metadata": {
                "schema_version": "persona-preference-v1",
                "persona": scene["persona"],
                "source_group": scene["source_group"],
                "source_ids": scene["source_ids"],
                "evidence_text_sha256": [fingerprint(e["content"]) for e in scene["evidence"]],
                "split": "train",
                "human_final_approved": True,
                "reviewer": locked["review"]["reviewer"],
                "review_method": locked["review"]["review_method"],
                "review_sha256": locked["locked_sha256"],
                "reason": decision["reason"],
                "chosen_candidate_id": mapped[winner]["candidate_id"],
                "rejected_candidate_id": mapped[loser]["candidate_id"],
                **(
                    {"canonical_source": mapped[winner]["source_reply"]}
                    if mapped[winner].get("status") == "source_excerpt"
                    else {}
                ),
            },
        }
        row["metadata"]["pair_content_sha256"] = reviewed_pair_digest(row)
        output.append(row)
    if output:
        validate_pairs(output, split="train")
    return {"pairs": output, "excluded": dict(exclusions), "status": "reviewed_preferences_not_model_evaluated"}


def summarize_evaluation(packet, locked, key):
    if packet["purpose"] != "evaluation" or not key.get("candidate") or key["candidate"] == key.get("baseline"):
        raise ValueError("requires a baseline/candidate evaluation packet")
    counts, dimensions, groups, errors = Counter(), defaultdict(Counter), defaultdict(list), Counter()
    new_hard_errors = []
    expected_models = {key["baseline"], key["candidate"]}
    pairs = reviewed_pairs(packet, locked, key)
    parents = {item["scene"]["id"]: item["scene"]["id"] for item, _, _ in pairs}

    def root(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    owners = {}
    for item, _, _ in pairs:
        scene = item["scene"]
        dependencies = [
            ("group", scene["source_group"]),
            *(("source", value) for value in scene["source_ids"]),
            *(("evidence", fingerprint(e["content"])) for e in scene["evidence"]),
        ]
        for dependency in dependencies:
            if dependency in owners:
                parents[root(scene["id"])] = root(owners[dependency])
            else:
                owners[dependency] = scene["id"]
    for item, decision, mapped in pairs:
        if {mapped[side]["model"]["name"] for side in ("A", "B")} != expected_models:
            raise ValueError("evaluation model identity mismatch")
        candidate_side = next(side for side in ("A", "B") if mapped[side]["model"]["name"] == key["candidate"])
        baseline_side = "B" if candidate_side == "A" else "A"
        winner = decision["winner"]
        result = "win" if winner == candidate_side else "loss" if winner == baseline_side else winner
        counts[result] += 1
        if result == "invalid":
            continue
        groups[root(item["scene"]["id"])].append({"win": 1, "loss": 0, "tie": 0.5}[result])
        added = set(decision["hard_errors"][candidate_side]) - set(decision["hard_errors"][baseline_side])
        if added:
            new_hard_errors.append({"scene_id": item["scene"]["id"], "violations": sorted(added)})
        for dimension, side in decision["dimensions"].items():
            outcome = "win" if side == candidate_side else "loss" if side == baseline_side else side
            dimensions[dimension][outcome] += 1
        for label, side in (("candidate", candidate_side), ("baseline", baseline_side)):
            for error in decision["hard_errors"][side]:
                errors[f"{label}/{error}"] += 1
    values = list(groups.values())
    # Resample whole scenes/groups rather than treating near-duplicate questions as independent.
    rng, bootstrap = random.Random(42), []
    if len(values) >= 2:
        for _ in range(1000):
            sample = [score for group in rng.choices(values, k=len(values)) for score in group]
            bootstrap.append(sum(sample) / len(sample))
        bootstrap.sort()
    scores = [score for group in values for score in group]
    return {
        "status": "human_reviewed_comparison_not_release_approval",
        "baseline": key["baseline"],
        "candidate": key["candidate"],
        "counts": dict(counts),
        "by_dimension": dict(dimensions),
        "hard_errors": dict(errors),
        "candidate_new_hard_errors": new_hard_errors,
        "no_new_hard_errors": not new_hard_errors,
        "independent_source_groups": len(values),
        "win_share_ties_half": sum(scores) / len(scores) if scores else None,
        "group_bootstrap_95ci": [bootstrap[24], bootstrap[974]] if bootstrap else None,
        "review_method": locked["review"]["review_method"],
        "review_sha256": locked["locked_sha256"],
        "note": "Fixed-history next-response comparison; not a free-running multi-turn PPO evaluation. Small group counts give unstable intervals.",
    }
