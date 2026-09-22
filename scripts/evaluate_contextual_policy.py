"""Paired persona-policy pilot: constraints and sensitivity, NOT answer quality.

Supplied hints isolate policy; inferred modes exercise the upstream estimator.
A persona-dependent strategy change is descriptive, not automatically desirable.
No reference label or expected action is shown to the reviewing model.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from character.context_builder import _compact_dynamic_projection  # noqa: E402
from character.contextual_policy import INSTRUCTION, ContextualDecisionPolicy  # noqa: E402
from character.decision_policy import STRATEGY_INSTRUCTIONS, DecisionPolicy  # noqa: E402
from character.models import CharacterProfile, InteractionState, RelationshipState, WeightedSignal  # noqa: E402
from character.semantic_state_estimator import SemanticStateEstimator  # noqa: E402
from character.situation_analyzer import ACT_LABELS, SITUATION_LABELS, SituationAnalyzer  # noqa: E402

PROFILES = (
    CharacterProfile(
        "reserved",
        "沉静型实验角色",
        traits=("克制", "不轻易自我表露"),
        values=("尊重个人空间", "认真倾听"),
        boundaries=("不通过追问强迫别人交流",),
    ),
    CharacterProfile(
        "playful",
        "活泼型实验角色",
        traits=("轻快", "喜欢适度调侃", "乐于表达态度"),
        values=("共同探索", "尊重对方自主选择"),
        boundaries=("对方不愿意时不继续调侃或追问",),
    ),
)


def constraint_violations(strategy_ids, acts, *, safety_required=False):
    chosen = set(strategy_ids)
    problems = []
    if safety_required and "ensure_safety" not in chosen:
        problems.append("required_safety_route_missing")
    if "information_request" in acts and "respond_directly" not in chosen:
        problems.append("explicit_question_dropped")
    if "advice_request" in acts and "advice_boundary" not in acts and "offer_suggestion" not in chosen:
        problems.append("explicit_advice_dropped")
    if "advice_boundary" in acts and chosen & {"offer_suggestion", "gentle_probe", "clarify_need"}:
        problems.append("no_advice_boundary_violated")
    if "recall_shared_context" in chosen:
        problems.append("memory_recalled_without_evidence")
    if (
        "closing" in acts
        and not {"information_request", "advice_request"} & set(acts)
        and chosen & {"offer_suggestion", "gentle_probe", "clarify_need"}
    ):
        problems.append("conversation_close_violated")
    return problems


async def evaluate(cases, reviewer, state_source="supplied"):
    if state_source not in {"supplied", "rules", "semantic", "semantic_all"}:
        raise ValueError("unknown state source")
    policy = ContextualDecisionPolicy(reviewer, timeout_seconds=120)
    estimator = SemanticStateEstimator(
        reviewer, timeout_seconds=120, review_mode="all_non_safety" if state_source == "semantic_all" else "selective"
    )
    rows = []
    state_rows = []
    for case in cases:
        if case["situation"] not in SITUATION_LABELS or not set(case["acts"]) <= set(ACT_LABELS):
            raise ValueError("fixture contains unknown situation or act labels")
        interaction = InteractionState(
            primary_situation=case["situation"],
            confidence=0.9,
            user_acts=tuple(WeightedSignal(act, 0.9) for act in case["acts"]),
            safety_triggered=case.get("safety_triggered", False),
        )
        state_status = "supplied"
        state_reason = ""
        review_reasons = ()
        if state_source != "supplied":
            # Gold acts/situation remain evaluator-only in inferred modes.
            interaction = SituationAnalyzer().estimate(case["query"], case.get("history", []))
            state_status = "rules"
            if state_source in {"semantic", "semantic_all"}:
                reviewed = await estimator.refine_with_diagnostics(case["query"], case.get("history", []), interaction)
                interaction = reviewed.state
                state_status = reviewed.status
                state_reason = reviewed.fallback_reason
                review_reasons = reviewed.reasons
        inferred_acts = {signal.signal_id: signal.score for signal in interaction.user_acts}
        # Fixture acts are required facets, not an exhaustive annotation of all
        # permissible acts. Do not misreport extra valid facets as false positives.
        expected_acts = set(case["acts"])
        active_acts = {key for key, score in inferred_acts.items() if score >= 0.5}
        state_rows.append(
            {
                "id": case["id"],
                "status": state_status,
                "fallback_reason": state_reason,
                "review_reasons": list(review_reasons),
                "expected_situation": case["situation"],
                "inferred_situation": interaction.primary_situation,
                "required_acts": sorted(expected_acts),
                "inferred_acts": inferred_acts,
                "missing_required_acts_at_0_5": sorted(expected_acts - active_acts),
                "safety_required": case.get("safety_triggered", False),
                "safety_triggered": interaction.safety_triggered,
            }
        )
        for profile in PROFILES:
            relationship = RelationshipState()
            baseline = DecisionPolicy().decide(
                profile, relationship, interaction.primary_situation, interaction=interaction
            )
            outcome = await policy.refine(
                baseline,
                query=case["query"],
                history=case.get("history", []),
                profile=profile,
                relationship=relationship,
                interaction=interaction,
                has_relevant_memory=False,
            )
            priorities, uncertain = _compact_dynamic_projection(interaction, outcome.plan)
            baseline_priorities, _ = _compact_dynamic_projection(interaction, baseline)
            baseline_projected_ids = [
                key
                for key, instruction in STRATEGY_INSTRUCTIONS.items()
                if any(instruction in priority for priority in baseline_priorities)
            ]
            projected_ids = [
                key
                for key, instruction in STRATEGY_INSTRUCTIONS.items()
                if any(instruction in priority for priority in priorities)
            ]
            rows.append(
                {
                    "id": case["id"],
                    "profile": profile.character_id,
                    "baseline_strategy_ids": list(baseline.strategy_ids),
                    "selected_strategy_ids": list(outcome.plan.strategy_ids),
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "state_source": state_source,
                    "state_review_status": state_status,
                    "state_review_reason": state_reason,
                    "inferred_situation": interaction.primary_situation,
                    "inferred_acts": {signal.signal_id: signal.score for signal in interaction.user_acts},
                    "baseline_constraint_violations": constraint_violations(
                        baseline_projected_ids, case["acts"], safety_required=case.get("safety_triggered", False)
                    ),
                    "projected_priorities": priorities,
                    "uncertain": uncertain,
                    "constraint_violations": constraint_violations(
                        outcome.plan.strategy_ids, case["acts"], safety_required=case.get("safety_triggered", False)
                    ),
                    "projected_strategy_ids": projected_ids,
                    "projection_constraint_violations": constraint_violations(
                        projected_ids, case["acts"], safety_required=case.get("safety_triggered", False)
                    ),
                }
            )
            print(f"{case['id']}/{profile.character_id}: {outcome.status}", flush=True)
    comparable = changed = 0
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        if first["status"] == second["status"] == "applied":
            comparable += 1
            changed += first["selected_strategy_ids"] != second["selected_strategy_ids"]
    return {
        "benchmark_status": "synthetic_policy_contract_and_sensitivity_pilot",
        "quality_claim": "Persona sensitivity is not a preference or generated-answer quality metric.",
        "supplied_state_hints": state_source == "supplied",
        "state_source": state_source,
        "state_cases": state_rows,
        "state_diagnostics": {
            "case_count": len(state_rows),
            "scoring_note": "Required-act coverage only; synthetic labels are not exhaustive. Supplied mode is not an estimator test.",
            "required_act_score_threshold": 0.5,
            "cases_missing_required_acts": sum(bool(row["missing_required_acts_at_0_5"]) for row in state_rows),
            "situation_matches": sum(row["expected_situation"] == row["inferred_situation"] for row in state_rows),
            "reviewed_cases": sum(row["status"] in {"applied", "fallback"} for row in state_rows),
        },
        "cases": rows,
        "summary": {
            "decisions": len(rows),
            "fallbacks": sum(row["status"] == "fallback" for row in rows),
            "state_review_fallback_cases": sum(row["state_review_status"] == "fallback" for row in rows[::2]),
            "baseline_projection_constraint_violations": sum(
                bool(row["baseline_constraint_violations"]) for row in rows
            ),
            "constraint_violations": sum(bool(row["constraint_violations"]) for row in rows),
            "projection_constraint_violations": sum(bool(row["projection_constraint_violations"]) for row in rows),
            "comparable_persona_pairs": comparable,
            "different_strategy_pairs": changed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "backend/evaluation/fixtures/contextual_policy_smoke.jsonl"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--state-source", choices=("supplied", "rules", "semantic", "semantic_all"), default="supplied")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    source = args.input.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("empty corpus or duplicate case IDs")
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/evaluate_contextual_policy.py",
            "backend/character/contextual_policy.py",
            "backend/character/context_builder.py",
            "backend/character/decision_policy.py",
            "backend/character/situation_analyzer.py",
            "backend/character/semantic_state_estimator.py",
            "backend/evaluation/offline_reviewer.py",
        )
    }
    from evaluation.offline_reviewer import OfflineTransformersReviewer

    reviewer = OfflineTransformersReviewer(args.model_path)
    report = asyncio.run(evaluate(cases, reviewer, args.state_source))
    report.update(
        input_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        policy_instruction_sha256=hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest(),
        evaluation_source_sha256=source_hashes,
        inference=reviewer.metadata,
        inference_calls=reviewer.calls,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if (
        report["summary"]["fallbacks"]
        or report["summary"]["state_review_fallback_cases"]
        or report["summary"]["constraint_violations"]
        or report["summary"]["projection_constraint_violations"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
