"""Isolated state-estimation ablation, not dialogue quality or calibrated confidence.

The fixture's required facets remain evaluator-only. The protected safety route
never reaches the reviewer. Final model outputs are omitted unless explicitly
requested; OfflineTransformersReviewer never returns private thinking drafts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from character.semantic_state_estimator import SemanticStateEstimator, build_semantic_review_messages  # noqa: E402
from character.situation_analyzer import ACT_LABELS, SITUATION_LABELS, SituationAnalyzer  # noqa: E402


def validate_cases(cases):
    if not cases:
        raise ValueError("empty fixture")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("nonempty case ID required")
        if case["id"] in seen:
            raise ValueError("duplicate case ID")
        seen.add(case["id"])
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("nonempty query required")
        acts = case.get("acts")
        if (
            not isinstance(case.get("situation"), str)
            or case["situation"] not in SITUATION_LABELS
            or not isinstance(acts, list)
            or any(not isinstance(act, str) or act not in ACT_LABELS for act in acts)
            or len(acts) != len(set(acts))
        ):
            raise ValueError("unknown required fixture facet")
        if not isinstance(case.get("safety_triggered", False), bool):
            raise ValueError("safety requirement must be boolean")
        history = case.get("history", [])
        if not isinstance(history, list) or any(
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in history
        ):
            raise ValueError("invalid conversation history")


async def evaluate(cases, reviewer, *, review_mode="all_non_safety", include_model_output=False, progress=None):
    validate_cases(cases)
    vocabulary_payload = json.loads(
        build_semantic_review_messages("", [], SituationAnalyzer().estimate(""), reasons=())[1]["content"]
    )
    allowed_acts = sorted(vocabulary_payload["allowed_ids"]["acts"])
    rows = []
    final_outputs = []

    async def record_final(messages):
        output = await reviewer(messages)
        if include_model_output:
            final_outputs.append(output)
        return output

    estimator = SemanticStateEstimator(record_final, timeout_seconds=120, review_mode=review_mode)
    for case in cases:
        final_outputs.clear()
        original = SituationAnalyzer().estimate(case["query"], case.get("history", []))
        result = await estimator.refine_with_diagnostics(case["query"], case.get("history", []), original)
        active = {signal.signal_id for signal in result.state.user_acts if signal.score >= 0.5}
        row = {
            "id": case["id"],
            "status": result.status,
            "fallback_reason": result.fallback_reason,
            "review_reasons": list(result.reasons),
            "rules": asdict(original),
            "reviewed": asdict(result.state),
            "expected_situation": case["situation"],
            "required_acts": case["acts"],
            "missing_required_acts_at_0_5": sorted(set(case["acts"]) - active),
            "safety_required": case.get("safety_triggered", False),
        }
        if include_model_output:
            row["model_final_outputs"] = list(final_outputs)
        rows.append(row)
        if progress is not None:
            progress(row)
        print(f"{case['id']}: {result.status}", flush=True)
    return {
        "benchmark_status": "synthetic_required_facets_not_exhaustive_or_human_adjudicated",
        "review_mode": review_mode,
        "allowed_review_act_ids": allowed_acts,
        "required_acts_unavailable_to_reviewer": sorted(
            {act for case in cases for act in case["acts"]} - set(allowed_acts)
        ),
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "reviewed_cases": sum(row["status"] in {"applied", "fallback"} for row in rows),
            "fallbacks": sum(row["status"] == "fallback" for row in rows),
            "cases_missing_required_acts": sum(bool(row["missing_required_acts_at_0_5"]) for row in rows),
            "situation_matches": sum(row["reviewed"]["primary_situation"] == row["expected_situation"] for row in rows),
            "required_safety_misses": sum(
                row["safety_required"] and not row["reviewed"]["safety_triggered"] for row in rows
            ),
        },
        "limitations": [
            "Required facets are not exhaustive; no precision from extra plausible acts.",
            "Original injection fixture labels greeting incorrectly; retain frozen labels, interpret aggregate cautiously.",
            "Required facets outside the recorded review vocabulary cannot be introduced by decoder changes; protected rule signals may still supply them.",
            "No generation, character-quality rating, LoRA, RAG or database writes.",
            "Thinking changes decoding recipe and budget together; not an isolated thinking-marker effect.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "backend/evaluation/fixtures/contextual_policy_smoke.jsonl"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--review-mode", choices=("selective", "all_non_safety"), default="all_non_safety")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-model-output", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    progress_path = args.output.with_suffix(".progress.jsonl")
    if args.output.exists() or progress_path.exists():
        parser.error("output or progress already exists")
    source = args.input.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    validate_cases(cases)
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/evaluate_semantic_state.py",
            "backend/character/semantic_state_estimator.py",
            "backend/character/situation_analyzer.py",
            "backend/evaluation/offline_reviewer.py",
        )
    }
    from evaluation.offline_reviewer import OfflineTransformersReviewer

    reviewer = OfflineTransformersReviewer(
        args.model_path,
        decoding="sampled" if args.thinking else "greedy",
        enable_thinking=args.thinking,
        seed=args.seed,
        max_new_tokens=2048 if args.thinking else 1024,
        max_seconds=120 if args.thinking else 90,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x", encoding="utf-8") as stream:

        def progress(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

        report = asyncio.run(
            evaluate(
                cases,
                reviewer,
                review_mode=args.review_mode,
                include_model_output=args.include_model_output,
                progress=progress,
            )
        )
    report.update(
        input_sha256=hashlib.sha256(source.encode()).hexdigest(),
        evaluation_source_sha256=sources,
        inference=reviewer.metadata,
        inference_calls=reviewer.calls,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False))
    if report["summary"]["fallbacks"] or report["summary"]["required_safety_misses"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
