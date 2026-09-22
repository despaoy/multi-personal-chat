"""A separately labelled development ablation, leaving frozen seed runs alone."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from character.evidence_selector import SELECTION_INSTRUCTION  # noqa: E402
from evaluation.temporal_metadata_view import VIEWS, TemporalMetadataReviewer  # noqa: E402
from scripts.evaluate_contextual_memory import evaluate, validate_cases  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--view", choices=sorted(VIEWS), required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not args.thinking and args.seed != 42:
        parser.error("seed is only relevant to sampled thinking in this ablation")
    source = args.input.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    validate_cases(cases)
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/evaluate_memory_metadata_view.py",
            "scripts/evaluate_contextual_memory.py",
            "backend/evaluation/temporal_metadata_view.py",
            "backend/evaluation/offline_reviewer.py",
            "backend/character/evidence_selector.py",
            "backend/character/memory_service.py",
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
    report = asyncio.run(evaluate(cases, "offline_model", TemporalMetadataReviewer(reviewer, args.view)))
    report.update(
        candidate_view=args.view,
        candidate_view_sha256=hashes["backend/evaluation/temporal_metadata_view.py"],
        input_sha256=hashlib.sha256(source.encode()).hexdigest(),
        selection_instruction_sha256=hashlib.sha256(SELECTION_INSTRUCTION.encode()).hexdigest(),
        evaluation_source_sha256=hashes,
        inference=reviewer.metadata,
        inference_calls=reviewer.calls,
        benchmark_status="post_failure_input_representation_development_ablation",
        quality_claim="Post-hoc representation experiment on known development failures; no independent generalization or production claim.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["metrics"], ensure_ascii=False))
    if report["metrics"]["provider_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
