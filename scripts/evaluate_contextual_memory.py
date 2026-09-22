"""Compare lexical recall with local-LLM contextual selection on fixed cases.

No gold labels are passed to the model. The bundled cases are synthetic contract
smoke tests, not a held-out research benchmark or evidence of production gains.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from character.evidence_selector import SELECTION_INSTRUCTION, ContextualEvidenceSelector, _local_reviewer  # noqa: E402
from character.memory_service import CharacterMemoryService  # noqa: E402
from character.models import UserScope  # noqa: E402
from evaluation.evidence_selection import (  # noqa: E402
    SelectionObservation,
    paired_selection_metrics,
    selection_metrics,
)


class FixtureRepository:
    def __init__(self, records):
        self.records = records

    async def list_memory_records(self, character_id, scope, limit=100, include_inactive=False):
        # Return all lifecycle states so the real service, not the fixture,
        # performs filtering. Production scope isolation is tested separately.
        return self.records[:limit]


def validate_cases(cases):
    """Reject malformed corpora before loading several GB of model weights."""
    if not cases:
        raise ValueError("evaluation corpus is empty")
    seen = set()
    pairs = Counter()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("case needs a nonempty string ID")
        if case["id"] in seen:
            raise ValueError("duplicate evaluation case ID")
        seen.add(case["id"])
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("case needs a query")
        records, gold = case.get("memories"), case.get("gold_ids")
        if not isinstance(records, list) or not isinstance(gold, list):
            raise ValueError("memories and gold_ids must be lists")
        if any(
            not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() for row in records
        ):
            raise ValueError("memory needs a nonempty string ID")
        ids = [row["id"] for row in records]
        if any(not isinstance(key, str) for key in gold):
            raise ValueError("gold ID must be a string")
        if len(ids) != len(set(ids)) or len(gold) != len(set(gold)) or not set(gold) <= set(ids):
            raise ValueError("invalid memory or gold IDs")
        if "pair_id" in case:
            if not isinstance(case["pair_id"], str) or not case["pair_id"].strip():
                raise ValueError("pair_id must be a nonempty string")
            pairs[case["pair_id"]] += 1
    if any(count != 2 for count in pairs.values()):
        raise ValueError("full fixture must contain exactly two cases per pair")


async def evaluate(
    cases, mode, reviewer=None, embedding_provider=None, capture_responses=False, selection_protocol="direct"
):
    observations = []
    details = []
    responses = []
    pair_counts = Counter(case["pair_id"] for case in cases if "pair_id" in case)

    async def observed_reviewer(messages):
        raw = await (reviewer or _local_reviewer)(messages)
        if capture_responses:
            responses.append(raw[:16000] if isinstance(raw, str) else {"type": type(raw).__name__})
        return raw

    selection_reviewer = observed_reviewer
    if selection_protocol == "structured":
        from evaluation.structured_evidence_review import StructuredEvidenceReviewer

        selection_reviewer = StructuredEvidenceReviewer(observed_reviewer)
    elif selection_protocol == "planned":
        from evaluation.planned_evidence_review import PlannedEvidenceReviewer

        selection_reviewer = PlannedEvidenceReviewer(observed_reviewer)
    elif selection_protocol != "direct":
        raise ValueError("unknown selection protocol")
    selector = (
        ContextualEvidenceSelector(selection_reviewer, timeout_seconds=120 if reviewer else 30)
        if mode in {"local_model", "offline_model"}
        else None
    )
    for case in cases:
        responses.clear()
        key = case["id"]
        records = case["memories"]
        ids = [str(record["id"]) for record in records]
        if len(ids) != len(set(ids)) or not set(case["gold_ids"]) <= set(ids):
            raise ValueError(f"invalid candidate or gold IDs in {key}")
        service = CharacterMemoryService(
            FixtureRepository(records),
            semantic_enabled=embedding_provider is not None,
            embedding_provider=embedding_provider,
        )
        scope = UserScope("eval", "fixture", key, key, "private")
        memories, _ = await service.load_relevant_memories(
            "evaluation",
            scope,
            case["query"],
            for_contextual_selection=selector is not None,
            retrieval_context="\n".join(
                message["content"][-600:] for message in case.get("history", [])[-6:] if message.get("role") == "user"
            )[-1200:],
        )
        candidate_ids = [item.memory_id for item in memories]
        if service._semantic_failure_logged:
            raise RuntimeError("embedding evaluation silently fell back; refusing to report it as semantic recall")
        status, reason = "selected", ""
        if selector is not None:
            outcome = await selector.select(case["query"], memories, history=case.get("history", []))
            memories, status, reason = outcome.memories, outcome.status, outcome.reason
        observation = SelectionObservation(
            key,
            frozenset(case["gold_ids"]),
            tuple(item.memory_id for item in memories),
            status,
            case.get("category", "general"),
            tuple(candidate_ids),
        )
        observations.append(observation)
        details.append(
            {
                "id": key,
                "selected_ids": list(observation.selected_ids),
                "gold_ids": case["gold_ids"],
                "candidate_ids": candidate_ids,
                "status": status,
                "fallback_reason": reason,
                "decisions": list(outcome.decisions) if selector is not None else [],
                **({"model_outputs": list(responses)} if capture_responses else {}),
            }
        )
    return {
        "mode": "hybrid_recall_baseline" if mode == "lexical_baseline" and embedding_provider is not None else mode,
        "embedding_enabled": embedding_provider is not None,
        "selection_protocol": selection_protocol,
        "metrics": selection_metrics(observations),
        "paired_metrics": paired_selection_metrics(
            observations,
            {case["id"]: case["pair_id"] for case in cases if "pair_id" in case and pair_counts[case["pair_id"]] == 2},
        ),
        "incomplete_pairs_not_scored": sorted(key for key, count in pair_counts.items() if count != 2),
        "by_category": {
            category: selection_metrics([row for row in observations if row.category == category])
            for category in sorted({row.category for row in observations})
        },
        "cases": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "backend/evaluation/fixtures/contextual_memory_smoke.jsonl"
    )
    parser.add_argument(
        "--mode", choices=("lexical_baseline", "local_model", "offline_model"), default="lexical_baseline"
    )
    parser.add_argument("--model-path", type=Path, help="Existing local snapshot; required for offline_model")
    parser.add_argument("--decoding", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--thinking", action="store_true", help="Offline sampled Qwen reasoning ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding", choices=("disabled", "minilm"), default="disabled")
    parser.add_argument("--selection-protocol", choices=("direct", "structured", "planned"), default="direct")
    parser.add_argument("--case-id", action="append", help="Explicit diagnostic subset; not a full benchmark")
    parser.add_argument(
        "--include-model-output",
        action="store_true",
        help="Local diagnostic report only; may contain input-derived text",
    )
    parser.add_argument("--output", type=Path, help="Optional NEW report file; never overwrites")
    args = parser.parse_args()
    if args.selection_protocol != "direct" and args.mode == "lexical_baseline":
        parser.error("selection protocol requires a model mode")
    if args.thinking and args.decoding != "sampled":
        parser.error("thinking requires --decoding sampled")
    if args.mode != "offline_model" and (args.thinking or args.decoding != "greedy" or args.seed != 42):
        parser.error("decoding controls apply only to offline_model")
    source = args.input.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    try:
        validate_cases(cases)
    except ValueError as error:
        parser.error(str(error))
    if args.case_id:
        requested = set(args.case_id)
        if not requested <= {case["id"] for case in cases}:
            parser.error("unknown diagnostic case ID")
        cases = [case for case in cases if case["id"] in requested]
    if args.output and args.output.exists():
        parser.error("output already exists")
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/evaluate_contextual_memory.py",
            "backend/character/memory_service.py",
            "backend/character/evidence_selector.py",
            "backend/evaluation/offline_reviewer.py",
            "backend/evaluation/structured_evidence_review.py",
            "backend/evaluation/planned_evidence_review.py",
        )
    }
    reviewer = None
    if args.mode == "offline_model":
        if not args.model_path:
            parser.error("offline_model requires --model-path")
        from evaluation.offline_reviewer import OfflineTransformersReviewer

        reviewer = OfflineTransformersReviewer(
            args.model_path,
            decoding=args.decoding,
            enable_thinking=args.thinking,
            seed=args.seed,
            max_new_tokens=2048 if args.thinking else 1024,
            max_seconds=120 if args.thinking else 90,
        )
    elif args.model_path:
        parser.error("--model-path is only valid with offline_model")
    provider = None
    if args.embedding == "minilm":
        from knowledge.retrieval_core.embedding import SentenceTransformerEmbeddingProvider

        # Keep GPU memory for the generator; no implicit remote model downloads.
        provider = SentenceTransformerEmbeddingProvider(device="cpu")
        provider.embed_query("offline evaluation warmup")
    report = asyncio.run(
        evaluate(cases, args.mode, reviewer, provider, args.include_model_output, args.selection_protocol)
    )
    if provider is not None:
        report["embedding_model"] = {"id": provider.model_id, "fingerprint": provider.model_fingerprint}
    if reviewer is not None:
        report["inference"] = reviewer.metadata
        report["inference_calls"] = reviewer.calls
    instruction = SELECTION_INSTRUCTION
    if args.selection_protocol == "structured":
        from evaluation.structured_evidence_review import INSTRUCTION

        instruction = INSTRUCTION
    elif args.selection_protocol == "planned":
        from evaluation.planned_evidence_review import PLAN_INSTRUCTION

        report["planning_instruction_sha256"] = hashlib.sha256(PLAN_INSTRUCTION.encode("utf-8")).hexdigest()
    report.update(
        {
            "input_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "explicit_diagnostic_subset": args.case_id,
            "selection_instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "evaluation_source_sha256": source_hashes,
            "benchmark_status": "synthetic_smoke_not_research_evidence",
            "quality_claim": "No production-quality improvement is established by these cases.",
        }
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(output + "\n")
    print(output)
    if report["metrics"]["provider_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
