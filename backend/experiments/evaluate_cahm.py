"""Run the four CAHM ablations on the lightweight Chinese memory Gold Set."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from character.memory_extractor import extract_memories  # noqa: E402
from character.memory_llm import (  # noqa: E402
    MAX_EXISTING_MEMORIES,
    MemoryLlmConfig,
    OpenAICompatibleMemoryCompletion,
    build_memory_llm_messages,
    parse_llm_proposals,
)
from character.memory_service import (  # noqa: E402
    HYBRID_WEIGHT_IMPORTANCE,
    HYBRID_WEIGHT_LEXICAL,
    HYBRID_WEIGHT_RECENCY,
    HYBRID_WEIGHT_SEMANTIC,
    MIN_HYBRID_MEMORY_SCORE,
    SEMANTIC_MEMORY_CANDIDATE_LIMIT,
    CharacterMemoryService,
)
from character.models import UserScope  # noqa: E402
from knowledge.retrieval_core.embedding import get_default_embedding_provider  # noqa: E402

DEFAULT_DATASET = BACKEND_ROOT / "evaluation" / "data" / "cahm_memory_gold.jsonl"
DEFAULT_OUTPUT = BACKEND_ROOT / "evaluation" / "results" / "cahm_ablation_results.json"


@dataclass(frozen=True)
class Ablation:
    name: str
    context_extraction: bool
    semantic_retrieval: bool
    confidence_gate: bool


ABLATIONS = (
    Ablation("A. Baseline", False, False, False),
    Ablation("B. Semantic Retrieval", False, True, False),
    Ablation("C. Context Extraction", True, False, False),
    Ablation("D. Full CAHM", True, True, True),
)


class _CaseRepository:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.records = [{**record, "updated_at": record.get("updated_at", now)} for record in records]

    async def list_memory_records(self, character_id, user_scope, limit=30):
        return self.records[:limit]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _gold_extraction_labels(case: dict[str, Any]) -> set[str]:
    if "gold_keys" in case:
        return {f"key:{value}" for value in case["gold_keys"]}
    return {f"semantic:{kind}:{value}" for kind, value in zip(case.get("gold_kinds", []), case.get("gold_values", []))}


def _rule_label(item, *, semantic: bool) -> str:
    key = item.memory_key
    if semantic and key.startswith(("goal_", "preference_", "promise_")):
        kind = key.split("_", 1)[0]
        if kind == "preference":
            kind = "dislike" if "不喜欢" in item.content else "like"
        return f"semantic:{kind}:{key.split('_', 1)[1]}"
    return f"key:{key}"


def _proposal_label(proposal, *, semantic: bool) -> str:
    memory = proposal.memory
    key = memory.memory_key
    if not semantic:
        return f"key:{key}"
    if key.startswith("goal_"):
        value = memory.content.split("：", 1)[-1].replace("用户正在准备或学习", "")
        return f"semantic:goal:{value}"
    if key.startswith("preference_"):
        kind = "dislike" if "不喜欢" in memory.content else "like"
        value = memory.content.split("不喜欢", 1)[-1] if kind == "dislike" else memory.content.split("喜欢", 1)[-1]
        return f"semantic:{kind}:{value}"
    return f"key:{key}"


def _prf(predictions: list[set[str]] | None, golds: list[set[str]]) -> dict[str, float | None]:
    if predictions is None:
        return {"precision": None, "recall": None, "f1": None}
    true_positive = sum(len(pred & gold) for pred, gold in zip(predictions, golds))
    predicted = sum(len(pred) for pred in predictions)
    expected = sum(len(gold) for gold in golds)
    precision = true_positive / predicted if predicted else (1.0 if expected == 0 else 0.0)
    recall = true_positive / expected if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


async def _context_predictions(
    cases: list[dict[str, Any]], config: MemoryLlmConfig | None
) -> tuple[list[set[str]] | None, dict[str, str]]:
    if config is None or not config.enabled:
        return None, {}
    completion = OpenAICompatibleMemoryCompletion(config)
    predictions: list[set[str]] = []
    raw_outputs: dict[str, str] = {}
    try:
        for case in cases:
            history = tuple(case.get("history", []))[-4:]
            existing = tuple(case.get("existing_memories", []))[:MAX_EXISTING_MEMORIES]
            rules = tuple(extract_memories(case["message"]))
            response = await completion.complete(
                build_memory_llm_messages(
                    case["message"],
                    rules,
                    history,
                    existing,
                    config.max_input_chars,
                    config.confidence_threshold,
                )
            )
            raw_outputs[case["id"]] = response
            proposals = parse_llm_proposals(
                response,
                source_message=case["message"],
                history=history,
                existing_memories=existing,
                confidence_threshold=config.confidence_threshold,
            )
            semantic = "gold_keys" not in case
            predictions.append({_proposal_label(item, semantic=semantic) for item in proposals})
    finally:
        await completion.close()
    return predictions, raw_outputs


def _extraction_failures(
    cases: list[dict[str, Any]],
    predictions: list[set[str]] | None,
    golds: list[set[str]],
    raw_outputs: dict[str, str],
) -> list[dict[str, Any]]:
    if predictions is None:
        return []
    failures = []
    for case, prediction, gold in zip(cases, predictions, golds):
        if prediction == gold:
            continue
        failures.append(
            {
                "id": case["id"],
                "category": case["category"],
                "message": case["message"],
                "gold": sorted(gold),
                "predicted": sorted(prediction),
                "false_positive": sorted(prediction - gold),
                "false_negative": sorted(gold - prediction),
                "raw_llm_output": raw_outputs.get(case["id"], ""),
            }
        )
    return failures


async def _retrieval_metrics(
    cases: list[dict[str, Any]], ablation: Ablation, provider, min_hybrid_score: float
) -> dict[str, Any]:
    relevant_count = 0
    hit1 = hit5 = 0
    reciprocal_ranks = []
    wrong_injected = total_injected = 0
    latencies_ms = []
    failures = []
    scope = UserScope("evaluation", "cahm", "synthetic-user", "synthetic-user", "private")
    for case in cases:
        service = CharacterMemoryService(
            _CaseRepository(case["records"]),
            embedding_provider=provider if ablation.semantic_retrieval else None,
            semantic_enabled=ablation.semantic_retrieval,
            gate_enabled=ablation.confidence_gate,
            min_hybrid_score=min_hybrid_score,
        )
        start = time.perf_counter()
        selected, _ = await service.load_relevant_memories("cahm-eval", scope, case["query"])
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        selected_ids = [item.memory_id for item in selected]
        gold = set(case.get("gold_ids", []))
        if gold:
            relevant_count += 1
            hit1 += int(bool(selected_ids[:1] and gold.intersection(selected_ids[:1])))
            hit5 += int(bool(gold.intersection(selected_ids[:5])))
            rank = next((index for index, item_id in enumerate(selected_ids, start=1) if item_id in gold), None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        wrong = [item_id for item_id in selected_ids if item_id not in gold]
        wrong_injected += len(wrong)
        total_injected += len(selected_ids)
        expected_empty = not gold
        if (gold and not gold.intersection(selected_ids[:5])) or (expected_empty and selected_ids) or wrong:
            failures.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "query": case["query"],
                    "gold_ids": sorted(gold),
                    "selected_ids": selected_ids,
                    "wrong_ids": wrong,
                }
            )
    return {
        "retrieval_recall_at_1": hit1 / relevant_count if relevant_count else 0.0,
        "retrieval_recall_at_5": hit5 / relevant_count if relevant_count else 0.0,
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "wrong_memory_injection_rate": wrong_injected / total_injected if total_injected else 0.0,
        "average_retrieval_latency_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "retrieval_failures": failures,
        "injected_memories": total_injected,
        "wrong_injected_memories": wrong_injected,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        "# CAHM Ablation Results",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Gold Set: {report['dataset']['total']} cases "
        f"({report['dataset']['extraction']} extraction, {report['dataset']['retrieval']} retrieval)",
        f"- Embedding: {report['configuration']['embedding_model']}",
        f"- Context extraction evaluated: {report['configuration']['context_extraction_evaluated']}",
        "",
        "| Group | Ext P | Ext R | Ext F1 | R@1 | R@5 | MRR | Wrong injection | Avg ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        ext = group["extraction"]
        lines.append(
            f"| {group['name']} | {fmt(ext['precision'])} | {fmt(ext['recall'])} | {fmt(ext['f1'])} | "
            f"{group['retrieval_recall_at_1']:.4f} | {group['retrieval_recall_at_5']:.4f} | "
            f"{group['mrr']:.4f} | {group['wrong_memory_injection_rate']:.4f} | "
            f"{group['average_retrieval_latency_ms']:.2f} |"
        )
    lines.extend(["", "## Failure cases", ""])
    for group in report["groups"]:
        extraction_failures = group["extraction_failures"][:10]
        lines.append(f"### {group['name']} extraction ({len(group['extraction_failures'])} total)")
        lines.append("")
        if not extraction_failures:
            lines.append("No extraction failures under the evaluator definition.")
        else:
            for item in extraction_failures:
                lines.append(f"- `{item['id']}` {item['message']} — gold={item['gold']}, predicted={item['predicted']}")
        lines.append("")
        failures = group["retrieval_failures"][:10]
        lines.append(f"### {group['name']} retrieval ({len(group['retrieval_failures'])} total)")
        lines.append("")
        if not failures:
            lines.append("No retrieval failures under the evaluator definition.")
        else:
            for item in failures:
                lines.append(
                    f"- `{item['id']}` {item['query']} — gold={item['gold_ids']}, "
                    f"selected={item['selected_ids']}, wrong={item['wrong_ids']}"
                )
        lines.append("")
    if not report["configuration"]["context_extraction_evaluated"]:
        lines.extend(
            [
                "## Limitation",
                "",
                "No reachable memory LLM was configured for this run. Context-extraction metrics for C/D are N/A; "
                "therefore this run cannot confirm H1. Retrieval metrics are real runs over the local embedding model.",
                "",
            ]
        )
    return "\n".join(lines)


async def evaluate(args) -> dict[str, Any]:
    rows = _load_jsonl(args.dataset)
    extraction_cases = [row for row in rows if row.get("task") == "extraction"]
    retrieval_cases = [row for row in rows if row.get("task") == "retrieval"]
    gold_extraction = [_gold_extraction_labels(case) for case in extraction_cases]
    rule_predictions = [
        {_rule_label(item, semantic="gold_keys" not in case) for item in extract_memories(case["message"])}
        for case in extraction_cases
    ]

    llm_config = None
    if args.memory_llm_base_url and args.memory_llm_model:
        llm_config = MemoryLlmConfig(
            enabled=True,
            base_url=args.memory_llm_base_url,
            model=args.memory_llm_model,
            api_key=args.memory_llm_api_key,
            confidence_threshold=args.memory_llm_confidence_threshold,
        )
    context_predictions, context_raw_outputs = await _context_predictions(extraction_cases, llm_config)

    provider = get_default_embedding_provider()
    provider.embed_texts(["CAHM 语义检索预热"])
    groups = []
    for ablation in ABLATIONS:
        extraction_predictions = context_predictions if ablation.context_extraction else rule_predictions
        retrieval = await _retrieval_metrics(retrieval_cases, ablation, provider, args.min_hybrid_score)
        groups.append(
            {
                **asdict(ablation),
                "extraction": _prf(extraction_predictions, gold_extraction),
                "extraction_failures": _extraction_failures(
                    extraction_cases,
                    extraction_predictions,
                    gold_extraction,
                    context_raw_outputs if ablation.context_extraction else {},
                ),
                **retrieval,
            }
        )

    return {
        "method": "CAHM: Context-Aware Hybrid Memory / 上下文感知混合长期记忆",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "total": len(rows),
            "extraction": len(extraction_cases),
            "retrieval": len(retrieval_cases),
        },
        "configuration": {
            "embedding_model": provider.model_id,
            "context_extraction_evaluated": context_predictions is not None,
            "semantic_candidate_limit": SEMANTIC_MEMORY_CANDIDATE_LIMIT,
            "min_hybrid_memory_score": args.min_hybrid_score,
            "hybrid_weights": {
                "semantic": HYBRID_WEIGHT_SEMANTIC,
                "lexical": HYBRID_WEIGHT_LEXICAL,
                "importance": HYBRID_WEIGHT_IMPORTANCE,
                "recency": HYBRID_WEIGHT_RECENCY,
            },
        },
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-llm-base-url", default="")
    parser.add_argument("--memory-llm-model", default="")
    parser.add_argument("--memory-llm-api-key", default="")
    parser.add_argument("--memory-llm-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--min-hybrid-score", type=float, default=MIN_HYBRID_MEMORY_SCORE)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
