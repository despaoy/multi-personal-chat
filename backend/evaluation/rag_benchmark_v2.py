"""Separate structured-citation RAG benchmark for character conversations."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.character_benchmark_v3 import _call
from evaluation.experiment_contracts import environment_snapshot, sha256_text_file, validate_frozen_gold
from evaluation.review_binding import bound_sample_review, structured_fact_score
from evaluation.retrieval_metrics import RetrievalMetrics
from inference.generation_request import (
    GenerationRequest,
    RetrievalResult,
    build_generation_request,
)
from inference.lora_registry import get_lora_system_prompt
from knowledge.rag_helper import get_rag_helper

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _result_id(result: dict[str, Any]) -> str:
    return str(
        result.get("id")
        or result.get("document_id")
        or result.get("doc_id")
        or result.get("chunk_id")
        or ""
    )


def _normalized(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", value.casefold())


def _answer_abstained(answer: str) -> bool:
    normalized = _normalized(answer)
    return not normalized or any(
        marker in normalized
        for marker in (
            "证据不足",
            "无法确认",
            "无法得知",
            "没有记载",
            "未提供",
            "不能根据",
            "无法回答",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured RAG benchmark")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--review-scores", type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    review_document = (
        json.loads(args.review_scores.read_text(encoding="utf-8"))
        if args.review_scores
        else None
    )
    evaluation_id = f"{args.model}:rag"
    if args.formal:
        errors = validate_frozen_gold(dataset, require_final_held_out=True)
        if errors:
            print(json.dumps({"formal_evaluation_refused": True, "errors": errors}, ensure_ascii=False))
            return 2
    prompts = [
        item for item in dataset.get("prompts", [])
        if item.get("benchmark_suite") == "rag"
    ]
    if not prompts:
        print("RAG benchmark dataset is empty")
        return 2
    persona_prompt = get_lora_system_prompt(str(dataset.get("persona_key", "kisaki")))
    helper = None if args.mock else get_rag_helper()
    metric = RetrievalMetrics()
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(prompts, 1):
        started = time.perf_counter()
        if args.mock:
            expected = list(item.get("expected_refs", []))
            results = [
                {"id": ref, "title": ref, "content": item.get("gold_answer", ""), "score": 1.0}
                for ref in expected
            ]
            citations = [
                {
                    "source_title": result["title"],
                    "evidence_excerpt": result["content"],
                    "score": 1.0,
                    "kb_revision": "mock",
                    "source_path": "mock",
                    "source_line": None,
                    "source_event_ids": [],
                    "source_lineage": [],
                    "section": "rag_grounded",
                    "version": "1.0",
                }
                for result in results
            ]
            abstained = item.get("expected_action") == "abstain"
            confidence = 0.0 if abstained else 1.0
        else:
            retrieved = helper.retrieve_with_citations(item["prompt"], top_k=args.top_k)
            results = retrieved["results"]
            citations = retrieved["citations"]
            confidence = retrieved["confidence"]
            abstained = retrieved["abstained"]
        retrieval_ms = (time.perf_counter() - started) * 1000
        retrieved_ids = [_result_id(result) for result in results]
        expected_ids = [str(value) for value in item.get("expected_refs", [])]
        distractor_ids = [str(value) for value in item.get("distractor_refs", [])]
        evidence = "\n".join(str(result.get("content", "")) for result in results)
        retrieval_error = ""
        if not abstained and not results:
            retrieval_error = "retriever returned no documents without abstaining"
        retrieval = RetrievalResult(
            status="error" if retrieval_error else ("abstained" if abstained else "ok"),
            evidence=evidence,
            documents=tuple(results),
            citations=tuple(citations),
            confidence=confidence,
            reason=retrieval_error or ("retriever abstained" if abstained else ""),
        )
        plan = build_generation_request(
            GenerationRequest(
                message=item["prompt"],
                persona_prompt=persona_prompt,
                interlocutor=str(item.get("interlocutor", "普通用户")),
                retrieval=retrieval,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            )
        )
        if args.mock:
            answer = "" if abstained else item.get("gold_answer", "")
            generation_ms, error = 10.0, ""
        elif abstained:
            answer, generation_ms, error = "", 0.0, ""
        elif retrieval_error:
            answer, generation_ms, error = "", 0.0, retrieval_error
        else:
            answer, generation_ms, error = _call(
                args.base_url,
                args.model,
                [dict(message) for message in plan.messages],
                dict(plan.generation),
                args.timeout,
            )
        expected_action = item.get("expected_action", "answer")
        answer_abstained = _answer_abstained(answer)
        effective_abstention = abstained or answer_abstained
        action_correct = effective_abstention if expected_action == "abstain" else not effective_abstention
        required_answer_facts = [str(value) for value in item.get("required_answer_facts", [])]
        supplied_review, review_binding = bound_sample_review(
            review_document,
            evaluation_id=evaluation_id,
            model=args.model,
            sample_id=item["id"],
            response=answer,
        )
        fact_evaluation = structured_fact_score(required_answer_facts, supplied_review)
        retrieval_evaluable = bool(expected_ids)
        samples.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "expected_refs": expected_ids,
                "retrieved_ids": retrieved_ids,
                "distractor_refs": distractor_ids,
                "retrieved_distractors": sorted(set(retrieved_ids) & set(distractor_ids)),
                "citations": citations,
                "confidence": confidence,
                "abstained": abstained,
                "answer_abstained": answer_abstained,
                "effective_abstention": effective_abstention,
                "retrieval_evaluable": retrieval_evaluable,
                "recall_at_k": metric.recall_at_k(retrieved_ids, expected_ids, args.top_k) if retrieval_evaluable else None,
                "mrr": metric.mrr(retrieved_ids, expected_ids) if retrieval_evaluable else None,
                "ndcg_at_k": metric.ndcg(retrieved_ids, expected_ids, args.top_k) if retrieval_evaluable else None,
                "citation_hit": bool(set(retrieved_ids) & set(expected_ids)) if retrieval_evaluable else None,
                "answer": answer,
                "gold_answer": item.get("gold_answer", ""),
                "required_answer_facts": required_answer_facts,
                "review_binding": review_binding,
                "fact_evaluation": fact_evaluation,
                "expected_action": expected_action,
                "action_correct": action_correct,
                "lexical_diagnostics": {
                    "evidence_overlap": metric.faithfulness(answer, citations) if answer else 0.0,
                    "gold_answer_overlap": metric.answer_correctness(answer, item.get("gold_answer", "")) if answer else 0.0,
                },
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "generation_latency_ms": round(generation_ms, 2),
                "error": error,
            }
        )
        print(f"[{index}/{len(prompts)}] {item['id']}")

    def avg(key: str) -> float:
        values = [float(item[key]) for item in samples if item.get(key) is not None]
        return round(statistics.mean(values), 4) if values else 0.0

    report = {
        "schema_version": 3,
        "evaluation_id": evaluation_id,
        "evaluation_status": "formal" if args.formal else "diagnostic",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mock": args.mock,
        "suite": "rag",
        "model": args.model,
        "provenance": {
            **environment_snapshot(PROJECT_ROOT),
            "dataset_sha256": sha256_text_file(args.dataset),
            "dataset_id": dataset.get("gold_id"),
            "dataset_status": dataset.get("status"),
            "dataset_role": dataset.get("evaluation_role"),
            "prompt_policy_version": plan.prompt_policy_version if samples else None,
            "generation": dict(plan.generation) if samples else None,
            "top_k": args.top_k,
            "citation_contract": [
                "source_title",
                "evidence_excerpt",
                "score",
                "kb_revision",
                "source_path",
                "source_line",
                "source_event_ids",
                "source_lineage",
                "section",
                "version",
            ],
        },
        "metrics": {
            "total": len(samples),
            "recall_at_k": avg("recall_at_k"),
            "mrr": avg("mrr"),
            "ndcg_at_k": avg("ndcg_at_k"),
            "citation_hit_rate": round(
                sum(bool(item["citation_hit"]) for item in samples if item["retrieval_evaluable"])
                / max(sum(item["retrieval_evaluable"] for item in samples), 1),
                4,
            ),
            "abstention_rate": round(sum(item["abstained"] for item in samples) / max(len(samples), 1), 4),
            "expected_action_accuracy": round(
                sum(item["action_correct"] for item in samples) / max(len(samples), 1), 4
            ),
            "structured_fact_scored_rate": round(
                sum(item["fact_evaluation"]["status"] in {"scored", "not_applicable"} for item in samples)
                / max(len(samples), 1),
                4,
            ),
            "distractor_retrieval_rate": round(
                sum(bool(item["retrieved_distractors"]) for item in samples if item["distractor_refs"])
                / max(sum(bool(item["distractor_refs"]) for item in samples), 1),
                4,
            ),
            "average_retrieval_latency_ms": avg("retrieval_latency_ms"),
            "average_generation_latency_ms": avg("generation_latency_ms"),
        },
        "diagnostics": {
            "average_evidence_overlap": round(
                statistics.mean(item["lexical_diagnostics"]["evidence_overlap"] for item in samples),
                4,
            ),
            "average_gold_answer_overlap": round(
                statistics.mean(item["lexical_diagnostics"]["gold_answer_overlap"] for item in samples),
                4,
            ),
            "note": "Lexical overlap is diagnostic and is not a factual-correctness score.",
        },
        "formal_review": {
            "status": "complete"
            if all(item["fact_evaluation"]["status"] in {"scored", "not_applicable"} for item in samples)
            else "pending",
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0 if not any(item["error"] for item in samples) else 2


if __name__ == "__main__":
    raise SystemExit(main())
