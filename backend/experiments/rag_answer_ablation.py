"""R2 answer-layer ablation after retrieval strategy selection."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.experiment_contracts import canonical_json_hash, sha256_text_file
from inference.prompt_policy import (
    PROMPT_POLICY_VERSION,
    build_grounded_user_message,
    compose_system_prompt,
)


@dataclass
class RAGAnswerResult:
    variant: str
    mock: bool = False
    citation_hit_rate: float = 0.0
    evidence_token_coverage: float = 0.0
    unanswerable_abstention_accuracy: float = 0.0
    over_abstention_rate: float = 0.0
    p95_latency_ms: float = 0.0
    failures: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower()))


def _coverage(answer: str, evidence: str) -> float:
    answer_tokens = _tokens(answer)
    return len(answer_tokens & _tokens(evidence)) / len(answer_tokens) if answer_tokens else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


class RAGAnswerAblation:
    VARIANTS = ("direct_answer", "confidence_abstain", "corrective_rag")

    def __init__(
        self,
        dataset_path: Path,
        system_prompt: str,
        *,
        formal: bool = False,
        threshold: float = 0.3,
        retrieval_strategy: str = "hybrid_reranker",
    ):
        self.dataset_path = dataset_path
        self.persona_prompt = system_prompt.strip()
        self.system_prompt = compose_system_prompt(self.persona_prompt, include_rag=True)
        self.formal = formal
        self.threshold = threshold
        self.retrieval_strategy = retrieval_strategy
        self._retrieval_runner = None

    def _dataset(self) -> dict[str, Any]:
        data = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if self.formal and (data.get("status") != "frozen" or not data.get("formal_use_allowed")):
            raise ValueError("formal R2 answer evaluation requires a reviewed and frozen dataset")
        return data

    def _retrieve_rows(self, question: str) -> list[dict[str, Any]]:
        if self._retrieval_runner is None:
            from experiments.rag_ablation import RAGAblation

            self._retrieval_runner = RAGAblation(
                str(self.dataset_path), k=5, formal=self.formal
            )
        retrieve = self._retrieval_runner._variant(self.retrieval_strategy)
        return retrieve(question, 5)

    def _package(self, rows: list[dict[str, Any]], *, abstain: bool) -> dict[str, Any]:
        from knowledge.rag_helper import get_rag_helper

        helper = get_rag_helper()
        confidence = helper.compute_confidence(rows)
        should_abstain = abstain and confidence < self.threshold
        return {
            "results": [] if should_abstain else rows,
            "citations": [] if should_abstain else helper.build_citations(rows),
            "confidence": confidence,
            "abstained": should_abstain,
            "reformulated": False,
        }

    def _retrieve(self, variant: str, question: str) -> dict[str, Any]:
        rows = self._retrieve_rows(question)
        if variant == "direct_answer":
            return self._package(rows, abstain=False)
        if variant == "confidence_abstain":
            return self._package(rows, abstain=True)
        if variant == "corrective_rag":
            first = self._package(rows, abstain=True)
            if not first["abstained"]:
                return first
            from knowledge.corrective_rag import CorrectiveRAG
            from knowledge.rag_helper import get_rag_helper

            reformulator = CorrectiveRAG(
                get_rag_helper(), threshold=self.threshold, max_retries=1
            )
            rewritten = reformulator.reformulate_query(question, rows)
            second_rows = self._retrieve_rows(rewritten)
            second = self._package(second_rows, abstain=True)
            second.update(
                reformulated=True,
                original_query=question,
                reformulated_query=rewritten,
            )
            return second
        raise ValueError(f"unknown R2 answer variant: {variant}")
    def run_variant(self, variant: str, generate: Callable[[list[dict[str, str]]], str], *, mock: bool = False) -> RAGAnswerResult:
        data = self._dataset()
        result = RAGAnswerResult(variant=variant, mock=mock)
        citation_hits: list[float] = []
        coverage: list[float] = []
        unanswerable_ok: list[float] = []
        answerable_abstained: list[float] = []
        latencies: list[float] = []
        for item in data["questions"]:
            started = time.perf_counter()
            error = ""
            evidence = ""
            try:
                retrieved = self._retrieve(variant, item["question"])
                abstained = bool(retrieved.get("abstained"))
                evidence = "\n".join(str(row.get("content", "")) for row in retrieved.get("results", []))
                if abstained:
                    answer = ""
                elif mock:
                    answer = str(item.get("gold_answer", ""))
                else:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": build_grounded_user_message(
                                item["question"], evidence, max_chars=4000
                            ),
                        },
                    ]
                    answer = generate(messages)
            except Exception as exc:
                retrieved = {"results": [], "citations": [], "confidence": 0.0, "abstained": True}
                abstained, answer, error = True, "", f"{type(exc).__name__}: {exc}"
                result.failures += 1
            latency = (time.perf_counter() - started) * 1000
            latencies.append(latency)
            expected = set(map(str, item.get("expected_doc_ids", [])))
            cited = {str(row.get("source_id", "")) for row in retrieved.get("citations", [])}
            answerable = bool(expected)
            if answerable:
                citation_hits.append(float(bool(expected & cited)))
                answerable_abstained.append(float(abstained))
                if answer:
                    coverage.append(_coverage(answer, evidence))
            else:
                unanswerable_ok.append(float(abstained))
            result.samples.append({
                "id": item["id"], "question_type": item["question_type"], "answer": answer,
                "expected_doc_ids": sorted(expected), "cited_doc_ids": sorted(cited),
                "confidence": retrieved.get("confidence"), "abstained": abstained,
                "reformulated": bool(retrieved.get("reformulated")),
                "evidence_token_coverage": round(_coverage(answer, evidence), 4) if answer else 0.0,
                "latency_ms": round(latency, 3), "error": error,
            })
        result.citation_hit_rate = round(statistics.mean(citation_hits), 4) if citation_hits else 0.0
        result.evidence_token_coverage = round(statistics.mean(coverage), 4) if coverage else 0.0
        result.unanswerable_abstention_accuracy = round(statistics.mean(unanswerable_ok), 4) if unanswerable_ok else 0.0
        result.over_abstention_rate = round(statistics.mean(answerable_abstained), 4) if answerable_abstained else 0.0
        result.p95_latency_ms = round(_percentile(latencies, 0.95), 3)
        return result


def vllm_generator(base_url: str, model: str) -> Callable[[list[dict[str, str]]], str]:
    def generate(messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": 256}).encode()
        request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]
    return generate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--retrieval-strategy", choices=("vector_only", "bm25_only", "hybrid", "hybrid_reranker"), default="hybrid_reranker")
    args = parser.parse_args()
    runner = RAGAnswerAblation(
        args.dataset, args.system_prompt_file.read_text(encoding="utf-8"),
        formal=args.formal, retrieval_strategy=args.retrieval_strategy,
    )
    generate = vllm_generator(args.base_url, args.model)
    results = [runner.run_variant(name, generate, mock=args.mock) for name in runner.VARIANTS]
    payload = {
        "schema_version": 1, "experiment_id": "R2-ANSWER", "mock": args.mock,
        "formal": args.formal and not args.mock, "retrieval_strategy": args.retrieval_strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "persona_prompt_path": str(args.system_prompt_file),
            "persona_prompt_sha256": sha256_text_file(args.system_prompt_file),
            "effective_system_prompt_sha256": canonical_json_hash(runner.system_prompt),
            "prompt_policy_version": PROMPT_POLICY_VERSION,
        },
        "automatic_faithfulness_is_diagnostic": True, "results": [asdict(row) for row in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(not row.failures for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
