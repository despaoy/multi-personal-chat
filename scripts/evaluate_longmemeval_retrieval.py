"""Read-only public-benchmark retrieval pilot; no QA score or memory writes.

Rank conversation chunks with BM25 and optional local MiniLM/RRF. The subset is
fixed by hash before retrieval. Gold labels are consulted only for scoring.
This is not a faithful reproduction of the original paper's retrieval setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.hierarchical_memory_retrieval import hierarchical_rankings, pack_turn_excerpts  # noqa: E402
from evaluation.longmemeval_adapter import adapt_case, recall_at_k, stable_subset  # noqa: E402
from knowledge.retrieval_core.index import BM25Index  # noqa: E402


def summarize(rows):
    result = {"cases": len(rows), "abstention_cases_not_scored": sum(row["abstention"] for row in rows)}
    for key in ("session_recall", "turn_recall", "all_sessions_recalled"):
        scores = [row[key] for row in rows if row[key] is not None]
        result[key] = statistics.mean(scores) if scores else None
        result[f"{key}_denominator"] = len(scores)
    for key in ("budgeted_turn_recall", "budgeted_all_turns_recalled", "evidence_tokens", "budgeted_turn_count"):
        scores = [row[key] for row in rows if row.get(key) is not None]
        if scores:
            result[key] = statistics.mean(scores)
            result[f"{key}_denominator"] = len(scores)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--hierarchical", action="store_true", help="Offline two-stage session/turn ablations")
    parser.add_argument("--context-token-budget", type=int, default=2048)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".progress.jsonl").exists():
        parser.error("output or progress file exists")
    if args.k < 1:
        parser.error("k must be positive")
    if args.hierarchical and not args.dense:
        parser.error("hierarchical comparison requires --dense")
    if args.context_token_budget < 1:
        parser.error("context token budget must be positive")
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/evaluate_longmemeval_retrieval.py",
            "backend/evaluation/longmemeval_adapter.py",
            "backend/evaluation/hierarchical_memory_retrieval.py",
            "backend/knowledge/retrieval_core/index.py",
            "backend/knowledge/multiscale_rag/vector_runtime.py",
        )
    }
    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw)
    selected = stable_subset(rows, args.per_category)
    del raw, rows
    provider = None
    if args.dense:
        from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider

        provider = LocalMeanPoolingEmbeddingProvider(max_length=512, batch_size=16)
        tokenizer, _model = provider._load()
    results = {"bm25": []}
    if provider:
        results.update(minilm=[], reciprocal_rank_fusion=[])
    if args.hierarchical:
        results.update(session_rrf_then_dense=[], session_rrf_round_robin_dense=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".progress.jsonl").open("x", encoding="utf-8") as progress:
        for raw_case in selected:
            case = adapt_case(
                raw_case,
                token_counter=(lambda text: len(tokenizer(text, truncation=False)["input_ids"])) if provider else None,
            )
            # Only query + documents cross the retrieval boundary. No answer,
            # category, original answer-prefixed session ID or has_answer flag.
            started = time.perf_counter()
            index = BM25Index()
            index.build(case.documents)
            lexical = [index for index, _score in index.search(case.query, len(case.documents))]
            rankings = {"bm25": lexical}
            if provider:
                texts = [case.query, *[doc.embedding_text for doc in case.documents]]
                lengths = [len(ids) for ids in tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]]
                if max(lengths) > 512:
                    raise ValueError("complete embedding input exceeds 512 tokens; no silent truncation")
                vectors = provider.embed_texts(texts)
                similarities = vectors[1:] @ vectors[0]
                dense = np.argsort(-similarities, kind="stable").tolist()
                fused = {}
                for ranking in (lexical, dense):
                    for rank, position in enumerate(ranking, start=1):
                        fused[position] = fused.get(position, 0.0) + 1 / (60 + rank)
                rankings.update(minilm=dense, reciprocal_rank_fusion=sorted(fused, key=lambda key: (-fused[key], key)))
                if args.hierarchical:
                    rankings.update(
                        hierarchical_rankings(
                            case.documents, rankings["reciprocal_rank_fusion"], dense, session_k=args.k
                        )
                    )
            for variant, ranking in rankings.items():
                row = {
                    "id": case.case_id,
                    "category": case.category,
                    "abstention": case.abstention,
                    "chunks": len(case.documents),
                    "repeated_session_occurrences": case.repeated_session_occurrences,
                    **recall_at_k(case, ranking, args.k),
                }
                if provider:
                    excerpts, used_tokens = pack_turn_excerpts(
                        case.documents, ranking, lengths[1:], budget=args.context_token_budget
                    )
                    turns = {case.documents[index].metadata["turn_id"] for index in excerpts}
                    row.update(
                        budgeted_turn_recall=(len(turns & case.gold_turn_ids) / len(case.gold_turn_ids))
                        if case.gold_turn_ids
                        else None,
                        budgeted_all_turns_recalled=(turns >= case.gold_turn_ids) if case.gold_turn_ids else None,
                        evidence_tokens=used_tokens,
                        budgeted_turn_count=len(turns),
                    )
                results[variant].append(row)
                progress.write(json.dumps({"variant": variant, **row}) + "\n")
            progress.flush()
            print(f"{case.case_id}: {len(case.documents)} chunks, {time.perf_counter() - started:.1f}s", flush=True)
    report = {
        "status": "public_benchmark_retrieval_pilot_not_official_qa_result",
        "source_sha256": digest,
        "source_file": args.input.name,
        "subset_seed": "contextual-evidence-pilot-v1",
        "per_category": args.per_category,
        "k": args.k,
        "chunk_chars_upper_bound": 800,
        "token_bounded_chunks": provider is not None,
        "overlap_chars": 100,
        "embedding_max_tokens": 512 if provider else None,
        "evaluation_source_sha256": source_hashes,
        "hierarchical_enabled": args.hierarchical,
        "hierarchical_session_k": args.k if args.hierarchical else None,
        "context_token_budget": args.context_token_budget if provider else None,
        "context_tokenizer": "MiniLM tokenizer; evidence-only proxy, not generator tokenizer" if provider else None,
        "embedding_model": {"id": provider.model_id, "path": str(provider.model_path)} if provider else None,
        "subset_ids": [row["question_id"] for row in selected],
        "notes": [
            "Retrieval-only; no generation or false-answer metric. Abstention queries are excluded from recall means.",
            "Turns are chunked without dropping text. Unique sessions and turns are independently deduplicated before top-k.",
            "Turn hits are location proxies; without answer spans a retrieved partial chunk may not itself contain the answer.",
            "Repeated identical session IDs retain every dated occurrence but count once for session/turn recall; conflicting reuse fails validation.",
            "English public histories do not establish Chinese role-chat quality or production memory-extraction accuracy.",
            "Budgeted retrieval packs one ranked excerpt per turn atomically under the same evidence-token cap; answer spans are not annotated, so turn hits do not establish complete answer evidence.",
        ],
        "variants": {key: {"summary": summarize(value), "cases": value} for key, value in results.items()},
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps({key: value["summary"] for key, value in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
