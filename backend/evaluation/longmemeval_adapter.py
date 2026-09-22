"""Read-only LongMemEval adapter with evaluator labels isolated from retrieval.

Opaque IDs matter: official session IDs can begin with ``answer_``. They must
never enter embeddings/prompts. Turn annotations and question metadata likewise
stay in the evaluator. This does not import sessions as approved user facts.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from knowledge.retrieval_core.documents import KnowledgeIndexDocument


@dataclass(frozen=True)
class LongMemEvalCase:
    case_id: str
    category: str
    query: str
    documents: tuple[KnowledgeIndexDocument, ...]
    gold_session_ids: frozenset[str]
    gold_turn_ids: frozenset[str]
    abstention: bool
    repeated_session_occurrences: int = 0


def stable_subset(rows, per_category: int, seed: str = "contextual-evidence-pilot-v1"):
    if per_category < 1:
        raise ValueError("per_category must be positive")
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        key = row["question_id"]
        if key in seen:
            raise ValueError("duplicate benchmark question ID")
        seen.add(key)
        category = "abstention" if key.endswith("_abs") else row["question_type"]
        groups[category].append(row)
    return [
        row
        for category in sorted(groups)
        for row in sorted(
            groups[category], key=lambda item: hashlib.sha256(f"{seed}:{item['question_id']}".encode()).digest()
        )[:per_category]
    ]


def adapt_case(
    row, *, chunk_chars: int = 800, overlap_chars: int = 100, token_counter=None, max_tokens: int = 512
) -> LongMemEvalCase:
    if not 0 <= overlap_chars < chunk_chars or chunk_chars < 100:
        raise ValueError("invalid complete-coverage chunk configuration")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    sessions = row["haystack_sessions"]
    session_ids = row["haystack_session_ids"]
    dates = row["haystack_dates"]
    if not sessions or len(sessions) != len(session_ids) or len(sessions) != len(dates):
        raise ValueError("misaligned or empty benchmark sessions")
    aliases = {}
    contents_by_id = {}
    for index, (key, turns) in enumerate(zip(session_ids, sessions, strict=True)):
        aliases.setdefault(key, f"s{index:04d}")
        if key in contents_by_id and contents_by_id[key] != turns:
            raise ValueError("duplicate benchmark session ID has conflicting content or labels")
        contents_by_id[key] = turns
    abstention = row["question_id"].endswith("_abs")
    expected_sessions = set(row["answer_session_ids"])
    if not expected_sessions <= set(session_ids):
        raise ValueError("gold session absent from history")
    gold_turns = set()
    documents = []
    for occurrence, (original_id, date, turns) in enumerate(zip(session_ids, dates, sessions, strict=True)):
        session_id = aliases[original_id]
        for turn_index, turn in enumerate(turns):
            role, content = turn["role"], turn["content"]
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("unsupported benchmark turn")
            turn_id = f"{session_id}t{turn_index:04d}"
            if turn.get("has_answer") is True and not abstention:
                gold_turns.add(turn_id)
            start = 0
            while start < len(content):
                end = min(len(content), start + chunk_chars)
                prefix = f"Session date: {date}\nSpeaker: {role}\n"
                while token_counter is not None and token_counter(prefix + content[start:end]) > max_tokens:
                    if end - start <= 1:
                        raise ValueError("even a single character and attribution exceed token budget")
                    end = start + max(1, (end - start) // 2)
                chunk = content[start:end]
                text = prefix + chunk
                documents.append(
                    KnowledgeIndexDocument(
                        id=f"{turn_id}o{occurrence:04d}c{start:06d}",
                        domain_id="longmemeval_offline",
                        document_type="conversation_excerpt",
                        title="Conversation excerpt",
                        summary="",
                        content=text,
                        embedding_text=text,
                        metadata={
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "occurrence": occurrence,
                            "start": start,
                            "end": start + len(chunk),
                        },
                        review_status="benchmark_only",
                    )
                )
                if end >= len(content):
                    break
                start = end - min(overlap_chars, end - start - 1)
    return LongMemEvalCase(
        row["question_id"],
        "abstention" if abstention else row["question_type"],
        f"Current date: {row['question_date']}\n{row['question']}",
        tuple(documents),
        frozenset(aliases[key] for key in expected_sessions) if not abstention else frozenset(),
        frozenset(gold_turns),
        abstention,
        len(session_ids) - len(aliases),
    )


def recall_at_k(case: LongMemEvalCase, ranked_indices, k: int) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    session_ids = list(dict.fromkeys(case.documents[index].metadata["session_id"] for index in ranked_indices))[:k]
    turn_ids = list(dict.fromkeys(case.documents[index].metadata["turn_id"] for index in ranked_indices))[:k]
    # The official retrieval setup excludes abstention questions. Returning
    # candidates alone does not measure downstream false-answer generation.
    return {
        "session_recall": len(set(session_ids) & case.gold_session_ids) / len(case.gold_session_ids)
        if case.gold_session_ids
        else None,
        "turn_recall": len(set(turn_ids) & case.gold_turn_ids) / len(case.gold_turn_ids)
        if case.gold_turn_ids
        else None,
        "all_sessions_recalled": set(session_ids) >= case.gold_session_ids if case.gold_session_ids else None,
        "retrieved_sessions": session_ids,
        "retrieved_turns": turn_ids,
    }
