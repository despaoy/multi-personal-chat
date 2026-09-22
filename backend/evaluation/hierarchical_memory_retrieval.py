"""Gold-blind hierarchical retrieval ablations over neutral source metadata.

No new embeddings, labels, summaries or production memory writes. These ranking
policies are hypotheses, not learned models or an established improvement.
"""

from __future__ import annotations

from collections import deque


def selected_sessions(documents, session_ranking, k):
    if k < 1:
        raise ValueError("session k must be positive")
    return list(dict.fromkeys(documents[index].metadata["session_id"] for index in session_ranking))[:k]


def hierarchical_rankings(documents, session_ranking, dense_ranking, *, session_k=5):
    sessions = selected_sessions(documents, session_ranking, session_k)
    allowed = set(sessions)
    dense_within = [index for index in dense_ranking if documents[index].metadata["session_id"] in allowed]
    queues = {key: deque() for key in sessions}
    seen_turns = set()
    duplicate_chunks = []
    for index in dense_within:
        metadata = documents[index].metadata
        if metadata["turn_id"] in seen_turns:
            duplicate_chunks.append(index)
            continue
        seen_turns.add(metadata["turn_id"])
        queues[metadata["session_id"]].append(index)
    round_robin = []
    while any(queues.values()):
        for key in sessions:
            if queues[key]:
                round_robin.append(queues[key].popleft())
    # Later chunks remain available; first representative per turn is the
    # scoring unit, consistent with the existing turn-location proxy.
    round_robin.extend(duplicate_chunks)
    return {"session_rrf_then_dense": dense_within, "session_rrf_round_robin_dense": round_robin}


def pack_turn_excerpts(documents, ranking, token_lengths, *, budget=2048):
    """Bound equal evidence-token budgets, without slicing individual excerpts.

    One highest-ranked excerpt per turn is eligible. This measures turn location,
    not answer-span coverage; long turns are not silently called complete facts.
    """
    if budget < 1 or len(token_lengths) != len(documents):
        raise ValueError("invalid token budget or aligned token lengths")
    if any(type(value) is not int or value < 1 for value in token_lengths):
        raise ValueError("token lengths must be positive integers")
    seen = set()
    chosen = []
    used = 0
    for index in ranking:
        turn = documents[index].metadata["turn_id"]
        if turn in seen:
            continue
        seen.add(turn)
        cost = token_lengths[index]
        if used + cost > budget:
            continue
        chosen.append(index)
        used += cost
    return chosen, used
