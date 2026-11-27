"""Second-stage ranking and bounded context for verified character sources."""

import asyncio
import logging
from hashlib import sha256
from threading import BoundedSemaphore

from knowledge.retrieval_core.tokenization import segment

SOURCE_CANDIDATE_LIMIT = 24
SOURCE_EVIDENCE_LIMIT = 8
_CONTEXT_LINES = 2
_CONTEXT_CHARS = 1800
logger = logging.getLogger(__name__)
_model_slots = BoundedSemaphore(2)


def rerank_source_model(query, documents):
    if not _model_slots.acquire(blocking=False):
        return None
    try:
        from knowledge.reranker import get_reranker

        return get_reranker().rerank(query, documents, top_k=len(documents))
    finally:
        # Keep the slot until native inference finishes, even after a request stops waiting.
        _model_slots.release()


async def rank_source_candidates(query, documents, *, rerank=None, timeout=None):
    async def rank(method):
        return await asyncio.to_thread(rank_verified_sources, query, documents, rerank=method)

    if rerank is None or timeout is None:
        return await rank(rerank)
    try:
        return await asyncio.wait_for(rank(rerank), timeout)
    except TimeoutError:
        logger.warning("Character source rerank exceeded its request time budget")
        return await rank(None)


def rank_verified_sources(query, documents, *, rerank=None):
    if not documents:
        return []
    if rerank is not None:
        try:
            ranked = rerank(query, [dict(document) for document in documents])
            if ranked is not None:
                by_id = {document["id"]: document for document in documents}
                order = [document["id"] for document in ranked]
                if len(order) != len(documents) or set(order) != by_id.keys():
                    raise ValueError("Source reranker returned a different candidate set")
                # Reordering does not alter the authoritative text or lineage.
                return [by_id[identity] for identity in order[:SOURCE_EVIDENCE_LIMIT]]
        except Exception as exc:
            logger.warning("Character source rerank unavailable type=%s", type(exc).__name__)

    terms = {term.casefold() for term in segment(query) if len(term) > 1}

    def coverage(document):
        body = document["content"].casefold()
        title = document["title"].casefold()
        return sum(term in body for term in terms) + sum(term in title for term in terms)

    # Literal matches can refine the hybrid order, but a paraphrased passage
    # may have no overlapping words. Keep both rankings in the evidence window.
    literal = sorted(documents, key=coverage, reverse=True)
    selected = {}
    for literal_hit, hybrid_hit in zip(literal, documents):
        for hit in (literal_hit, hybrid_hit):
            selected.setdefault(hit["id"], hit)
            if len(selected) == SOURCE_EVIDENCE_LIMIT:
                return list(selected.values())
    return list(selected.values())


def expand_source_context(documents, corpus):
    """Expand within the same original paragraph and report the exact citable span."""
    selected = {document["source_id"] for document in documents}
    originals = {source["id"]: source["content"].splitlines() for source in corpus["sources"]
                 if source["id"] in selected}
    expanded = {}
    for document in documents:
        lines = originals[document["source_id"]]
        start, end = document["line_start"], document["line_end"]
        anchor_start, anchor_end = start, end
        size = len(document["content"])
        for _ in range(_CONTEXT_LINES):
            if start > 1 and lines[start - 2].strip() and size + len(lines[start - 2]) + 1 <= _CONTEXT_CHARS:
                start -= 1
                size += len(lines[start - 1]) + 1
            if end < len(lines) and lines[end].strip() and size + len(lines[end]) + 1 <= _CONTEXT_CHARS:
                size += len(lines[end]) + 1
                end += 1
        if (start, end) != (anchor_start, anchor_end):
            content = "\n".join(lines[start - 1:end])
            document = {
                **document, "id": f'{corpus["id"]}:{document["source_id"]}:{start}:{end}',
                "content": content, "line_start": start, "line_end": end,
                "content_sha256": sha256(content.encode()).hexdigest(),
            }
        expanded.setdefault(document["id"], document)
    return list(expanded.values())
