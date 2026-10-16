"""Offline identity-slot ablation; never imported by the application."""

from types import SimpleNamespace

from knowledge.entity_scope import identity_evidence_subject
from knowledge.multiscale_rag.service import rerank_with_title_frames
from knowledge.relation_scope import relation_endpoints


def select_identity_coverage(ranked, top_k: int, subject: str):
    """Experimental type coverage, not a factual or temporal validity verdict.

    Keep the best overall evidence and reserve at most one existing slot for a
    directly owned relation. Do not synthesize ownership from text mentions or
    fetch hidden/out-of-scope cards. Callers must filter visibility beforehand.
    """
    selected = ranked[:top_k]
    if not subject or top_k < 2 or not selected:
        return selected
    eligible = [c for c in ranked if subject in relation_endpoints(c.document)]
    if eligible and not any(subject in relation_endpoints(c.document) for c in selected):
        selected = [*selected[:-1], eligible[0]]
    return selected


def rerank_with_identity_coverage(analysis, candidates, *, top_k, reranker, enabled=False):
    """Replay the old ablation on the full pool without changing production."""
    subject = identity_evidence_subject(analysis) if enabled and top_k >= 2 else ""
    if not subject:
        return rerank_with_title_frames(analysis, candidates, top_k=top_k, reranker=reranker)
    # Keep the original experiment's inference request limit while the title
    # formatter returns its complete pool for the final one-slot selection.
    limit = max(top_k * 4, 20, len(candidates))
    pool = SimpleNamespace(rerank=lambda a, rows, **kw: reranker.rerank(a, rows, top_k=limit))
    ranked = rerank_with_title_frames(analysis, candidates, top_k=len(candidates), reranker=pool)
    return select_identity_coverage(ranked, top_k, subject)
