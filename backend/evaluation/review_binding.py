"""Bind human judgments to the exact evaluation response they reviewed."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def bound_sample_review(
    review_document: Mapping[str, Any] | None,
    *,
    evaluation_id: str,
    model: str,
    sample_id: str,
    response: str,
) -> tuple[dict[str, Any] | None, str]:
    """Return a review only when its run, model, sample, and text all match."""

    if not review_document:
        return None, "missing_review"
    if review_document.get("evaluation_id") != evaluation_id:
        return None, "evaluation_id_mismatch"
    if review_document.get("model") != model:
        return None, "model_mismatch"
    samples = review_document.get("samples")
    if not isinstance(samples, Mapping):
        return None, "missing_samples"
    supplied = samples.get(sample_id)
    if not isinstance(supplied, Mapping):
        return None, "missing_sample_review"
    if supplied.get("sample_id", sample_id) != sample_id:
        return None, "sample_id_mismatch"
    if supplied.get("response") != response:
        return None, "response_mismatch"
    return dict(supplied), "matched"


def structured_fact_score(
    required_facts: list[str], supplied: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Score explicit human fact judgments; no lexical proxy is used."""

    if not required_facts:
        return {"status": "not_applicable", "score": None, "facts": []}
    judgments = supplied.get("facts") if supplied else None
    if not isinstance(judgments, Mapping):
        judgments = {}
    rows = []
    complete = True
    for fact in required_facts:
        raw_score = judgments.get(fact)
        if isinstance(raw_score, bool):
            score = 1.0 if raw_score else 0.0
        elif isinstance(raw_score, (int, float)) and 0 <= raw_score <= 1:
            score = float(raw_score)
        else:
            score = None
            complete = False
        rows.append({"fact": fact, "score": score})
    return {
        "status": "scored" if complete else "pending_human_review",
        "score": round(sum(row["score"] for row in rows) / len(rows), 4)
        if complete
        else None,
        "facts": rows,
    }
