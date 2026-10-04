"""Bounded, scope-preserving retrieval buckets for independent public tasks.

Candidate retention establishes retrieval provenance, never semantic sufficiency.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

_AUTHORITY = (
    "id",
    "chunk_id",
    "document_id",
    "title",
    "content",
    "knowledge_base_id",
    "category",
    "chunk_index",
    "kb_revision",
    "source_path",
    "version",
)


@dataclass(frozen=True)
class TaskCandidatePlan:
    results: list[dict[str, Any]]
    coverage: tuple[dict[str, Any], ...]


def collect_task_candidates(
    query: str,
    views: tuple[str, ...],
    *,
    top_k: int,
    filters: dict[str, Any] | None,
    retrieve: Callable,
    confidence: Callable,
    stable_key: Callable,
    snapshot: Callable,
    threshold: float = 0.3,
) -> TaskCandidatePlan:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("Task candidate limit must be a positive integer")
    if (
        not isinstance(views, tuple)
        or len(views) > 4
        or any(
            not isinstance(view, str)
            or not view.strip()
            or len(view) > 1024
            or any(span not in query for span in view.split())
            for view in views
        )
        or len(set(views)) != len(views)
        or query in views
    ):
        raise ValueError("Invalid bounded task views")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid candidate confidence threshold")
    original_snapshot = snapshot()
    combined: dict[str, dict[str, Any]] = {}
    coverage = []
    for index, question in enumerate((query, *views)):
        if snapshot() != original_snapshot:
            raise RuntimeError("Index changed between retrieval tasks")
        bucket = retrieve(question, top_k=top_k, filters=deepcopy(filters))
        if snapshot() != original_snapshot:
            raise RuntimeError("Index changed during retrieval task")
        if len(bucket) > top_k:
            raise RuntimeError("Retrieval exceeded the per-task candidate bound")
        strength = confidence(bucket)
        accepted = bool(bucket) and strength >= threshold
        coverage.append(
            {
                "task_index": index,
                "kind": "original_question" if index == 0 else "public_task",
                "candidate_count": len(bucket),
                "retained_candidate_count": len(bucket) if accepted else 0,
                "confidence": strength,
                "status": "candidates_retrieved" if accepted else "low_confidence" if bucket else "no_candidates",
                "semantic_coverage": "unverified" if accepted else "not_established",
            }
        )
        if not accepted:
            continue
        for rank, result in enumerate(bucket, 1):
            key = stable_key(result)
            existing = combined.get(key)
            if existing is None:
                existing = deepcopy(result)
                existing["retrieval_tasks"] = ()
                existing["retrieval_task_ranks"] = ()
                combined[key] = existing
            elif any(existing.get(field) != result.get(field) for field in _AUTHORITY):
                raise RuntimeError("Conflicting source authority between task buckets")
            if index not in existing["retrieval_tasks"]:
                existing["retrieval_tasks"] += (index,)
                existing["retrieval_task_ranks"] += ((index, rank),)
    return TaskCandidatePlan(list(combined.values()), tuple(coverage))
