"""Source-blind question bindings shared by domain routing and evidence review."""

import asyncio
import hashlib
import json

from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from knowledge.public_object_scope import RESOLVE_INSTRUCTION, parse_object_scopes


class QuestionBindingCapacityError(ValueError):
    """Complete question data cannot fit the resolver profile."""


class QuestionBindingReviewError(ValueError):
    """An actual resolver exchange failed; it grants no question binding."""

    def __init__(self, reason, messages, raw):
        super().__init__(f"Question binding review unavailable: {reason}")
        self.reason = reason
        self.diagnostic = dict(stage="object", input=messages, raw=raw if isinstance(raw, str) else None)


def question_scope_input(dependencies, query, public_obligations=(), *, history=(), context_budget=None):
    if dependencies is None or "".join(dependencies.segments) != query:
        raise ValueError("Question binding changed the original query")
    from knowledge.public_obligations import validate_public_obligations

    indices = dict(dependencies.groups)["public_knowledge"]
    if not indices:
        raise ValueError("No public question dependencies")
    obligations = (
        validate_public_obligations(public_obligations, query, public_indices=indices) if public_obligations else ()
    )
    payload = dict(
        query=query,
        public_tasks=(
            [dict(id=task["index"], text=task["query"]) for task in obligations]
            if obligations
            else [dict(id=i, text=dependencies.segments[i]) for i in indices]
        ),
    )
    if history:
        from knowledge.question_reference_context import build_reference_context

        context = build_reference_context(history, context_budget=context_budget)
        if context is not None:
            payload["reference_context"] = context
    return payload


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_question_binding(binding, payload):
    if (
        not isinstance(binding, dict)
        or set(binding) != {"input", "input_sha256", "raw", "scopes"}
        or binding["input"] != payload
        or binding["input_sha256"] != _digest(payload)
        or not isinstance(binding["raw"], str)
    ):
        raise ValueError("Question binding is not bound to the complete original input")
    from knowledge.question_reference_context import binding_payload_for_tasks

    if payload != binding_payload_for_tasks(binding, payload["query"], payload["public_tasks"]):
        raise ValueError("Question reference input changed")
    scopes = parse_object_scopes(binding["raw"], payload["query"], [task["id"] for task in payload["public_tasks"]],
                                 reference_context=payload.get("reference_context"))
    if scopes != binding["scopes"]:
        raise ValueError("Question binding changed the actual resolver response")
    return scopes


async def resolve_question_binding(dependencies, query, *, window_tokens, public_obligations=(), history=(), context_budget=None, reviewer=None):
    try:
        payload = question_scope_input(dependencies, query, public_obligations, history=history, context_budget=context_budget)
    except ValueError as exc:
        raise QuestionBindingCapacityError("Complete question references cannot fit the resolver profile") from exc
    messages = [
        dict(role="system", content=RESOLVE_INSTRUCTION),
        dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    if sum(estimated_tokens(m["content"]) + 4 for m in messages) + 768 + CONTEXT_SAFETY_MARGIN_TOKENS > window_tokens:
        raise QuestionBindingCapacityError("Complete question input exceeds resolver capacity")
    if reviewer is None:
        from knowledge.public_task_evidence import _review

        reviewer = _review
    raw = None
    try:
        raw = await asyncio.wait_for(reviewer(messages), timeout=30)
        scopes = parse_object_scopes(raw, query, [task["id"] for task in payload["public_tasks"]],
                                     reference_context=payload.get("reference_context"))
        binding = dict(input=payload, input_sha256=_digest(payload), raw=raw, scopes=scopes)
        validate_question_binding(binding, payload)
        return binding
    except asyncio.TimeoutError as exc:
        raise QuestionBindingReviewError("timeout", messages, raw) from exc
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise QuestionBindingReviewError("invalid_or_incomplete_review", messages, raw) from exc
    except Exception as exc:
        raise QuestionBindingReviewError("provider_error", messages, raw) from exc


def failed_question_binding_review(error, dependencies, query, public_obligations=(), *, history=(), context_budget=None):
    """Carry unavailable obligations and diagnostics, never resolver assertions."""
    payload = question_scope_input(dependencies, query, public_obligations, history=history, context_budget=context_budget)
    expected = [
        dict(role="system", content=RESOLVE_INSTRUCTION),
        dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    if (
        not isinstance(error, QuestionBindingReviewError)
        or not isinstance(error.diagnostic, dict)
        or set(error.diagnostic) != {"stage", "input", "raw"}
        or error.diagnostic["stage"] != "object"
        or error.diagnostic["input"] != expected
        or (error.diagnostic["raw"] is not None and not isinstance(error.diagnostic["raw"], str))
    ):
        raise ValueError("Failed question binding is not bound to the complete original input")
    return dict(
        query=query,
        tasks=list(public_obligations) if public_obligations else [
            dict(index=task["id"], query=task["text"]) for task in payload["public_tasks"]
        ],
        task_granularity="literal_partition" if public_obligations else "segment_unverified",
        receipt_schema_version=2,
        public_segment_indices=list(dict(dependencies.groups)["public_knowledge"]),
        decisions=[], review_status="unavailable", reason=f"question_binding_{error.reason}",
        failure_diagnostic=error.diagnostic,
    )


def all_bound_objects_outside_character_domain(binding, config):
    """Registered aliases own whole names, never substrings of other objects.

    Empty/unresolved scopes and explicit curated targets keep the strict domain
    path. Mixed genuine curated/generic routing is not certified here.
    """
    scopes = validate_question_binding(binding, binding["input"])
    names = [obj["query_text"] for obj in scopes["objects"]]
    if not names or any(not row["object_ids"] for row in scopes["task_scopes"]):
        return False
    return all(config.canonical_entity(name) is None and name not in config.story_titles for name in names)


def bound_object_search_views(binding, retrieval_query):
    """Supplement a full original-query search when planner hints are absent."""
    scopes = validate_question_binding(binding, binding["input"])
    names = tuple(obj["query_text"] for obj in scopes["objects"])
    if len(names) > 4:
        raise QuestionBindingCapacityError("Complete object view set exceeds retrieval capacity")
    if binding["input"]["query"] not in retrieval_query:
        raise ValueError("Object search view is not literal retrieval input: original question discarded")
    if "reference_context" not in binding["input"] and any(name not in retrieval_query for name in names):
        raise ValueError("Object search view is not literal retrieval input")
    return names
