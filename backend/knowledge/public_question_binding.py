"""Source-blind question bindings shared by domain routing and evidence review."""

import asyncio
import hashlib
import json

from inference.context_budget import InputBudgetError
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from knowledge.public_object_scope import parse_object_scopes, resolver_instruction


class QuestionBindingCapacityError(ValueError):
    """Complete question data cannot fit the resolver profile."""

    def public_detail(self):
        return {
            "stage": "question_binding", "reason": "capacity_exceeded",
            "message": "公共任务或完整输入超出处理预算，本轮未生成回复。请缩小问题范围或拆分后重试。",
        }


class QuestionBindingReviewError(ValueError):
    """An actual resolver exchange failed; it grants no question binding."""

    def __init__(self, reason, messages, raw):
        super().__init__(f"Question binding review unavailable: {reason}")
        self.reason = reason
        self.diagnostic = dict(stage="object", input=messages, raw=raw if isinstance(raw, str) else None)


def question_scope_input(dependencies, query, public_obligations=(), *, history=(), context_budget=None, with_retrieval_scope=False):
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
    if with_retrieval_scope:
        groups = dict(dependencies.groups)
        payload["scope_policy"] = dict(
            allow_independent_identity=len(payload["public_tasks"]) == 1 and not any(groups[kind] for kind in ("private_memory", "current_input", "unresolved_source")),
            include_unresolved=bool(groups["unresolved_source"]),
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
                                 reference_context=payload.get("reference_context"), scope_policy=payload.get("scope_policy"))
    if scopes != binding["scopes"]:
        raise ValueError("Question binding changed the actual resolver response")
    return scopes


async def resolve_question_binding(dependencies, query, *, window_tokens, public_obligations=(), history=(), context_budget=None, reviewer=None, with_retrieval_scope=False):
    try:
        payload = question_scope_input(dependencies, query, public_obligations, history=history, context_budget=context_budget, with_retrieval_scope=with_retrieval_scope)
    except InputBudgetError as exc:
        raise QuestionBindingCapacityError("Complete question references cannot fit the resolver profile") from exc
    messages = [
        dict(role="system", content=resolver_instruction(payload.get("scope_policy"))),
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
        if not isinstance(raw, str):
            raise ValueError("Question resolver response must be text")
        scopes = parse_object_scopes(raw, query, [task["id"] for task in payload["public_tasks"]],
                                     reference_context=payload.get("reference_context"), scope_policy=payload.get("scope_policy"))
        return dict(input=payload, input_sha256=_digest(payload), raw=raw, scopes=scopes)
    except asyncio.TimeoutError as exc:
        raise QuestionBindingReviewError("timeout", messages, raw) from exc
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise QuestionBindingReviewError("invalid_or_incomplete_review", messages, raw) from exc
    except Exception as exc:
        raise QuestionBindingReviewError("provider_error", messages, raw) from exc


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


def bound_retrieval_query(binding, query):
    """Keep the original question beside object names from its validated proof."""
    if binding is None:
        return query
    anchors = tuple(name for name in bound_object_search_views(binding, query) if name not in query)
    return "\n".join((*anchors, query))


def bound_task_search_views(binding, retrieval_query):
    """Preserve each complete public task beside its proven object anchors."""
    bound_object_search_views(binding, retrieval_query)
    scopes = binding["scopes"]
    objects = {row["object_id"]: row["query_text"] for row in scopes["objects"]}
    texts = {task["id"]: task["text"] for task in binding["input"]["public_tasks"]}
    views = []
    for task in scopes["task_scopes"]:
        text = texts[task["task_id"]]
        for identity in task["object_ids"] or (None,):
            view = objects[identity] + "\n" + text if identity is not None else text
            if view != retrieval_query and view not in views:
                views.append(view)
    if len(views) > 4:
        raise QuestionBindingCapacityError("Complete public task views exceed retrieval capacity")
    return tuple(views)
