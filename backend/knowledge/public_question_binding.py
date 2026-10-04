"""Source-blind question bindings shared by domain routing and evidence review."""

import asyncio
import hashlib
import json

from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from knowledge.public_object_scope import RESOLVE_INSTRUCTION, parse_object_scopes


class QuestionBindingCapacityError(ValueError):
    """Complete question data cannot fit the resolver profile."""


def question_scope_input(dependencies, query, public_obligations=()):
    if dependencies is None or "".join(dependencies.segments) != query:
        raise ValueError("Question binding changed the original query")
    from knowledge.public_obligations import validate_public_obligations

    indices = dict(dependencies.groups)["public_knowledge"]
    if not indices:
        raise ValueError("No public question dependencies")
    obligations = (
        validate_public_obligations(public_obligations, query, public_indices=indices) if public_obligations else ()
    )
    return dict(
        query=query,
        public_tasks=(
            [dict(id=task["index"], text=task["query"]) for task in obligations]
            if obligations
            else [dict(id=i, text=dependencies.segments[i]) for i in indices]
        ),
    )


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
    scopes = parse_object_scopes(binding["raw"], payload["query"], [task["id"] for task in payload["public_tasks"]])
    if scopes != binding["scopes"]:
        raise ValueError("Question binding changed the actual resolver response")
    return scopes


async def resolve_question_binding(dependencies, query, *, window_tokens, public_obligations=(), reviewer=None):
    payload = question_scope_input(dependencies, query, public_obligations)
    messages = [
        dict(role="system", content=RESOLVE_INSTRUCTION),
        dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    if sum(estimated_tokens(m["content"]) + 4 for m in messages) + 768 + CONTEXT_SAFETY_MARGIN_TOKENS > window_tokens:
        raise QuestionBindingCapacityError("Complete question input exceeds resolver capacity")
    if reviewer is None:
        from knowledge.public_task_evidence import _review

        reviewer = _review
    raw = await asyncio.wait_for(reviewer(messages), timeout=30)
    scopes = parse_object_scopes(raw, query, [task["id"] for task in payload["public_tasks"]])
    binding = dict(input=payload, input_sha256=_digest(payload), raw=raw, scopes=scopes)
    validate_question_binding(binding, payload)
    return binding


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
    if any(name not in retrieval_query for name in names):
        raise ValueError("Object search view is not literal retrieval input")
    return names
