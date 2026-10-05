"""Whole authorized history is reference data, never facts or new permissions."""

from copy import deepcopy


def validate_reference_context(context):
    if not isinstance(context, dict) or set(context) != {"history", "omitted_messages"}:
        raise ValueError("Invalid question reference context")
    history, omitted = context["history"], context["omitted_messages"]
    if (
        not isinstance(history, list)
        or not history
        or type(omitted) is not int
        or omitted < 0
        or any(
            not isinstance(row, dict)
            or set(row) != {"role", "content"}
            or row["role"] not in {"user", "assistant"}
            or not isinstance(row["content"], str)
            or not row["content"].strip()
            for row in history
        )
    ):
        raise ValueError("Incomplete or untrusted reference message shape")
    return context


def build_reference_context(history, *, context_budget=None):
    from character.evidence_selector import _history_view

    complete = _history_view(
        history,
        **(
            dict(max_messages=context_budget.history_messages, max_chars=4 * context_budget.window_tokens)
            if context_budget
            else {}
        ),
    )
    if not complete:
        return None
    count = sum(
        isinstance(row, dict)
        and row.get("role") in {"user", "assistant"}
        and isinstance(row.get("content"), str)
        and bool(row["content"].strip())
        for row in history
    )
    return validate_reference_context(dict(history=complete, omitted_messages=count - len(complete)))


def object_reference_origin(name, query, context=None):
    """Exact user spans may locate objects; assistant guesses cannot name them."""
    if name in query:
        return None
    if context is not None:
        validate_reference_context(context)
        for index in reversed(range(len(context["history"]))):
            row = context["history"][index]
            if row["role"] == "user" and name in row["content"]:
                return dict(message_index=index, quote=name)
    raise ValueError("Object name lacks an original query or user-history span")


def binding_payload_for_tasks(binding, query, tasks):
    """Keep independently checked current obligations beside frozen references."""
    payload = dict(query=query, public_tasks=tasks)
    original = binding["input"]
    if not isinstance(original, dict) or set(original) not in (
        {"query", "public_tasks"},
        {"query", "public_tasks", "reference_context"},
    ):
        raise ValueError("Question binding has unexpected input fields")
    if "reference_context" in original:
        payload["reference_context"] = deepcopy(validate_reference_context(original["reference_context"]))
    return payload
