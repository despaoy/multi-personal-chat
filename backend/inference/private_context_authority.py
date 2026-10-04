"""Recheck selected original speech with the trusted prepared owner scope.

No new recall, model review, source substitution or memory mutation happens here.
The callback lives on a generation request, never on a serialized completion.
"""

import json
from dataclasses import replace

from character.models import UserScope
from inference.source_context_budget import _complete_packet


def _records(text):
    if not text:
        return []
    if not _complete_packet(text):
        raise ValueError("Invalid complete private source packet")
    return json.loads(text)["records"]


def _retain_packet(text, rows, granted):
    kept = [row for row in rows if row["source_id"] in granted]
    if len(kept) == len(rows):
        return text
    if not kept:
        return ""
    value = json.loads(text)
    value["records"] = kept
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _without_history_turns(history, bodies):
    kept = []
    excluded_turn = False
    for item in history:
        contains = any(body in item.get("content", "") for body in bodies)
        if item.get("role") == "user":
            excluded_turn = contains
        if not contains and not excluded_turn:
            kept.append(item)
    return history if len(kept) == len(history) else tuple(kept)


async def revalidate_private_sources(request, repository, character_id, scope, *, preferred_address=""):
    context = request.character_context
    if context is None:
        return request
    text = context.episodic_reference_context
    candidate = context.source_candidate_context
    if not text and not candidate:
        return request

    unavailable = False
    try:
        rows = _records(text)
        deferred = _records(candidate)
    except (ValueError, TypeError, KeyError):
        rows, deferred = [], []
        unavailable = True
    originals = {}
    for row in [*rows, *deferred]:
        source_id = row["source_id"]
        if source_id in originals and originals[source_id] != row:
            unavailable = True
        originals[source_id] = row
    ids = tuple(originals)
    granted = set()
    if not unavailable:
        try:
            fresh = []
            for offset in range(0, len(ids), 100):
                fresh.extend(
                    await repository.list_sources(
                        character_id,
                        scope,
                        source_message_ids=ids[offset : offset + 100],
                    )
                )
            seen = set()
            for row in fresh:
                source_id = row["source_message_id"]
                if source_id not in originals or source_id in seen:
                    raise ValueError("Invalid fresh scoped private source identities")
                seen.add(source_id)
                original = originals[source_id]
                if row["body"] == original["text"] and row["observed_at"] == original["observed_at"]:
                    granted.add(source_id)
        except Exception:
            unavailable = True
            granted.clear()

    revoked = set(ids) - granted
    if not unavailable and not revoked:
        return request
    bodies = tuple(originals[source_id]["text"] for source_id in revoked)
    admitted = "" if unavailable else _retain_packet(text, rows, granted)
    deferred_text = "" if unavailable else _retain_packet(candidate, deferred, granted)
    status = (
        "available"
        if admitted
        else context.memory_source_status
        if deferred_text
        else ("authority_unavailable" if unavailable else "authority_changed")
    )
    updated = replace(
        context,
        episodic_reference_context=admitted,
        source_candidate_context=deferred_text,
        memory_source_status=status,
        source_reference_backup="",
        memory_review_text="",
        memory_field_presence=tuple((key, None) for key, _value in context.memory_field_presence),
    )

    # Source erasure can also change a copied claim's evidence. Keep independent
    # claims; a retained projection is valid only if the old packet still equals
    # the currently scoped stored evidence, never by its old source ID alone.
    removed = set()
    for packet in context.memory_packets:
        if not revoked.intersection(packet.source_message_ids):
            continue
        try:
            row = await repository.get_memory_record(int(packet.memory_id), character_id, scope)
            valid = row is not None and row["content"] == packet.content and row["evidence"] == list(packet.evidence)
        except Exception:
            valid = False
        if not valid:
            removed.add(packet.memory_id)
    if removed:
        from character.context_builder import compile_reference_context

        remaining = tuple(packet for packet in context.memory_packets if packet.memory_id not in removed)
        reference, used_ids = compile_reference_context(
            remaining,
            preferred_address=preferred_address,
            complete_evidence=True,
            observation_semantics=any(packet.source_observation for packet in remaining),
            max_chars=None,
        )
        updated = replace(
            updated,
            memory_packets=remaining,
            used_memory_ids=used_ids,
            reference_context=reference,
            memory_status="available" if remaining else "retrieval_error" if unavailable else "no_match",
            source_shared_memory_ids=tuple(key for key in context.source_shared_memory_ids if key not in removed),
        )
    elif any(body in context.reference_context for body in bodies):
        updated = replace(
            updated,
            reference_context="",
            used_memory_ids=(),
            memory_packets=(),
            source_shared_memory_ids=(),
            memory_status="retrieval_error",
        )

    changes = {}
    for field in ("conversation_reference_context", "branch_context"):
        if any(body in getattr(updated, field) for body in bodies):
            changes[field] = ""
    if changes:
        updated = replace(updated, **changes)
    return replace(request, character_context=updated, history=_without_history_turns(request.history, bodies))


def make_private_context_revalidator(prepared, database):
    """Only trusted application preparation supplies a role/owner scope."""
    scope = getattr(prepared, "user_scope", None)
    character_id = getattr(prepared, "character_id", None)
    if not isinstance(scope, UserScope) or not isinstance(character_id, str) or not character_id:
        return None
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    repository = DatabaseCharacterMemoryRepository(database)
    preferred_address = getattr(getattr(prepared, "relationship", None), "preferred_address", "")

    async def revalidate(request):
        from inference.structured_context_authority import revalidate_private_memories

        request = await revalidate_private_memories(
            request, repository, character_id, scope, preferred_address=preferred_address,
        )
        return await revalidate_private_sources(
            request,
            repository,
            character_id,
            scope,
            preferred_address=preferred_address,
        )

    return revalidate
