"""Recheck selected original speech with the trusted prepared owner scope.

No new recall, model review, source substitution or memory mutation happens here.
The callback lives on a generation request, never on a serialized completion.
"""

import json
from dataclasses import replace

from character.models import UserScope
from inference.source_context_budget import _parse_complete_packet
from inference.history import without_history_turns
from inference.structured_context_authority import revalidate_private_memories, without_memories


def _packet(text):
    if not text:
        return None
    packet = _parse_complete_packet(text)
    if packet is None:
        raise ValueError("Invalid complete private source packet")
    return packet


def _retain_packet(text, packet, granted):
    if packet is None:
        return text
    rows = packet["records"]
    kept = [row for row in rows if row["source_id"] in granted]
    if len(kept) == len(rows):
        return text
    if not kept:
        return ""
    return json.dumps(dict(packet, records=kept), ensure_ascii=False, separators=(",", ":"))



async def revalidate_private_sources(
    request, repository, character_id, scope, *, preferred_address="", max_chars=None,
):
    context = request.character_context
    if context is None:
        return request
    text = context.episodic_reference_context
    candidate = context.source_candidate_context
    if not text and not candidate:
        return request

    packet = _packet(text)
    deferred_packet = _packet(candidate)
    rows = packet["records"] if packet is not None else []
    deferred = deferred_packet["records"] if deferred_packet is not None else []
    originals = {}
    for row in [*rows, *deferred]:
        source_id = row["source_id"]
        if source_id in originals and originals[source_id] != row:
            raise ValueError("Conflicting private source snapshots")
        originals[source_id] = row
    ids = tuple(originals)
    granted = set()
    seen = set()
    for offset in range(0, len(ids), 100):
        fresh = await repository.list_sources(
            character_id, scope, source_message_ids=ids[offset : offset + 100],
        )
        for row in fresh:
            source_id = row["source_message_id"]
            if source_id not in originals or source_id in seen:
                raise ValueError("Invalid fresh scoped private source identities")
            seen.add(source_id)
            original = originals[source_id]
            if row["body"] == original["text"] and row["observed_at"] == original["observed_at"]:
                granted.add(source_id)

    revoked = set(ids) - granted
    if not revoked:
        return request
    bodies = tuple(originals[source_id]["text"] for source_id in revoked)
    admitted = _retain_packet(text, packet, granted)
    deferred_text = _retain_packet(candidate, deferred_packet, granted)
    status = (
        "available"
        if admitted
        else context.memory_source_status
        if deferred_text
        else "authority_changed"
    )
    updated = replace(
        context,
        episodic_reference_context=admitted,
        source_candidate_context=deferred_text,
        memory_source_status=status,
        memory_review_text="",
        memory_field_presence=tuple((key, None) for key, _value in context.memory_field_presence),
    )

    # Source erasure can also change a copied claim's evidence. Keep independent
    # claims; a retained projection is valid only if the old packet still equals
    # the currently scoped stored evidence, never by its old source ID alone.
    affected = tuple(packet for packet in context.memory_packets if revoked.intersection(packet.source_message_ids))
    records = {}
    for offset in range(0, len(affected), 200):
        batch = affected[offset:offset + 200]
        rows = await repository.list_memory_records(
            character_id, scope, limit=None, include_inactive=True,
            memory_ids=tuple(int(packet.memory_id) for packet in batch),
        )
        records.update((str(row["id"]), row) for row in rows)
    removed = set()
    for packet in affected:
        row = records.get(packet.memory_id)
        valid = row is not None and row["content"] == packet.content and row["evidence"] == list(packet.evidence)
        if not valid:
            removed.add(packet.memory_id)
    reference_changed = any(body in context.reference_context for body in bodies)
    if removed or (reference_changed and context.memory_packets):

        updated = without_memories(updated, removed, preferred_address=preferred_address, max_chars=max_chars)
    elif reference_changed:
        updated = replace(
            updated,
            reference_context="",
            used_memory_ids=(),
            memory_packets=(),
            memory_status="no_match",
        )

    changes = {}
    for field in ("conversation_reference_context", "branch_context"):
        if any(body in getattr(updated, field) for body in bodies):
            changes[field] = ""
    if changes:
        updated = replace(updated, **changes)
    return replace(request, character_context=updated, history=without_history_turns(request.history, bodies))


def make_private_context_revalidator(prepared, database):
    """Only trusted application preparation supplies a role/owner scope."""
    scope = getattr(prepared, "user_scope", None)
    character_id = getattr(prepared, "character_id", None)
    if not isinstance(scope, UserScope) or not isinstance(character_id, str) or not character_id:
        return None
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    repository = DatabaseCharacterMemoryRepository(database)
    preferred_address = getattr(getattr(prepared, "relationship", None), "preferred_address", "")
    max_chars = prepared.memory_budget.get("budget_chars")

    async def revalidate(request):
        request = await revalidate_private_memories(
            request, repository, character_id, scope, preferred_address=preferred_address, max_chars=max_chars,
        )
        return await revalidate_private_sources(
            request,
            repository,
            character_id,
            scope,
            preferred_address=preferred_address,
            max_chars=max_chars,
        )

    return revalidate
