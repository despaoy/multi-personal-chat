"""Refresh selected stored claim inputs before every generation send."""

from dataclasses import replace
from datetime import datetime, timezone

from character.context_builder import _memory_is_injectable, compile_reference_context
from character.memory_read_authority import record_version, source_record_version, source_version


async def _read_source_versions(repository, method, character_id, scope, pairs, version):
    pairs = tuple(dict.fromkeys((int(key), sid) for key, sid in pairs))
    versions = {}
    if not pairs:
        return versions
    reader = getattr(repository, method)
    for start in range(0, len(pairs), 200):
        batch = pairs[start:start + 200]
        requested = {(str(key), sid) for key, sid in batch}
        for row in await reader(character_id, scope, claim_sources=batch):
            pair = (str(row["memory_id"]), str(row["source_message_id"]))
            if pair not in requested or pair in versions:
                raise RuntimeError("Memory source reader returned a duplicate or unrequested pair")
            versions[pair] = version(row)
    return versions


def without_memories(context, removed, *, preferred_address="", max_chars=None):
    """Rebuild the still-authorized packets under this turn's original budget."""
    remaining = tuple(packet for packet in context.memory_packets if packet.memory_id not in removed)
    reference, used_ids = compile_reference_context(
        remaining,
        preferred_address=preferred_address,
        complete_evidence=True,
        max_chars=max_chars,
    )
    admitted = set(used_ids)
    return replace(
        context,
        memory_packets=tuple(packet for packet in remaining if packet.memory_id in admitted),
        used_memory_ids=used_ids,
        reference_context=reference,
        memory_status="available" if used_ids else "no_match",
    )


async def revalidate_private_memories(
    request, repository, character_id, scope, *, preferred_address="", max_chars=None,
):
    context = request.character_context
    if context is None or not context.memory_packets:
        return request
    claim_versions = {}
    for packet in context.memory_packets:
        for memory_id, _expected in packet.storage_versions:
            if memory_id not in claim_versions:
                row = await repository.get_memory_record(int(memory_id), character_id, scope)
                claim_versions[memory_id] = record_version(row) if row is not None else None
    revisions = await _read_source_versions(
        repository, "linked_source_revisions", character_id, scope,
        (pair for packet in context.memory_packets for pair in packet.source_record_pairs),
        source_record_version,
    )
    receipts = await _read_source_versions(
        repository, "linked_source_receipts", character_id, scope,
        ((key, sid) for packet in context.memory_packets for key, sid, _version in packet.source_versions),
        source_version,
    )
    removed = set()
    for packet in context.memory_packets:
        valid = bool(packet.storage_versions)
        for memory_id, expected in packet.storage_versions:
            if claim_versions[memory_id] != expected:
                valid = False
        if packet.source_record_pairs:
            expected = {(key, sid): version for key, sid, version in packet.source_record_versions}
            if (set(expected) != set(packet.source_record_pairs)
                    or any(revisions.get(pair) != version for pair, version in expected.items())):
                valid = False
        if any(receipts.get((key, sid)) != expected for key, sid, expected in packet.source_versions):
            valid = False
        if not valid:
            removed.add(packet.memory_id)
    # Use the already validated effective view, not guessed raw storage dates.
    # Take the clock after all asynchronous reads, immediately before rebuilding.
    now = datetime.now(timezone.utc)
    removed.update(packet.memory_id for packet in context.memory_packets if not _memory_is_injectable(packet, now))
    if not removed:
        return request
    updated = without_memories(context, removed, preferred_address=preferred_address, max_chars=max_chars)
    updated = replace(
        updated,
        memory_review_text="",
        memory_field_presence=tuple((key, None) for key, _value in context.memory_field_presence),
    )
    bodies = tuple(
        value
        for packet in context.memory_packets
        if packet.memory_id in removed
        for value in (packet.content, *packet.evidence)
        if value
    )
    changes = {
        name: ""
        for name in ("conversation_reference_context", "branch_context")
        if any(body in getattr(updated, name) for body in bodies)
    }
    if changes:
        updated = replace(updated, **changes)
    from inference.private_context_authority import _without_history_turns

    return replace(request, character_context=updated, history=_without_history_turns(request.history, bodies))
