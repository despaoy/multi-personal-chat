"""Refresh selected stored claim inputs before every generation send."""

from dataclasses import replace
from datetime import datetime, timezone

from character.context_builder import _memory_is_injectable, compile_reference_context
from character.memory_read_authority import record_version, source_record_version, source_version


async def revalidate_private_memories(request, repository, character_id, scope, *, preferred_address=""):
    context = request.character_context
    if context is None or not context.memory_packets:
        return request
    removed = set()
    unavailable = False
    claim_versions = {}
    for packet in context.memory_packets:
        valid = bool(packet.storage_versions)
        try:
            for memory_id, expected in packet.storage_versions:
                if memory_id not in claim_versions:
                    row = await repository.get_memory_record(int(memory_id), character_id, scope)
                    claim_versions[memory_id] = record_version(row) if row is not None else None
                if claim_versions[memory_id] != expected:
                    valid = False
            if packet.source_record_pairs:
                expected = {(key, sid): version for key, sid, version in packet.source_record_versions}
                pairs = tuple((int(key), sid) for key, sid in packet.source_record_pairs)
                revisions = await repository.linked_source_revisions(character_id, scope, claim_sources=pairs)
                fresh = {(str(row['memory_id']), str(row['source_message_id'])): source_record_version(row)
                         for row in revisions}
                if (set(expected) != set(packet.source_record_pairs) or len(fresh) != len(revisions)
                        or set(fresh) != set(expected) or any(fresh.get(pair) != version for pair, version in expected.items())):
                    valid = False
            if packet.source_versions:
                pairs = tuple((int(memory_id), source_id) for memory_id, source_id, _version in packet.source_versions)
                receipts = await repository.linked_source_receipts(character_id, scope, claim_sources=pairs)
                fresh = {
                    (str(row["memory_id"]), str(row["source_message_id"])): source_version(row) for row in receipts
                }
                if len(fresh) != len(receipts) or any(
                    fresh.get((memory_id, source_id)) != expected
                    for memory_id, source_id, expected in packet.source_versions
                ):
                    valid = False
        except Exception:
            valid = False
            unavailable = True
        if not valid:
            removed.add(packet.memory_id)
    # Use the already validated effective view, not guessed raw storage dates.
    # Take the clock after all asynchronous reads, immediately before rebuilding.
    now = datetime.now(timezone.utc)
    removed.update(packet.memory_id for packet in context.memory_packets if not _memory_is_injectable(packet, now))
    if not removed:
        return request
    remaining = tuple(packet for packet in context.memory_packets if packet.memory_id not in removed)
    shared = tuple(key for key in context.source_shared_memory_ids if key not in removed)
    kwargs = dict(
        preferred_address=preferred_address,
        complete_evidence=True,
        observation_semantics=any(packet.source_observation for packet in remaining),
        max_chars=None,
    )
    full_reference, used_ids = compile_reference_context(remaining, **kwargs)
    visible = tuple(packet for packet in remaining if packet.memory_id not in shared)
    reference = compile_reference_context(visible, **kwargs)[0] if shared else full_reference
    updated = replace(
        context,
        memory_packets=remaining,
        used_memory_ids=used_ids,
        reference_context=reference,
        source_shared_memory_ids=shared,
        source_reference_backup=full_reference if shared else "",
        memory_review_text="",
        memory_field_presence=tuple((key, None) for key, _value in context.memory_field_presence),
        memory_status="available" if remaining else "retrieval_error" if unavailable else "no_match",
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
