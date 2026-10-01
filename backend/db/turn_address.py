"""A turn-derived address update shares the owner erasure write lock."""

from datetime import datetime, timezone

from db.memory_source import lock_owner, source_identity, source_scope, utc_stamp


def address_plan(fields, *, source_message_id, observed_at, address, postgres=False):
    """Keep manual relationship fields intact; reject obsolete turn authority."""
    scope = source_scope(*fields)
    identity = source_identity(scope, source_message_id)
    stamp = utc_stamp(observed_at)
    if not isinstance(address, str) or not 1 <= len(address.strip()) <= 64:
        raise ValueError("An explicit bounded address is required")
    fence = yield from lock_owner(scope, postgres=postgres)
    if stamp <= fence:
        return dict(status="stale")
    rows = yield (
        "SELECT state, observed_at FROM memory_sources WHERE source_key = :source_key",
        identity,
    )
    if rows:
        row = rows[0]
        if row["state"] == "revoked":
            return dict(status="revoked")
        if row["observed_at"] and row["observed_at"] != stamp:
            return dict(status="conflict")
    names = ("character_id", "platform", "adapter", "sender_id", "conversation_type", "conversation_id")
    params = dict(zip(names, fields, strict=True), address=address.strip(), now=utc_stamp(datetime.now(timezone.utc)))
    yield (
        "INSERT INTO character_relationships ("
        "character_id, platform, adapter, sender_id, conversation_type, conversation_id, "
        "relationship_stage, preferred_address, summary, interaction_count, created_at, updated_at) "
        "VALUES (:character_id, :platform, :adapter, :sender_id, :conversation_type, :conversation_id, "
        "'stranger', :address, '', 0, :now, :now) "
        "ON CONFLICT (character_id, platform, adapter, sender_id, conversation_type, conversation_id) "
        "DO UPDATE SET preferred_address = excluded.preferred_address, updated_at = excluded.updated_at",
        params,
    )
    rows = yield (
        "SELECT * FROM character_relationships WHERE " + " AND ".join(name + " = :" + name for name in names),
        params,
    )
    if not rows:
        raise RuntimeError("Turn address write has no matching relationship")
    return dict(status="written", relationship=rows[0])
