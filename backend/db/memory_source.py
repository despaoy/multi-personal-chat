"""Independent speech storage; never a grant to treat speech as a user fact.

The small SQL plans run inside the caller's SQLite/PG transaction. PostgreSQL
locks the owner fence before writes; SQLite uses BEGIN IMMEDIATE. Keeping the
plans shared makes revocation semantics identical, including delayed capture.
No text is duplicated on claims, and no chat-history fallback is provided.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from db.memory_source_search import SCHEMA as SEARCH_SCHEMA
from db.memory_source_search import index_plan

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS memory_source_fences (
        owner_key TEXT PRIMARY KEY, revoked_before TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS memory_sources (
        source_key TEXT PRIMARY KEY, owner_key TEXT NOT NULL, scope_key TEXT NOT NULL,
        source_message_id TEXT NOT NULL, observed_at TEXT, body TEXT,
        body_digest TEXT, state TEXT NOT NULL CHECK (state IN ('pending', 'recorded', 'revoked')),
        CHECK ((state = 'recorded' AND body IS NOT NULL AND observed_at IS NOT NULL)
            OR (state <> 'recorded' AND body IS NULL)))""",
    """CREATE INDEX IF NOT EXISTS idx_memory_sources_scope
        ON memory_sources (scope_key, state, observed_at, source_key)""",
    """CREATE TABLE IF NOT EXISTS memory_source_links (
        memory_id INTEGER NOT NULL, source_key TEXT NOT NULL,
        PRIMARY KEY (memory_id, source_key))""",
    "CREATE INDEX IF NOT EXISTS idx_memory_source_links_source ON memory_source_links(source_key)",
) + SEARCH_SCHEMA


class ClaimSourceRevokedError(ValueError):
    """A trusted pre-erasure receipt cannot commit a late claim."""


def validate_claim_receipt(revoked_before, observed_at):
    """Called under the owner's write lock, not before the model request.

    Current writers supply a server receipt. Unknown legacy timestamps are not
    assigned a timezone or treated as proof; they retain legacy semantics.
    """
    if not revoked_before or not observed_at:
        return
    try:
        stamp = utc_stamp(datetime.fromisoformat(observed_at))
    except (TypeError, ValueError):
        return
    if stamp <= revoked_before:
        raise ClaimSourceRevokedError("memory source receipt predates erasure")


def utc_stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Source time requires a trusted timezone-aware receipt timestamp")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def owner_scope(character_id, platform, adapter, sender_id, conversation_type, conversation_id):
    return dict(owner_key=json.dumps((platform, adapter, sender_id)),
                scope_key=json.dumps((character_id, platform, adapter, sender_id,
                                      conversation_type, conversation_id)))


def source_scope(character_id, platform, adapter, sender_id, conversation_type, conversation_id):
    fields = (character_id, platform, adapter, sender_id, conversation_type, conversation_id)
    if any(not isinstance(item, str) for item in fields):
        raise ValueError("Explicit source identity is required")
    if any(not item.strip() or item == "*" for item in fields[:5]):
        raise ValueError("Source scope cannot inherit expanded claim scope")
    if conversation_type not in {"private", "group", "channel"}:
        raise ValueError("Unsupported source conversation type")
    if conversation_id == "*" or (conversation_type != "private" and not conversation_id.strip()):
        raise ValueError("Conversation-local source requires an exact ID")
    if adapter == "narrative":
        raise ValueError("Narrative source needs explicit branch authorization")
    # JSON tuples are collision-free identity serialization, not text fingerprints.
    return owner_scope(*fields)


def claim_source_identity(fields, source_message_id):
    """Legacy/branch claims remain supported, without inventing source authority."""
    if not source_message_id:
        return None
    try:
        return source_identity(source_scope(*fields), source_message_id)
    except ValueError:
        return None


def source_identity(scope, source_message_id):
    if not isinstance(source_message_id, str) or not source_message_id.strip():
        raise ValueError("Source message ID is required")
    return dict(scope, source_message_id=source_message_id,
                source_key=json.dumps((scope["scope_key"], source_message_id)))


def admission_plan(identity, body):
    """Read one authenticated source identity without returning its body.

    A recorded source and its owner fence are compared in one SQL snapshot.
    This is read-only admission; final capture still enforces write rights.
    """
    if not isinstance(body, str) or not body.strip():
        raise ValueError("Complete source text is required")
    rows = yield ("SELECT state, observed_at, body_digest, body = :body AS body_matches, "
                  "(SELECT revoked_before FROM memory_source_fences WHERE owner_key = :owner_key) AS revoked_before "
                  "FROM memory_sources WHERE source_key = :source_key AND scope_key = :scope_key AND owner_key = :owner_key",
                  dict(identity, body=body))
    if not rows:
        return "new"
    row = rows[0]
    if row["state"] == "revoked":
        return "revoked"
    if row["state"] == "pending":
        if row["observed_at"] and row["observed_at"] <= (row["revoked_before"] or ""):
            return "stale"
        if row["body_digest"] and row["body_digest"] != hashlib.sha256(body.encode()).hexdigest():
            return "conflict"
        return "pending"
    if row["state"] != "recorded":
        raise ValueError("Unsupported source state")
    if row["observed_at"] <= (row["revoked_before"] or ""):
        return "stale"
    return "recorded" if row["body_matches"] else "conflict"


def lock_owner(scope, *, postgres=False):
    yield ("INSERT INTO memory_source_fences (owner_key, revoked_before) "
           "VALUES (:owner_key, '') ON CONFLICT (owner_key) DO NOTHING", scope)
    rows = yield ("SELECT revoked_before FROM memory_source_fences WHERE owner_key = :owner_key"
                  + (" FOR UPDATE" if postgres else ""), scope)
    return rows[0]["revoked_before"]


def reserve_plan(identity, body, observed_at, *, postgres=False):
    """Atomically bind a current message identity before model admission.

    Pending bindings contain a digest and trusted receipt, never speech.
    Same-body retries reuse the first receipt; capture and erasure share locks.
    """
    stamp = utc_stamp(observed_at)
    fence = yield from lock_owner(identity, postgres=postgres)
    if stamp <= fence:
        return dict(status="stale", observed_at=None)
    status = yield from admission_plan(identity, body)
    if status not in {"new", "pending"}:
        return dict(status=status, observed_at=None)
    params = dict(identity, body_digest=hashlib.sha256(body.encode()).hexdigest(), observed_at=stamp)
    yield ("INSERT INTO memory_sources "
           "(source_key, owner_key, scope_key, source_message_id, body_digest, observed_at, state) "
           "VALUES (:source_key, :owner_key, :scope_key, :source_message_id, :body_digest, :observed_at, 'pending') "
           "ON CONFLICT (source_key) DO UPDATE SET body_digest = excluded.body_digest, "
           "observed_at = excluded.observed_at WHERE memory_sources.state = 'pending' "
           "AND memory_sources.body_digest IS NULL", params)
    rows = yield ("SELECT observed_at FROM memory_sources WHERE source_key = :source_key", identity)
    return dict(status="pending", observed_at=rows[0]["observed_at"])


def capture_plan(identity, body, observed_at, *, postgres=False):
    if not isinstance(body, str) or not body.strip():
        raise ValueError("Complete source text is required")
    stamp = utc_stamp(observed_at)
    fence = yield from lock_owner(identity, postgres=postgres)
    if stamp <= fence:
        return "stale"
    rows = yield ("SELECT state, body_digest, observed_at FROM memory_sources WHERE source_key = :source_key", identity)
    if rows and rows[0]["state"] == "pending" and rows[0]["body_digest"]:
        row = rows[0]
        if row["observed_at"] <= fence:
            return "stale"
        if row["body_digest"] != hashlib.sha256(body.encode()).hexdigest() or row["observed_at"] != stamp:
            return "conflict"
    params = dict(identity, body=body, observed_at=stamp)
    # Only pending anchors can be filled. Replays cannot overwrite text or undo
    # revocation, even when a caller retries with a newer observation timestamp.
    yield ("INSERT INTO memory_sources "
           "(source_key, owner_key, scope_key, source_message_id, observed_at, body, state) "
           "VALUES (:source_key, :owner_key, :scope_key, :source_message_id, :observed_at, :body, 'recorded') "
           "ON CONFLICT (source_key) DO UPDATE SET body = excluded.body, "
           "observed_at = excluded.observed_at, body_digest = NULL, state = 'recorded' "
           "WHERE memory_sources.state = 'pending'", params)
    rows = yield ("SELECT state, body, observed_at FROM memory_sources WHERE source_key = :source_key", params)
    row = rows[0]
    if row["state"] == "revoked":
        return "revoked"
    if row["body"] != body or row["observed_at"] != stamp:
        return "conflict"
    yield from index_plan(identity, body)
    return "recorded"


def link_plan(identity, memory_id):
    """Caller holds owner lock and has just inserted the claim in this transaction.

    Use the current source's exact original scope, not normalized claim scope or
    arbitrary historical source IDs supplied by the model.
    """
    yield ("INSERT INTO memory_sources "
           "(source_key, owner_key, scope_key, source_message_id, state) "
           "VALUES (:source_key, :owner_key, :scope_key, :source_message_id, 'pending') "
           "ON CONFLICT (source_key) DO NOTHING", identity)
    rows = yield ('SELECT state FROM memory_sources WHERE source_key = :source_key', identity)
    if rows[0]['state'] == 'revoked':
        # Receipt age is not sufficient: retries/legacy writers may have no
        # original timestamp. Identity revocation survives all receipt changes.
        # The caller's transaction rolls back the claim inserted before linking.
        raise ClaimSourceRevokedError('memory source identity was revoked')
    yield ("INSERT INTO memory_source_links (memory_id, source_key) VALUES (:memory_id, :source_key) "
           "ON CONFLICT (memory_id, source_key) DO NOTHING", dict(identity, memory_id=memory_id))


def revoke_plan(scope, memory_ids, *, clear=False):
    """Caller holds owner lock; source purge and claim deletion commit together.

    The remaining revoked anchor contains identity only, never erased text or a
    reason. A per-owner receipt watermark excludes work queued before deletion;
    Independent sibling facts survive. All earlier raw source context becomes
    ineligible for recall, including unlinked repetitions of the erased fact.
    """
    now = utc_stamp(datetime.now(timezone.utc))
    params = dict(scope, now=now)
    yield ("UPDATE memory_source_fences SET revoked_before = "
           "CASE WHEN revoked_before < :now THEN :now ELSE revoked_before END "
           "WHERE owner_key = :owner_key", params)
    if clear:
        yield ("DELETE FROM memory_source_terms WHERE scope_key = :scope_key", scope)
        yield ("UPDATE memory_sources SET body = NULL, body_digest = NULL, observed_at = NULL, state = 'revoked' "
               "WHERE scope_key = :scope_key", scope)
    # Chunk statements keep SQLite bind counts bounded for bulk erasure.
    ids = tuple(memory_ids)
    for offset in range(0, len(ids), 200):
        bound = {f"id{i}": value for i, value in enumerate(ids[offset:offset + 200])}
        placeholders = ",".join(":" + key for key in bound)
        yield ("DELETE FROM memory_source_terms WHERE source_key IN (SELECT source_key "
               f"FROM memory_source_links WHERE memory_id IN ({placeholders}))", bound)
        yield ("UPDATE memory_sources SET body = NULL, body_digest = NULL, observed_at = NULL, state = 'revoked' "
               "WHERE owner_key = :owner_key AND source_key IN (SELECT source_key "
               f"FROM memory_source_links WHERE memory_id IN ({placeholders}))", dict(scope, **bound))
        yield (f"DELETE FROM memory_source_links WHERE memory_id IN ({placeholders})", bound)


def read_plan(scope, *, source_message_ids=None, limit=100):
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("Source read limit must be 1..200")
    params = dict(scope, limit=limit)
    where = "scope_key = :scope_key AND state = 'recorded'"
    if source_message_ids is not None:
        if isinstance(source_message_ids, (str, bytes)):
            raise ValueError("Source IDs must be a batch, not a single string")
        ids = tuple(dict.fromkeys(source_message_ids))
        if len(ids) > 200:
            raise ValueError("At most 200 source IDs per batch")
        if not ids:
            return []
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise ValueError("Invalid source ID")
        # Derive exact primary keys from the authenticated ORIGINAL scope. An
        # IN filter on message_id after a scope/time scan could traverse all of
        # a large conversation for missing IDs. This path does bounded PK seeks.
        bound = {f"source{i}": source_identity(scope, item)["source_key"] for i, item in enumerate(ids)}
        params.update(bound)
        where = "state = 'recorded' AND source_key IN (" + ",".join(":" + key for key in bound) + ")"
    # A missing claim link does NOT prove the full utterance lacks an erased
    # fact. Fence the whole raw-source layer for this owner on explicit erase;
    # normalized independent facts are not affected. This scalar PK lookup is
    # in the same statement/snapshot as the source read, not an in-memory check.
    where += (" AND observed_at > COALESCE((SELECT revoked_before FROM memory_source_fences "
              "WHERE owner_key = :owner_key), '')")
    rows = yield ("SELECT source_message_id, observed_at, body FROM memory_sources WHERE "
                  + where + " ORDER BY observed_at DESC, source_key DESC LIMIT :limit", params)
    return rows


def linked_read_plan(scope, memory_ids, *, limit=100):
    ids = tuple(dict.fromkeys(memory_ids))
    if len(ids) > 200 or any(type(item) is not int or item < 1 for item in ids):
        raise ValueError("Expected at most 200 valid claim IDs")
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("Source read limit must be 1..200")
    if not ids:
        return []
    bound = {f"memory{i}": item for i, item in enumerate(ids)}
    placeholders = ",".join(":" + key for key in bound)
    rows = yield ("SELECT DISTINCT l.memory_id, s.source_message_id, s.observed_at, s.body FROM memory_source_links l "
                  "JOIN memory_sources s ON s.source_key = l.source_key "
                  f"WHERE l.memory_id IN ({placeholders}) AND s.scope_key = :scope_key AND s.state = 'recorded' "
                  "AND s.observed_at > COALESCE((SELECT revoked_before FROM memory_source_fences "
                  "WHERE owner_key = :owner_key), '') ORDER BY s.observed_at DESC, s.source_message_id, l.memory_id LIMIT :limit",
                  dict(scope, **bound, limit=limit))
    return rows


def run_sqlite(cursor, plan):
    rows = None
    while True:
        try:
            sql, params = plan.send(rows)
        except StopIteration as result:
            return result.value
        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()] if cursor.description else []


async def run_postgres(session, plan):
    from sqlalchemy import text

    rows = None
    while True:
        try:
            sql, params = plan.send(rows)
        except StopIteration as result:
            return result.value
        result = await session.execute(text(sql), params)
        rows = [dict(row) for row in result.mappings().all()] if result.returns_rows else []
