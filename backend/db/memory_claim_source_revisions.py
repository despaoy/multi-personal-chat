"""Fingerprints for exactly linked sources of currently visible claims.

Claim visibility may span conversations; raw speech visibility does not.
Only irreversible metadata leaves this layer, never a full-source read grant.
"""

import hashlib
import json
from collections import Counter

from db.memory_source import source_scope


def linked_revision_plan(scope, claim_sources):
    pairs = tuple(dict.fromkeys(claim_sources))
    if len(pairs) > 200 or any(
        not isinstance(pair, tuple) or len(pair) != 2 or type(pair[0]) is not int
        or pair[0] < 1 or not isinstance(pair[1], str) for pair in pairs
    ):
        raise ValueError("Expected at most 200 exact claim/source pairs")
    if not pairs:
        return []
    fields = json.loads(scope["scope_key"])
    if not isinstance(fields, list) or len(fields) != 6 or source_scope(*fields) != scope:
        raise ValueError("Expected an exact authenticated source scope")
    params = dict(scope, **dict(zip(("character_id", "platform", "adapter", "sender_id",
                                   "conversation_type", "conversation_id"), fields, strict=True)))
    values = []
    for index, (memory_id, source_id) in enumerate(pairs):
        params[f"memory{index}"] = memory_id
        params[f"source{index}"] = source_id
        values.append(f"(CAST(:memory{index} AS BIGINT),CAST(:source{index} AS TEXT))")
    # Authorize the stored claim first. Its actual link, not the request's
    # conversation or a bare source ID, supplies the source's original scope.
    rows = yield (
        "WITH requested(memory_id,source_message_id) AS (VALUES " + ",".join(values) + ") "
        "SELECT l.memory_id,s.source_message_id,s.observed_at,s.body,s.scope_key AS origin_scope,"
        "c.scope_level FROM requested r JOIN character_memories c ON c.id=r.memory_id "
        "JOIN memory_source_links l ON l.memory_id=c.id "
        "JOIN memory_sources s ON s.source_key=l.source_key AND s.source_message_id=r.source_message_id "
        "WHERE s.owner_key=:owner_key AND s.state='recorded' "
        "AND c.platform=:platform AND c.adapter=:adapter AND c.sender_id=:sender_id AND ("
        "(c.scope_level='conversation' AND c.character_id=:character_id "
        "AND c.conversation_type=:conversation_type AND c.conversation_id=:conversation_id) OR "
        "(c.scope_level='user_character' AND c.character_id=:character_id "
        "AND c.conversation_type='*' AND c.conversation_id='*') OR "
        "(c.scope_level='user_global' AND c.character_id='*' "
        "AND c.conversation_type='*' AND c.conversation_id='*')) "
        "ORDER BY l.memory_id,s.source_message_id,s.source_key",
        params,
    )
    valid = []
    for row in rows:
        try:
            origin = json.loads(row["origin_scope"])
            if (not isinstance(origin, list) or len(origin) != 6
                    or source_scope(*origin)["owner_key"] != scope["owner_key"]
                    or row["scope_level"] == "conversation" and origin != fields
                    or row["scope_level"] == "user_character" and origin[0] != fields[0]):
                continue
        except (TypeError, ValueError):
            continue
        valid.append(row)
    counts = Counter((row["memory_id"], row["source_message_id"]) for row in valid)
    # Legacy pair bindings cannot distinguish two linked original scopes that
    # reuse the same source ID. Fail closed; do not collapse them to one hash.
    return [dict(memory_id=row["memory_id"], source_message_id=row["source_message_id"],
                 observed_at=row["observed_at"], body_sha256=hashlib.sha256(row["body"].encode()).hexdigest())
            for row in valid if counts[row["memory_id"], row["source_message_id"]] == 1]
