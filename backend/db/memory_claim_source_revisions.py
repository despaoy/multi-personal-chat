"""Private claim/source fingerprints without granting original speech reads.

The DB transaction reads the exactly linked source to compute a revision;
only irreversible SHA256 metadata leaves this layer. Original speech recall
keeps its owner-wide fence. An independent normalized claim can thus survive
another claim's erasure without exposing fenced raw source text.
"""

import hashlib

from db.memory_source import source_identity


def linked_revision_plan(scope, claim_sources):
    pairs = tuple(dict.fromkeys(claim_sources))
    if len(pairs) > 200 or any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or type(pair[0]) is not int
        or pair[0] < 1
        or not isinstance(pair[1], str)
        for pair in pairs
    ):
        raise ValueError("Expected at most 200 exact claim/source pairs")
    if not pairs:
        return []
    params = dict(scope)
    values = []
    for index, (memory_id, source_id) in enumerate(pairs):
        params[f"memory{index}"] = memory_id
        params[f"source{index}"] = source_identity(scope, source_id)["source_key"]
        values.append(f"(CAST(:memory{index} AS BIGINT),CAST(:source{index} AS TEXT))")
    rows = yield (
        "WITH requested(memory_id,source_key) AS (VALUES " + ",".join(values) + ") "
        "SELECT l.memory_id,s.source_message_id,s.observed_at,s.body "
        "FROM requested r JOIN memory_source_links l ON l.memory_id=r.memory_id AND l.source_key=r.source_key "
        "JOIN memory_sources s ON s.source_key=l.source_key "
        "WHERE s.scope_key=:scope_key AND s.state='recorded' "
        "ORDER BY l.memory_id,s.source_message_id",
        params,
    )
    return [
        dict(
            memory_id=row["memory_id"],
            source_message_id=row["source_message_id"],
            observed_at=row["observed_at"],
            body_sha256=hashlib.sha256(row["body"].encode()).hexdigest(),
        )
        for row in rows
    ]
