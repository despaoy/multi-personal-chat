"""Indexed local source windows; adjacency is context, not a semantic edge."""
from __future__ import annotations

from db.memory_source import read_plan, source_identity


def window_plan(scope, source_message_ids, *, radius=1):
    ids = tuple(dict.fromkeys(source_message_ids))
    if isinstance(source_message_ids, (str, bytes)) or len(ids) > 4:
        raise ValueError("Expected at most four anchor IDs")
    if type(radius) is not int or not 1 <= radius <= 2:
        raise ValueError("Source window radius must be 1..2")
    anchors = yield from read_plan(scope, source_message_ids=ids)
    windows = []
    for anchor in anchors:
        params = dict(scope, at=anchor["observed_at"],
                      anchor_key=source_identity(scope, anchor["source_message_id"])["source_key"], limit=radius)
        sides = []
        for operator, direction in (("<", "DESC"), (">", "ASC")):
            base = ("SELECT source_message_id, observed_at, body, source_key FROM memory_sources "
                    "WHERE scope_key = :scope_key AND state = 'recorded' "
                    "AND observed_at > COALESCE((SELECT revoked_before FROM memory_source_fences "
                    "WHERE owner_key = :owner_key), '')")
            order = f" ORDER BY observed_at {direction}, source_key {direction} LIMIT :limit"
            # Separate time and same-time key ranges keep both engines on the
            # existing scope/state/time/key index, including large time ties.
            same = base + f" AND observed_at = :at AND source_key {operator} :anchor_key" + order
            other = base + f" AND observed_at {operator} :at" + order
            rows = yield ("SELECT source_message_id, observed_at, body FROM (SELECT * FROM ("
                          + same + ") AS same_time UNION ALL SELECT * FROM (" + other
                          + ") AS other_time) AS candidates" + order, params)
            sides.extend(rows)
        windows.append(dict(anchor_id=anchor["source_message_id"], rows=[anchor, *sides]))
    return windows
