"""Atomic deletion of explicitly selected, unlinked original speech records."""

from db.memory_source import lock_owner, source_identity


class SourceClaimConflict(ValueError):
    """A source acquired a claim and must use the claim lifecycle eraser."""


def erase_unlinked_plan(scope, source_message_ids, *, postgres=False):
    if isinstance(source_message_ids, (str, bytes)):
        raise ValueError('Source IDs must be an explicit batch')
    ids = tuple(source_message_ids)
    if not 1 <= len(ids) <= 200 or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError('Expected 1..200 nonempty source IDs')
    ids = tuple(dict.fromkeys(ids))
    params = {f's{i}': source_identity(scope, value)['source_key'] for i, value in enumerate(ids)}
    bound = ','.join(':' + key for key in params)
    # Capture, claim linking and deletion share this owner lock. Checking only
    # an earlier candidate snapshot would miss a concurrently committed claim.
    yield from lock_owner(scope, postgres=postgres)
    linked = yield (f'SELECT memory_id FROM memory_source_links WHERE source_key IN ({bound}) LIMIT 1', params)
    if linked:
        raise SourceClaimConflict('source is linked to a memory claim')
    rows = yield (f"SELECT source_key FROM memory_sources WHERE source_key IN ({bound}) AND state <> 'revoked'", params)
    if not rows:
        return 0
    yield (f'DELETE FROM memory_source_terms WHERE source_key IN ({bound})', params)
    yield ("UPDATE memory_sources SET body = NULL, body_digest = NULL, observed_at = NULL, state = 'revoked' "
           f'WHERE source_key IN ({bound})', params)
    # No owner-wide watermark: unrelated historical speech remains available.
    # Revoked identity anchors reject later capture and claim-link retries.
    return len(rows)
