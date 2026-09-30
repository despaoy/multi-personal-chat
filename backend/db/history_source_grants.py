"""Remove explicitly revoked source turns from model history, not chat storage."""

from db.memory_source import source_identity, source_scope


def filter_plan(rows, platform, adapter, sender_id):
    from character.context_builder import build_user_scope

    keyed = []
    for row in rows:
        row_keys = set()
        if row.get('sourceMessageId') and row.get('characterId'):
            try:
                scope = source_scope(row['characterId'], platform, adapter, sender_id,
                                     row.get('conversationType') or 'private', row.get('conversationId') or '')
                row_keys.add(source_identity(scope, row['sourceMessageId'])['source_key'])
                # Message storage retains the client session ID; memory private
                # scope uses stable sender identity. Reuse the runtime boundary.
                normalized = build_user_scope(platform, adapter, sender_id,
                    row.get('conversationId') or '', row.get('conversationType') or 'private')
                canonical = source_scope(row['characterId'], normalized.platform, normalized.adapter,
                    normalized.sender_id, normalized.conversation_type, normalized.conversation_id)
                row_keys.add(source_identity(canonical, row['sourceMessageId'])['source_key'])
            except ValueError:
                pass  # Legacy/branch rows have no independent source authority.
        keyed.append((row, row_keys))
    keys = tuple(sorted({key for _, row_keys in keyed for key in row_keys}))
    revoked = set()
    for offset in range(0, len(keys), 200):
        params = {f'k{i}': key for i, key in enumerate(keys[offset:offset + 200])}
        placeholders = ','.join(':' + name for name in params)
        found = yield ('SELECT source_key FROM memory_sources WHERE state = \'revoked\' '
                       f'AND source_key IN ({placeholders})', params)
        revoked.update(row['source_key'] for row in found)
    return [row for row, row_keys in keyed if not row_keys.intersection(revoked)]
