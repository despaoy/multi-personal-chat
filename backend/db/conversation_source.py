"""Scoped original speech records, deliberately not accepted memory facts.

Shared SQLite/PostgreSQL query contract. Caller supplies authenticated scope;
this repository operation is not an authorization endpoint or a recall policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from db.integration_receipts import HISTORY_DELIVERY_FILTER

if TYPE_CHECKING:
    from character.models import UserScope


@dataclass(frozen=True)
class TurnCursor:
    timestamp: str
    message_id: int


@dataclass(frozen=True)
class StoredUtterance:
    role: Literal['user', 'assistant']
    speaker_id: str
    text: str


@dataclass(frozen=True)
class StoredConversationTurn:
    source_id: int
    session_id: str
    timestamp: str
    utterances: tuple[StoredUtterance, ...]


@dataclass(frozen=True)
class ScopedTurnPage:
    scope: tuple[str, ...]
    turns: tuple[StoredConversationTurn, ...]
    older_rows_omitted: bool
    next_cursor: TurnCursor | None


def turn_source_query(scope: UserScope, character_id: str, *, limit=200, before=None):
    if type(limit) is not int or not 1 <= limit <= 2000:
        raise ValueError('Expected bounded integer source limit')
    if not all(isinstance(value, str) and value.strip() for value in
               (scope.platform, scope.adapter, scope.sender_id, character_id)):
        raise ValueError('Explicit source scope and character are required')
    if scope.conversation_type not in {'private', 'group', 'channel'}:
        raise ValueError('Unsupported conversation type')
    if scope.adapter == 'narrative':
        raise ValueError('Narrative sources require a branch-aware scope adapter')
    local = scope.conversation_type in {'group', 'channel'}
    if local and (not isinstance(scope.conversation_id, str) or not scope.conversation_id.strip()):
        raise ValueError('Conversation-local source requires an ID')
    clauses = ['platform = :platform', 'adapter = :adapter', '"senderId" = :sender',
               '"characterId" = :character', '"branchId" IS NULL', HISTORY_DELIVERY_FILTER]
    params = dict(platform=scope.platform, adapter=scope.adapter, sender=scope.sender_id,
                  character=character_id, limit=limit + 1)
    if local:
        clauses += ['"conversationType" = :kind', '"conversationId" = :conversation']
        params.update(kind=scope.conversation_type, conversation=scope.conversation_id)
    else:
        clauses.append('"conversationType" IN (\'private\', \'\')')
    if before is not None:
        if (not isinstance(before, TurnCursor) or not isinstance(before.timestamp, str)
                or not before.timestamp.strip() or type(before.message_id) is not int or before.message_id < 1):
            raise ValueError('Invalid source cursor')
        params.update(before_at=before.timestamp, before_id=before.message_id)
    base = ('SELECT id, "sessionId", "createdAt", message, reply FROM messages WHERE '
            + ' AND '.join(clauses))
    order = ' ORDER BY "createdAt" DESC, id DESC LIMIT :limit'
    query = base + order
    if before is not None:
        # Disjoint ranges permit a seek even for many identical timestamps.
        # SQLite can scan the whole timestamp tie with a row-value/OR cursor.
        # Bound each range before merging; at most 2*(limit+1) rows are sorted.
        same_time = base + ' AND "createdAt" = :before_at AND id < :before_id' + order
        earlier = base + ' AND "createdAt" < :before_at' + order
        query = ('SELECT * FROM (SELECT * FROM (' + same_time
                 + ') AS same_time UNION ALL SELECT * FROM (' + earlier
                 + ') AS earlier) AS candidates' + order)
    return query, params


def assemble_turn_page(rows, scope: UserScope, character_id: str, *, limit: int) -> ScopedTurnPage:
    """Rows must already be SQL-scoped, descending, and limited to limit+1.

    Preserve both texts including opt-out/correction markers. Recall consumers
    still apply deletion/privacy/admission policy; this isn't an evidence grant.
    """
    selected = rows[:limit]
    turns = []
    for row in reversed(selected):
        utterances = tuple(StoredUtterance(role, speaker, row[key])
            for role, speaker, key in (('user', scope.sender_id, 'message'),
                                      ('assistant', character_id, 'reply')) if row[key])
        turns.append(StoredConversationTurn(int(row['id']), str(row['sessionId']),
                                            str(row['createdAt']), utterances))
    more = len(rows) > limit
    cursor = TurnCursor(turns[0].timestamp, turns[0].source_id) if more and turns else None
    local = scope.conversation_type in {'group', 'channel'}
    source_scope = (scope.platform, scope.adapter, scope.sender_id, character_id,
                    scope.conversation_type, scope.conversation_id if local else '')
    return ScopedTurnPage(source_scope, tuple(turns), more, cursor)
