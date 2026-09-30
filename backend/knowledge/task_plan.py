"""Bounded task dependencies; opaque requests remain intact, never discarded."""
from __future__ import annotations

import re
from dataclasses import dataclass

from character.memory_query import lookup_fields
from knowledge.query_tasks import is_dialogue_control_clause


@dataclass(frozen=True)
class TurnTask:
    kind: str
    original: str
    query: str
    start: int
    end: int
    memory_fields: tuple[str, ...] = ()


def plan_turn_tasks(message: str) -> tuple[TurnTask, ...]:
    """Expose local-memory dependencies without asserting task completion.

    Only complete personal lookups are marked memory. Other clauses remain
    content tasks, including pronouns, bridge questions and unknown language.
    Quoted messages are opaque to avoid splitting instructions inside data.
    """
    if any(mark in message for mark in ('“', '”', '「', '」', '『', '』', '"')):
        return (TurnTask('content', message, message, 0, len(message)),)
    result = []
    for match in re.finditer(r'[^，,。；;\n]+', message):
        original = match[0]
        query = original.strip()
        if not query:
            continue
        if is_dialogue_control_clause(query):
            kind, fields = 'control', ()
        else:
            query = re.sub(r'^(?:先|再|然后|接着|最后)?(?:请)?(?:告诉我|说说|说|回答一下)', '', query)
            fields = lookup_fields(query)
            kind = 'memory' if fields else 'content'
        result.append(TurnTask(kind, original, query, match.start(), match.end(), fields))
    return tuple(result)
