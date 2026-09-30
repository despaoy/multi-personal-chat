"""Read-only suppression of facts explicitly replaced in the current message.

This is not a persistence preview: the current user message supplies the new
assertion, and no synthetic saved-memory packet or write acknowledgement is made.
"""
from __future__ import annotations

from character.conditional_memory import same_necessary_condition
from character.memory_extractor import ExtractedMemory, extract_memories
from character.memory_query import PERSONAL_MEMORY_KEYS


def validated_constraint_replacement(item: ExtractedMemory, old_qualifiers: object) -> bool:
    if item.operation != 'replace' or not isinstance(old_qualifiers, dict):
        return False
    new = dict(item.qualifiers)
    return (
        old_qualifiers.get('kind') in {'necessary_condition', 'necessary_condition_set'}
        and new.get('kind') == 'necessary_condition'
        and old_qualifiers.get('action') == new.get('action')
        and not same_necessary_condition(old_qualifiers, new)
        and item in extract_memories(item.evidence)
    )


def shadowed_memory_ids(message: str, records: list[dict]) -> set[str]:
    """Only exact, unambiguous rule slots; unknown/merged evidence stays intact."""
    changes: dict[str, list[ExtractedMemory]] = {}
    for item in extract_memories(message):
        changes.setdefault(item.memory_key, []).append(item)
    hidden: set[str] = set()
    for key, changes_for_key in changes.items():
        # Multiple assertions in one turn need an explicit sequential plan.
        if len(changes_for_key) != 1:
            continue
        item = changes_for_key[0]
        matches = [r for r in records if r.get('memory_key') == key
                   and r.get('status', 'active') in {'active', 'current'}]
        if len(matches) != 1:
            continue
        row = matches[0]
        metadata = row.get('metadata') or {}
        if (not isinstance(metadata, dict) or metadata.get('origin') != 'rule_v2'
                or row.get('scope_level', 'conversation') != 'conversation'
                or not row.get('id') or row.get('content') == item.content):
            continue
        old = metadata.get('qualifiers')
        simple_slot_update = key in PERSONAL_MEMORY_KEYS.values() and not old and not item.qualifiers
        if simple_slot_update or validated_constraint_replacement(item, old):
            hidden.add(str(row['id']))
    return hidden
