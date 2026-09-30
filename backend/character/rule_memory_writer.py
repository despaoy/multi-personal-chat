"""Evidence-preserving rule writes through the existing append-only repository."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from character.conditional_memory import constraint_set_content, same_necessary_condition, verified_constraint_atoms
from character.memory_clock import observation_clock
from character.models import MemoryItem
from db.memory_claim_guard import MemoryClaimConflict, observation_precedes

if TYPE_CHECKING:
    from datetime import datetime

    from character.memory_extractor import ExtractedMemory


def _simple_conditions(text: str) -> dict[str, str]:
    """Only merge explicitly independent beverage restrictions, never free text."""
    match = re.fullmatch(
        r"我(?:很)?喜欢(?:喝)?(?:咖啡|红茶|绿茶|奶茶|牛奶|茶)[，,](?:但|不过)(晚上不喝|不加糖|不加奶)[。]?",
        text.strip(),
    )
    if not match:
        return {}
    value = match[1]
    return {{"晚上不喝": "time", "不加糖": "sugar", "不加奶": "milk"}[value]: value}


async def write_rule_memory(repo, character_id, scope, item: ExtractedMemory, source_message_id=None,
                            *, observed_at: datetime | None = None) -> bool:
    if not item.evidence:
        return False
    observation = observation_clock(observed_at)
    if item.event is not None:
        from character.event_memory import write_event_memory

        return await write_event_memory(repo, character_id, scope, item, source_message_id, observed_at=observation)
    from character.current_turn_memory import validated_constraint_replacement
    for attempt in range(3):
        keys = set(item.aliases) | {item.memory_key}
        reader = getattr(repo, "find_rule_memory_records", None)
        if reader:
            rows = await reader(character_id, scope, tuple(sorted(keys)))
        else:
            rows = await repo.list_memory_records(
                character_id, scope, limit=None, scope_levels=("conversation",), include_inactive=True
            )
        matches = [row for row in rows if row.get("memory_key") in keys and row.get("status", "active") == "active"]
        # Conflicting legacy aliases are not safe to silently merge by recency.
        pending_reason = "ambiguous_legacy_aliases" if len(matches) > 1 else ""
        previous = matches[0] if len(matches) == 1 else None
        if previous and observation_precedes(observation.isoformat(), previous.get('observed_at')):
            # Arrival order is not observation order. Retain the original
            # evidence as inactive history; never merge it into the newer fact.
            if any(row.get('status') == 'archived' and row.get('content') == item.content
                   and row.get('source_message_id') == source_message_id
                   and row.get('observed_at') == observation.isoformat() for row in rows):
                return False
            await repo.append_claim(
                character_id, scope,
                MemoryItem(memory_id='', memory_type=item.memory_type, content=item.content, importance=item.importance),
                memory_key=item.memory_key, relation_type='ADD', status='archived',
                evidence=(item.evidence,), source_message_id=source_message_id,
                observed_at=observation.isoformat(), confidence=0.9,
                metadata={'origin': 'rule_history', 'review_reason': 'older_observation',
                          'qualifiers': dict(item.qualifiers), 'polarity': item.polarity},
            )
            return True
        if previous and previous.get("content") == item.content:
            return False
        metadata = {"origin": "rule_v2", "qualifiers": dict(item.qualifiers), "polarity": item.polarity}
        content = item.content
        evidence = (item.evidence,)
        source_ids = (source_message_id,) if source_message_id else ()
        old_qualifiers = (previous.get("metadata") or {}).get("qualifiers") if previous else None
        old_atoms = verified_constraint_atoms(old_qualifiers, previous['content'], previous.get('evidence')) if previous else None
        new_atoms = verified_constraint_atoms(dict(item.qualifiers), content, evidence)
        if (old_atoms and new_atoms and old_atoms[1] == new_atoms[1]
                and item.operation != 'replace' and new_atoms[0][0] in old_atoms[0]):
            return False
        explicit_addition = bool(previous and not pending_reason and old_atoms and new_atoms
                                 and old_atoms[1] == new_atoms[1] and item.operation == 'append'
                                 and (previous.get('metadata') or {}).get('origin') == 'rule_v2')
        if explicit_addition:
            from character.memory_extractor import MAX_MEMORY_CONTENT_CHARS, extract_memories

            explicit_addition = item in extract_memories(item.evidence)
            if explicit_addition:
                conditions = (*old_atoms[0], *new_atoms[0])
                merged_content = constraint_set_content(old_atoms[1], conditions)
                explicit_addition = len(merged_content) <= MAX_MEMORY_CONTENT_CHARS
                if explicit_addition:
                    content = merged_content
                    evidence = (*previous['evidence'], item.evidence)
                    source_ids = tuple(dict.fromkeys([*(previous.get('source_message_ids') or ()), *source_ids]))
                    metadata.update(operation='append', qualifiers={
                        'kind': 'necessary_condition_set', 'action': old_atoms[1],
                        'conditions': json.dumps(conditions, ensure_ascii=False), 'context': content})
        if (previous and (previous.get('metadata') or {}).get('origin') == 'rule_v2'
                and same_necessary_condition(old_qualifiers, dict(item.qualifiers))):
            return False
        explicit_replacement = (
            previous is not None and not pending_reason
            and (previous.get('metadata') or {}).get('origin') == 'rule_v2'
            and validated_constraint_replacement(item, old_qualifiers)
        )
        if explicit_replacement:
            if old_qualifiers.get('condition') == dict(item.qualifiers).get('condition'):
                return False
            metadata['operation'] = 'replace'
        if previous and old_qualifiers and item.qualifiers and old_qualifiers != dict(item.qualifiers) and not (explicit_replacement or explicit_addition):
            pending_reason = "conditional_change_requires_review"
            old_meta = previous.get("metadata") or {}
            old_conditions = old_meta.get("compatible_conditions") or _simple_conditions(old_qualifiers.get("context", ""))
            new_conditions = _simple_conditions(dict(item.qualifiers).get("context", ""))
            if (item.memory_key.startswith("preference_") and item.polarity == old_meta.get("polarity")
                    and old_conditions and new_conditions
                    and all(key not in old_conditions or old_conditions[key] == value for key, value in new_conditions.items())):
                if all(old_conditions.get(key) == value for key, value in new_conditions.items()):
                    return False
                merged = {**old_conditions, **new_conditions}
                content = previous["content"] + "；补充：" + item.evidence
                if len(content) <= 120:
                    pending_reason = ""
                    metadata["compatible_conditions"] = merged
                    metadata["qualifiers"] = {"context": content}
                    evidence = tuple(dict.fromkeys([*(previous.get("evidence") or ()), item.evidence]))
                    source_ids = tuple(dict.fromkeys([*(previous.get("source_message_ids") or ()), *source_ids]))
        # A repeated positive assertion does not implicitly cancel exceptions.
        if previous and old_qualifiers and not item.qualifiers and item.memory_key.startswith("preference_"):
            old_polarity = (previous.get("metadata") or {}).get("polarity")
            if not old_polarity or old_polarity == item.polarity:
                return False  # Repetition does not revoke an existing exception.
        now = observation.isoformat()
        if pending_reason:
            if any(row.get("status") == "pending" and (
                row.get("content") == content or (
                    isinstance(row.get('metadata'), dict)
                    and row['metadata'].get('origin') == 'rule_candidate'
                    and same_necessary_condition(row['metadata'].get('qualifiers'), dict(item.qualifiers))
                )
            ) for row in rows):
                return False
            await repo.append_claim(
                character_id,
                scope,
                MemoryItem(memory_id="", memory_type=item.memory_type, content=content, importance=item.importance),
                memory_key=item.memory_key,
                relation_type="PENDING",
                evidence=evidence,
                parent_memory_id=previous["id"] if previous else None,
                source_message_id=source_message_id,
                source_message_ids=source_ids,
                confidence=0.65,
                observed_at=now,
                metadata={**metadata, "origin": "rule_candidate", "review_reason": pending_reason},
            )
            return True
        if explicit_addition or explicit_replacement:
            adopted = verified_constraint_atoms(metadata['qualifiers'], content, evidence)
            resolved_ids = []
            for row in rows:
                candidate_meta = row.get('metadata') or {}
                if (row.get('status') != 'pending' or row.get('memory_key') != item.memory_key
                        or not isinstance(candidate_meta, dict)
                        or candidate_meta.get('origin') != 'rule_candidate' or not row.get('source_message_ids')):
                    continue
                candidate = verified_constraint_atoms(candidate_meta.get('qualifiers'), row.get('content', ''), row.get('evidence'))
                if adopted and candidate and adopted[1] == candidate[1] and set(candidate[0]) <= set(adopted[0]):
                    resolved_ids.append(int(row['id']))
            if resolved_ids:
                # Storage closes these exact scoped candidates atomically with
                # the new version. Their original evidence remains untouched.
                metadata['resolved_pending_ids'] = sorted(resolved_ids)
        try:
            await repo.append_claim(
                character_id,
                scope,
                MemoryItem(memory_id="", memory_type=item.memory_type, content=content, importance=item.importance),
                memory_key=previous["memory_key"] if previous else item.memory_key,
                relation_type="SUPERSEDE" if previous else "ADD",
                supersedes_memory_id=previous["id"] if previous else None,
                parent_memory_id=previous["id"] if previous else None,
                evidence=evidence,
                source_message_id=source_message_id,
                source_message_ids=source_ids,
                confidence=0.9,
                observed_at=now,
                valid_from=now,
                metadata=metadata,
            )
            return True
        except ValueError as exc:
            if (not isinstance(exc, MemoryClaimConflict) and str(exc) not in {
                    "rule memory changed concurrently", "pending memory is not available in the exact rule scope"
            }) or attempt == 2:
                raise
    return False
