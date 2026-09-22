"""Evidence-preserving rule writes through the existing append-only repository."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from character.models import MemoryItem

if TYPE_CHECKING:
    from character.memory_extractor import ExtractedMemory


async def write_rule_memory(repo, character_id, scope, item: ExtractedMemory, source_message_id=None) -> bool:
    if not item.evidence:
        return False
    if item.event is not None:
        from character.event_memory import write_event_memory

        return await write_event_memory(repo, character_id, scope, item, source_message_id)
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
        if previous and previous.get("content") == item.content:
            return False
        metadata = {"origin": "rule_v2", "qualifiers": dict(item.qualifiers), "polarity": item.polarity}
        content = item.content
        evidence = (item.evidence,)
        source_ids = (source_message_id,) if source_message_id else ()
        old_qualifiers = (previous.get("metadata") or {}).get("qualifiers") if previous else None
        if previous and old_qualifiers and item.qualifiers and old_qualifiers != dict(item.qualifiers):
            pending_reason = "conditional_change_requires_review"
        # A repeated positive assertion does not implicitly cancel exceptions.
        if previous and old_qualifiers and not item.qualifiers and item.memory_key.startswith("preference_"):
            old_polarity = (previous.get("metadata") or {}).get("polarity")
            if not old_polarity or old_polarity == item.polarity:
                return False  # Repetition does not revoke an existing exception.
        now = datetime.now(timezone.utc).isoformat()
        if pending_reason:
            if any(row.get("status") == "pending" and row.get("content") == content for row in rows):
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
            if str(exc) != "rule memory changed concurrently" or attempt == 2:
                raise
    return False
