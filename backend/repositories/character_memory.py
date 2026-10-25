"""Repository boundary for character relationship and long-term memory.

只做两件事：
1. 把 character.models 的不可变对象与数据库行 dict 互转；
2. 把同步数据库门面（SQLiteDB / SyncPgAdapter）包装成异步仓储接口。

隔离规则由 UserScope.memory_scope_key 语义保证：私聊按用户隔离，
群聊/频道按"会话+用户"隔离，跨角色/跨平台/跨适配器互不可见。
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from datetime import datetime

from character.memory_read_authority import MemoryRecordError
from character.models import (
    MemoryItem,
    RelationshipState,
    UserScope,
)
from character.temporal_provenance import semantic_observation_clock
from db.memory_claim_guard import (
    MEMORY_RELATION_TYPES as _VALID_RELATION_TYPES,
    MEMORY_SCOPE_LEVELS as _VALID_SCOPE_LEVELS,
    MEMORY_STATUSES as _VALID_MEMORY_STATUSES,
)

_VALID_STAGES: tuple[str, ...] = ("stranger", "acquaintance", "familiar", "close")
_VALID_MEMORY_TYPES: tuple[str, ...] = (
    "user_fact",
    "shared_event",
    "promise",
    "conversation_summary",
)


def _require_enum(value: Any, field: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise MemoryRecordError(field, f"Invalid memory {field}")
    return value


class CharacterMemoryRepository(Protocol):
    """角色关系与长期记忆的持久化接口。"""

    @property
    def storage_owner(self) -> object: ...

    async def erase_unlinked_sources(self, character_id: str, user_scope: UserScope, *,
                                     source_message_ids: tuple[str, ...]) -> int: ...

    async def capture_source(self, character_id: str, user_scope: UserScope, *,
                             source_message_id: str, body: str, observed_at: datetime) -> str: ...

    async def source_windows(self, character_id: str, user_scope: UserScope, *,
                             source_message_ids: tuple[str, ...], radius: int = 1) -> list[dict[str, Any]]: ...

    async def search_sources(self, character_id: str, user_scope: UserScope, *, query: str,
                             limit: int | None = 32) -> list[dict[str, Any]]: ...

    async def linked_source_revisions(self, character_id: str, user_scope: UserScope, *,
                                      claim_sources: tuple[tuple[int, str], ...]) -> list[dict[str, Any]]: ...

    async def linked_source_receipts(self, character_id: str, user_scope: UserScope, *,
                                     claim_sources: tuple[tuple[int,str],...]) -> list[dict[str,Any]]: ...

    async def linked_sources(self, character_id: str, user_scope: UserScope, *,
                             memory_ids: tuple[int, ...]) -> list[dict[str, Any]]: ...

    async def list_sources(self, character_id: str, user_scope: UserScope, *,
                           source_message_ids: tuple[str, ...] | None = None,
                           limit: int = 100) -> list[dict[str, Any]]: ...

    async def list_relationship_notes(self, character_id: str, user_scope: UserScope) -> list[dict[str, Any]]: ...

    async def get_relationship_record(
        self, character_id: str, user_scope: UserScope
    ) -> Optional[dict[str, Any]]: ...

    async def upsert_relationship(
        self, character_id: str, user_scope: UserScope, state: RelationshipState
    ) -> dict[str, Any]: ...

    async def set_address_from_turn(self, character_id: str, user_scope: UserScope, *,
                                    source_message_id: str, observed_at: datetime, address: str) -> dict[str, Any]: ...

    async def increment_interaction(self, character_id: str, user_scope: UserScope) -> int: ...

    async def list_memory_records(
        self,
        character_id: str,
        user_scope: UserScope,
        limit: int | None = 30,
        *,
        include_inactive: bool = False,
        scope_levels: Optional[tuple[str, ...]] = None,
        memory_ids: Optional[tuple[int, ...]] = None,
    ) -> list[dict[str, Any]]: ...

    async def find_rule_memory_records(
        self, character_id: str, user_scope: UserScope, keys: tuple[str, ...]
    ) -> list[dict[str, Any]]: ...

    async def get_memory_record(
        self, memory_id: int, character_id: str, user_scope: UserScope
    ) -> Optional[dict[str, Any]]: ...

    async def append_claim(
        self,
        character_id: str,
        user_scope: UserScope,
        memory: MemoryItem,
        *,
        memory_key: str,
        relation_type: str = "ADD",
        scope_level: str = "conversation",
        status: Optional[str] = None,
        parent_memory_id: Optional[int] = None,
        supersedes_memory_id: Optional[int] = None,
        evidence: tuple[str | dict[str, str], ...] = (),
        confidence: float = 1.0,
        attributed_to: str = "user",
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        observed_at: Optional[str] = None,
        source_message_id: Optional[str] = None,
        source_message_ids: tuple[str, ...] = (),
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]: ...

    async def delete_memory(
        self, memory_id: int, character_id: str, user_scope: UserScope
    ) -> bool: ...

    async def clear_memories(self, character_id: str, user_scope: UserScope) -> int: ...

    async def erase_memory(
        self,
        character_id: str,
        user_scope: UserScope,
        *,
        memory_id: Optional[int] = None,
        memory_key: Optional[str] = None,
        scope_level: Optional[str] = None,
        protected_memory_keys: tuple[str, ...] = (),
    ) -> int: ...


def relationship_from_record(row: dict[str, Any] | None) -> RelationshipState:
    if not row:
        return RelationshipState()
    stage = row.get("relationship_stage", "stranger")
    if stage not in _VALID_STAGES:
        raise ValueError("Invalid relationship_stage in stored relationship")
    return RelationshipState(
        stage=stage,  # type: ignore[arg-type]
        preferred_address=str(row.get("preferred_address") or ""),
        summary=str(row.get("summary") or ""),
    )


def _decode_json(value: Any, expected_type: type, field: str) -> Any:
    if value is None:
        return expected_type()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise MemoryRecordError(field, f"Invalid JSON in memory field {field}") from None
    if not isinstance(value, expected_type):
        raise MemoryRecordError(field, f"Memory field {field} must contain {expected_type.__name__}")
    return value


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _decode_text_list(value: Any, field: str) -> list[str]:
    values = _decode_json(value, list, field)
    if not all(_nonempty_text(item) for item in values):
        raise MemoryRecordError(field, f"Memory field {field} must contain non-empty text elements")
    return values


def _decode_evidence(value: Any) -> list[str | dict[str, str]]:
    values = _decode_json(value, list, "evidence_json")
    for item in values:
        if _nonempty_text(item):
            continue
        if (isinstance(item, dict) and set(item) == {"message_id", "quote"}
                and all(_nonempty_text(part) for part in item.values())):
            continue
        raise MemoryRecordError("evidence_json", "Memory field evidence_json must contain text or attributed quote records")
    return values


def decode_legacy_memory_lifecycle(row: dict[str, Any]) -> dict[str, str]:
    """Normalize the documented legacy evaluation format before recall."""
    status = row.get("status", row.get("memory_status", "active"))
    if isinstance(status, str) and status.strip().lower() == "current":
        status = "active"
    relation = row.get("relation_type", row.get("relation", "ADD"))
    return {
        "status": _require_enum(status.strip().lower() if isinstance(status, str) else status,
                                "status", (*_VALID_MEMORY_STATUSES, "erased", "deleted")),
        "relation_type": _require_enum(relation.strip().upper() if isinstance(relation, str) else relation,
                                       "relation_type", (*_VALID_RELATION_TYPES, "ERASE")),
    }


def decode_memory_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw or typed evidence once at a repository adapter boundary."""
    values = {}
    for field in ("evidence", "source_message_ids"):
        if field in row:
            value = row[field]
            if not isinstance(value, (list, tuple)):
                raise MemoryRecordError(field, f"Memory {field} must contain a list")
            values[field] = list(value)
        else:
            values[field] = row.get(field + "_json")
    ids = _decode_text_list(values["source_message_ids"], "source_message_ids_json").copy()
    legacy_source = str(row.get("source_message_id") or "").strip()
    if legacy_source and legacy_source not in ids:
        ids.insert(0, legacy_source)
    return {"source_message_ids": ids, "evidence": _decode_evidence(values["evidence"])}


def _decode_memory_record(row: dict[str, Any]) -> dict[str, Any]:
    """Expose JSON payloads as typed convenience fields without hiding raw columns."""
    record = dict(row)
    record.update(decode_memory_evidence({
        key: record.get(key) for key in ("evidence_json", "source_message_ids_json", "source_message_id")
    }))
    metadata = _decode_json(record.get("metadata_json"), dict, "metadata_json")
    record["metadata"] = metadata
    for field, allowed in (("memory_type", _VALID_MEMORY_TYPES), ("status", _VALID_MEMORY_STATUSES),
                           ("relation_type", _VALID_RELATION_TYPES)):
        _require_enum(record.get(field), field, allowed)
    semantic_observation_clock(metadata, record.get("observed_at"))
    record["attributed_to"] = str(metadata.get("attributed_to") or "user")
    record["qualifiers"] = metadata.get("qualifiers", [])
    return record


class DatabaseCharacterMemoryRepository:
    """适配现有同步数据库门面的角色记忆仓储。"""

    def __init__(self, database: Any) -> None:
        self._database = database

    @property
    def storage_owner(self) -> object:
        """The backing store owns buffering across short-lived repository views."""
        return self._database

    async def source_windows(self, character_id: str, user_scope: UserScope, *,
                             source_message_ids: tuple[str, ...], radius: int = 1) -> list[dict[str, Any]]:
        reader = self._database.memory_source_windows
        return await asyncio.to_thread(
            reader, character_id, user_scope.platform, user_scope.adapter, user_scope.sender_id,
            user_scope.conversation_type, user_scope.conversation_id,
            source_message_ids=source_message_ids, radius=radius)

    async def linked_source_receipts(self, character_id: str, user_scope: UserScope, *,
                                     claim_sources: tuple[tuple[int,str],...]) -> list[dict[str,Any]]:
        reader = self._database.linked_memory_source_receipts
        return await asyncio.to_thread(reader,character_id,user_scope.platform,user_scope.adapter,user_scope.sender_id,
            user_scope.conversation_type,user_scope.conversation_id,claim_sources=claim_sources)

    async def linked_source_revisions(self, character_id: str, user_scope: UserScope, *,
                                     claim_sources: tuple[tuple[int,str],...]) -> list[dict[str,Any]]:
        reader = self._database.linked_memory_source_revisions
        return await asyncio.to_thread(reader,character_id,user_scope.platform,user_scope.adapter,user_scope.sender_id,
            user_scope.conversation_type,user_scope.conversation_id,claim_sources=claim_sources)

    async def linked_sources(self, character_id: str, user_scope: UserScope, *,
                             memory_ids: tuple[int, ...]) -> list[dict[str, Any]]:
        reader = self._database.linked_memory_sources
        return await asyncio.to_thread(
            reader, character_id, user_scope.platform, user_scope.adapter, user_scope.sender_id,
            user_scope.conversation_type, user_scope.conversation_id, memory_ids=memory_ids)

    async def search_sources(self, character_id: str, user_scope: UserScope, *, query: str,
                             limit: int | None = 32) -> list[dict[str, Any]]:
        reader = self._database.search_memory_sources
        return await asyncio.to_thread(
            reader, character_id, user_scope.platform, user_scope.adapter, user_scope.sender_id,
            user_scope.conversation_type, user_scope.conversation_id, query=query, limit=limit)

    async def capture_source(self, character_id: str, user_scope: UserScope, *,
                             source_message_id: str, body: str, observed_at: datetime) -> str:
        return await asyncio.to_thread(
            self._database.capture_memory_source, character_id, user_scope.platform, user_scope.adapter,
            user_scope.sender_id, user_scope.conversation_type, user_scope.conversation_id,
            source_message_id=source_message_id, body=body, observed_at=observed_at)

    async def list_sources(self, character_id: str, user_scope: UserScope, *,
                           source_message_ids: tuple[str, ...] | None = None,
                           limit: int = 100) -> list[dict[str, Any]]:
        reader = self._database.list_memory_sources
        return await asyncio.to_thread(
            reader, character_id, user_scope.platform, user_scope.adapter, user_scope.sender_id,
            user_scope.conversation_type, user_scope.conversation_id,
            source_message_ids=source_message_ids, limit=limit)

    async def erase_unlinked_sources(self, character_id: str, user_scope: UserScope, *,
                                     source_message_ids: tuple[str, ...]) -> int:
        eraser = self._database.erase_unlinked_memory_sources
        return int(await asyncio.to_thread(eraser, character_id, user_scope.platform, user_scope.adapter,
            user_scope.sender_id, user_scope.conversation_type, user_scope.conversation_id,
            source_message_ids=source_message_ids))

    async def find_rule_memory_records(self, character_id: str, user_scope: UserScope, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(
            self._database.list_character_memory_claims,
            character_id, user_scope.platform, user_scope.adapter, user_scope.sender_id,
            user_scope.conversation_type, user_scope.conversation_id, None,
            scope_levels=("conversation",), memory_keys=keys, include_inactive=True,
        )
        return [_decode_memory_record(row) for row in rows]

    async def list_relationship_notes(self, character_id: str, user_scope: UserScope) -> list[dict[str, Any]]:
        from character.natural_relationship import is_note

        reader = self._database.list_character_memory_claims
        rows = await asyncio.to_thread(
            reader, character_id, user_scope.platform, user_scope.adapter,
            user_scope.sender_id, user_scope.conversation_type, user_scope.conversation_id,
            None, relationship_notes_only=True,
        )
        rows = [_decode_memory_record(row) for row in rows]
        return [row for row in rows if is_note(row)]

    async def get_relationship_record(
        self, character_id: str, user_scope: UserScope
    ) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(
            self._database.get_character_relationship,
            character_id,
            user_scope.platform,
            user_scope.adapter,
            user_scope.sender_id,
            user_scope.conversation_type,
            user_scope.conversation_id,
        )

    async def upsert_relationship(
        self, character_id: str, user_scope: UserScope, state: RelationshipState
    ) -> dict[str, Any]:
        """写入关系状态并返回写入后的完整记录（含 interaction_count 与时间戳）。

        管理接口直接把返回值响应给前端，必须返回数据库记录而非 None。
        """
        if state.stage not in _VALID_STAGES:
            raise ValueError(f"未知的关系阶段: {state.stage!r}")
        return dict(
            await asyncio.to_thread(
                self._database.upsert_character_relationship,
                character_id,
                user_scope.platform,
                user_scope.adapter,
                user_scope.sender_id,
                user_scope.conversation_type,
                user_scope.conversation_id,
                state.stage,
                state.preferred_address,
                state.summary,
            )
        )

    async def set_address_from_turn(self, character_id: str, user_scope: UserScope, *,
                                    source_message_id: str, observed_at: datetime, address: str) -> dict[str, Any]:
        """Turn writes use original receipt authority, unlike manual updates."""
        return await asyncio.to_thread(
            self._database.set_character_address_from_turn, character_id,
            user_scope.platform, user_scope.adapter, user_scope.sender_id,
            user_scope.conversation_type, user_scope.conversation_id,
            source_message_id=source_message_id, observed_at=observed_at, address=address)

    async def increment_interaction(self, character_id: str, user_scope: UserScope) -> int:
        return int(
            await asyncio.to_thread(
                self._database.increment_character_interaction,
                character_id,
                user_scope.platform,
                user_scope.adapter,
                user_scope.sender_id,
                user_scope.conversation_type,
                user_scope.conversation_id,
            )
        )

    async def list_memory_records(
        self,
        character_id: str,
        user_scope: UserScope,
        limit: int | None = 30,
        *,
        include_inactive: bool = False,
        scope_levels: Optional[tuple[str, ...]] = None,
        memory_ids: Optional[tuple[int, ...]] = None,
    ) -> list[dict[str, Any]]:
        """Return visible claims, optionally restricted to exact storage IDs."""
        invalid_levels = set(scope_levels or ()) - set(_VALID_SCOPE_LEVELS)
        if invalid_levels:
            raise ValueError(f"未知的记忆作用域: {sorted(invalid_levels)!r}")
        rows = await asyncio.to_thread(
            self._database.list_character_memory_claims,
            character_id,
            user_scope.platform,
            user_scope.adapter,
            user_scope.sender_id,
            user_scope.conversation_type,
            user_scope.conversation_id,
            limit,
            include_inactive=include_inactive,
            scope_levels=scope_levels,
            memory_ids=memory_ids,
        )
        return [_decode_memory_record(row) for row in rows]

    async def get_memory_record(
        self, memory_id: int, character_id: str, user_scope: UserScope
    ) -> Optional[dict[str, Any]]:
        row = await asyncio.to_thread(
            self._database.get_character_memory_claim,
            int(memory_id),
            character_id,
            user_scope.platform,
            user_scope.adapter,
            user_scope.sender_id,
            user_scope.conversation_type,
            user_scope.conversation_id,
        )
        return _decode_memory_record(row) if row else None

    async def append_claim(
        self,
        character_id: str,
        user_scope: UserScope,
        memory: MemoryItem,
        *,
        memory_key: str,
        relation_type: str = "ADD",
        scope_level: str = "conversation",
        status: Optional[str] = None,
        parent_memory_id: Optional[int] = None,
        supersedes_memory_id: Optional[int] = None,
        evidence: tuple[str | dict[str, str], ...] = (),
        confidence: float = 1.0,
        attributed_to: str = "user",
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        observed_at: Optional[str] = None,
        source_message_id: Optional[str] = None,
        source_message_ids: tuple[str, ...] = (),
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        relation = _require_enum(relation_type, "relation_type", _VALID_RELATION_TYPES)
        _require_enum(scope_level, "scope_level", _VALID_SCOPE_LEVELS)
        if status is not None:
            _require_enum(status, "status", _VALID_MEMORY_STATUSES)
        _require_enum(memory.memory_type, "memory_type", _VALID_MEMORY_TYPES)
        key = str(memory_key or "").strip()
        if not key:
            raise ValueError("memory_key 为空，拒绝写入长期记忆")
        if relation == "NOOP":
            return {"id": 0, "persisted": False, "relation_type": "NOOP", "memory_key": key}

        metadata_payload = dict(metadata or {})
        metadata_payload["attributed_to"] = str(attributed_to or "user")
        source_ids = list(source_message_ids)
        if source_message_id is not None and source_message_id != "" and source_message_id not in source_ids:
            source_ids.insert(0, source_message_id)
        source_ids = [item.strip() for item in _decode_text_list(source_ids, "source_message_ids_json")]
        evidence_payload = _decode_evidence(list(evidence))
        semantic_observation_clock(metadata_payload, observed_at)
        record = await asyncio.to_thread(
            self._database.append_character_memory_claim,
            character_id,
            user_scope.platform,
            user_scope.adapter,
            user_scope.sender_id,
            user_scope.conversation_type,
            user_scope.conversation_id,
            memory.memory_type,
            key,
            memory.content,
            memory.importance,
            source_message_id,
            relation_type=relation,
            scope_level=scope_level,
            status=status,
            parent_memory_id=parent_memory_id,
            supersedes_memory_id=supersedes_memory_id,
            evidence_json=json.dumps(evidence_payload, ensure_ascii=False),
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
            observed_at=observed_at,
            source_message_ids_json=json.dumps(source_ids, ensure_ascii=False),
            metadata_json=json.dumps(metadata_payload, ensure_ascii=False),
        )
        return _decode_memory_record(record)

    async def delete_memory(
        self, memory_id: int, character_id: str, user_scope: UserScope
    ) -> bool:
        return bool(
            await asyncio.to_thread(
                self._database.delete_character_memory,
                int(memory_id),
                character_id,
                user_scope.platform,
                user_scope.adapter,
                user_scope.sender_id,
                user_scope.conversation_type,
                user_scope.conversation_id,
            )
        )

    async def clear_memories(self, character_id: str, user_scope: UserScope) -> int:
        return int(
            await asyncio.to_thread(
                self._database.clear_character_memories,
                character_id,
                user_scope.platform,
                user_scope.adapter,
                user_scope.sender_id,
                user_scope.conversation_type,
                user_scope.conversation_id,
            )
        )

    async def erase_memory(
        self,
        character_id: str,
        user_scope: UserScope,
        *,
        memory_id: Optional[int] = None,
        memory_key: Optional[str] = None,
        scope_level: Optional[str] = None,
        protected_memory_keys: tuple[str, ...] = (),
    ) -> int:
        """Physically erase a claim or complete logical key version chain."""
        if scope_level is not None and scope_level not in _VALID_SCOPE_LEVELS:
            raise ValueError(f"未知的记忆作用域: {scope_level!r}")
        return int(
            await asyncio.to_thread(
                self._database.erase_character_memories,
                character_id,
                user_scope.platform,
                user_scope.adapter,
                user_scope.sender_id,
                user_scope.conversation_type,
                user_scope.conversation_id,
                memory_id=memory_id,
                memory_key=memory_key,
                scope_level=scope_level,
                **({"protected_memory_keys": protected_memory_keys} if protected_memory_keys else {}),
            )
        )


_default_repository: DatabaseCharacterMemoryRepository | None = None


def get_default_character_memory_repository() -> DatabaseCharacterMemoryRepository:
    """返回基于全局数据库适配器的默认仓储实例（进程内单例）。"""
    global _default_repository
    if _default_repository is None:
        from db.adapter import db as _db

        _default_repository = DatabaseCharacterMemoryRepository(_db)
    return _default_repository
