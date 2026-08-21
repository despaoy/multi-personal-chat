"""Repository boundary for character relationship and long-term memory.

只做两件事：
1. 把 character.models 的不可变对象与数据库行 dict 互转；
2. 把同步数据库门面（SQLiteDB / SyncPgAdapter）包装成异步仓储接口。

隔离规则由 UserScope.memory_scope_key 语义保证：私聊按用户隔离，
群聊/频道按"会话+用户"隔离，跨角色/跨平台/跨适配器互不可见。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Protocol

from character.models import (
    MemoryItem,
    MemoryType,
    RelationshipStage,
    RelationshipState,
    UserScope,
)

_VALID_STAGES: tuple[str, ...] = ("stranger", "acquaintance", "familiar", "close")
_VALID_MEMORY_TYPES: tuple[str, ...] = (
    "user_fact",
    "shared_event",
    "promise",
    "conversation_summary",
)


class CharacterMemoryRepository(Protocol):
    """角色关系与长期记忆的持久化接口。"""

    async def get_relationship(
        self, character_id: str, user_scope: UserScope
    ) -> RelationshipState: ...

    async def get_relationship_record(
        self, character_id: str, user_scope: UserScope
    ) -> Optional[dict[str, Any]]: ...

    async def upsert_relationship(
        self, character_id: str, user_scope: UserScope, state: RelationshipState
    ) -> dict[str, Any]: ...

    async def increment_interaction(self, character_id: str, user_scope: UserScope) -> int: ...

    async def list_memories(
        self, character_id: str, user_scope: UserScope, limit: int = 30
    ) -> list[MemoryItem]: ...

    async def list_memory_records(
        self, character_id: str, user_scope: UserScope, limit: int = 30
    ) -> list[dict[str, Any]]: ...

    async def add_or_update_memory(
        self,
        character_id: str,
        user_scope: UserScope,
        memory: MemoryItem,
        *,
        memory_key: str,
        source_message_id: Optional[str] = None,
    ) -> int: ...

    async def delete_memory(
        self, memory_id: int, character_id: str, user_scope: UserScope
    ) -> bool: ...

    async def clear_memories(self, character_id: str, user_scope: UserScope) -> int: ...


def _row_to_relationship(row: Optional[dict[str, Any]]) -> RelationshipState:
    if not row:
        return RelationshipState()
    stage = row.get("relationship_stage", "stranger")
    if stage not in _VALID_STAGES:
        stage = "stranger"
    return RelationshipState(
        stage=stage,  # type: ignore[arg-type]
        preferred_address=str(row.get("preferred_address") or ""),
        summary=str(row.get("summary") or ""),
    )


class DatabaseCharacterMemoryRepository:
    """适配现有同步数据库门面的角色记忆仓储。"""

    def __init__(self, database: Any) -> None:
        self._database = database

    async def get_relationship(
        self, character_id: str, user_scope: UserScope
    ) -> RelationshipState:
        row = await self.get_relationship_record(character_id, user_scope)
        return _row_to_relationship(row)

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

    async def list_memories(
        self, character_id: str, user_scope: UserScope, limit: int = 30
    ) -> list[MemoryItem]:
        rows = await self.list_memory_records(character_id, user_scope, limit)
        items: list[MemoryItem] = []
        for row in rows:
            memory_type = row.get("memory_type", "user_fact")
            if memory_type not in _VALID_MEMORY_TYPES:
                memory_type = "user_fact"
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            items.append(
                MemoryItem(
                    memory_id=str(row.get("id", "")),
                    memory_type=memory_type,  # type: ignore[arg-type]
                    content=content,
                    importance=float(row.get("importance") or 0.0),
                )
            )
        return items

    async def list_memory_records(
        self, character_id: str, user_scope: UserScope, limit: int = 30
    ) -> list[dict[str, Any]]:
        """返回原始记忆行（含时间戳与 memory_key），供检索排序使用。"""
        rows = await asyncio.to_thread(
            self._database.list_character_memories,
            character_id,
            user_scope.platform,
            user_scope.adapter,
            user_scope.sender_id,
            user_scope.conversation_type,
            user_scope.conversation_id,
            limit,
        )
        return [dict(row) for row in rows]

    async def add_or_update_memory(
        self,
        character_id: str,
        user_scope: UserScope,
        memory: MemoryItem,
        *,
        memory_key: str,
        source_message_id: Optional[str] = None,
    ) -> int:
        if memory.memory_type not in _VALID_MEMORY_TYPES:
            raise ValueError(f"未知的记忆类型: {memory.memory_type!r}")
        key = memory_key.strip()
        if not key:
            raise ValueError("memory_key 为空，拒绝写入长期记忆")
        record = await asyncio.to_thread(
            self._database.add_or_update_character_memory,
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
        )
        return int(record.get("id") or 0)

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


_default_repository: DatabaseCharacterMemoryRepository | None = None


def get_default_character_memory_repository() -> DatabaseCharacterMemoryRepository:
    """返回基于全局数据库适配器的默认仓储实例（进程内单例）。"""
    global _default_repository
    if _default_repository is None:
        from db.adapter import db as _db

        _default_repository = DatabaseCharacterMemoryRepository(_db)
    return _default_repository
