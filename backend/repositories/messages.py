"""Repository boundary for message history and session administration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from character.models import UserScope


@dataclass(frozen=True, slots=True)
class MessageQuery:
    """Database-neutral filters accepted by the message history use case."""

    search: str | None = None
    session_type: str | None = None
    lora_name: str | None = None
    session_id: str | None = None
    session_name: str | None = None
    platform: str | None = None


@dataclass(frozen=True, slots=True)
class MessagePage:
    """One page of message history plus filtered and global totals."""

    messages: list[dict[str, Any]]
    total: int
    total_all: int


class MessageRepository(Protocol):
    """Persistence operations required by the message API."""

    async def list_page(self, query: MessageQuery, *, limit: int, offset: int) -> MessagePage: ...

    async def list_session_summaries(self) -> list[dict[str, Any]]: ...

    async def set_session_bot_enabled(
        self,
        session_id: str,
        enabled: bool,
        *,
        platform: str,
        conversation_id: str | None,
        conversation_type: str,
    ) -> None: ...

    async def list_recent_conversation_history(
        self,
        user_scope: UserScope,
        *,
        limit: int = 8,
        max_chars: int = 6000,
    ) -> list[dict[str, str]]: ...

    async def delete_filtered(self, query: MessageQuery) -> int: ...

    async def delete(self, message_id: int) -> bool: ...


class DatabaseMessageRepository:
    """Adapt the existing synchronous database facade to message use cases."""

    def __init__(self, database: Any) -> None:
        self._database = database

    async def list_page(self, query: MessageQuery, *, limit: int, offset: int) -> MessagePage:
        def load_page() -> MessagePage:
            filters = self._query_filters(query)
            return MessagePage(
                total_all=int(self._database.get_message_count()),
                messages=list(
                    self._database.get_messages_filtered(
                        **filters,
                        limit=limit,
                        offset=offset,
                    )
                ),
                total=int(self._database.get_message_count_filtered(**filters)),
            )

        # The sync PostgreSQL facade and SQLite both block the event loop.
        # Keeping all three reads in one worker also preserves SQLite's
        # thread-local connection behavior.
        return await asyncio.to_thread(load_page)

    async def list_session_summaries(self) -> list[dict[str, Any]]:
        return list(await asyncio.to_thread(self._database.get_session_summaries))

    async def list_recent_conversation_history(
        self,
        user_scope: UserScope,
        *,
        limit: int = 8,
        max_chars: int = 6000,
    ) -> list[dict[str, str]]:
        """按用户范围读取最近对话历史（时间正序，超预算从最旧一侧截断）。

        私聊读取该用户全部私聊记录；群聊/频道只读取该用户在该会话内的
        记录，与长期记忆的隔离范围保持一致。
        """
        if not hasattr(self._database, "list_conversation_history"):
            return []
        return list(
            await asyncio.to_thread(
                self._database.list_conversation_history,
                user_scope.platform,
                user_scope.adapter,
                user_scope.sender_id,
                user_scope.conversation_type,
                user_scope.conversation_id,
                limit,
                max_chars,
            )
        )

    async def set_session_bot_enabled(
        self,
        session_id: str,
        enabled: bool,
        *,
        platform: str,
        conversation_id: str | None,
        conversation_type: str,
    ) -> None:
        await asyncio.to_thread(
            self._database.set_session_bot_enabled,
            session_id,
            enabled,
            platform,
            conversation_id,
            conversation_type,
        )

    async def delete_filtered(self, query: MessageQuery) -> int:
        return int(
            await asyncio.to_thread(
                self._database.delete_messages_by_filter,
                search=query.search,
                sessionType=query.session_type,
                lora=query.lora_name,
                sessionName=query.session_name,
                platform=query.platform,
            )
        )

    async def delete(self, message_id: int) -> bool:
        return bool(await asyncio.to_thread(self._database.delete_message, message_id))

    @staticmethod
    def _query_filters(query: MessageQuery) -> dict[str, str | None]:
        return {
            "search": query.search,
            "session_type": query.session_type,
            "lora_name": query.lora_name,
            "session_id": query.session_id,
            "session_name": query.session_name,
            "platform": query.platform,
        }
