"""Repository boundary for per-user page data."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class UserDataUserNotFoundError(LookupError):
    """Raised when authenticated identity has no persisted user record."""


class UserDataRepository(Protocol):
    """Persistence operations required by the user-data API."""

    async def load(self, username: str, page_key: str | None = None) -> dict[str, Any] | None: ...

    async def save(self, username: str, page_key: str, data_json: str) -> None: ...


class DatabaseUserDataRepository:
    """Adapt the existing synchronous database facade to user-data use cases."""

    def __init__(self, database: Any) -> None:
        self._database = database

    async def load(self, username: str, page_key: str | None = None) -> dict[str, Any] | None:
        def operation() -> dict[str, Any] | None:
            user_id = self._resolve_user_id(username)
            return self._database.get_user_data(user_id, page_key)

        return await asyncio.to_thread(operation)

    async def save(self, username: str, page_key: str, data_json: str) -> None:
        def operation() -> None:
            user_id = self._resolve_user_id(username)
            self._database.save_user_data(user_id, page_key, data_json)

        await asyncio.to_thread(operation)

    def _resolve_user_id(self, username: str) -> int:
        user = self._database.get_user_by_username(username)
        if not user:
            raise UserDataUserNotFoundError(username)
        return int(user["id"])
