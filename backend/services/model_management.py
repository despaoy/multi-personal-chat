"""Transport-independent model administration boundary."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class ModelManagementService(Protocol):
    """Operations required by the model-management HTTP API."""

    async def list_available_models(self) -> list[dict[str, Any]]: ...

    async def check_model_exists(self, model_name: str) -> bool: ...

    async def download_model(self, model_name: str, *, force: bool = False) -> Any: ...

    async def delete_model(self, model_name: str) -> bool: ...


class ModelManagerService:
    """Adapt the synchronous model manager to an asynchronous use-case API.

    Model discovery, downloads, and file deletion may touch disk or the
    network. Running them in worker threads keeps FastAPI's event loop
    responsive without coupling routes to the concrete singleton manager.
    """

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    async def list_available_models(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._manager.list_available_models)

    async def check_model_exists(self, model_name: str) -> bool:
        return await asyncio.to_thread(self._manager.check_model_exists, model_name)

    async def download_model(self, model_name: str, *, force: bool = False) -> Any:
        return await asyncio.to_thread(
            self._manager.download_model_from_hf,
            model_name=model_name,
            force=force,
        )

    async def delete_model(self, model_name: str) -> bool:
        return await asyncio.to_thread(self._manager.delete_model, model_name)
