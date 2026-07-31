"""Cached, single-flight readiness checks with bounded sync offloading."""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from typing import Any, Awaitable, Callable

from infra.bounded_executor import BoundedThreadExecutor

ReadinessCheck = Callable[[], Any | Awaitable[Any]]


class ReadinessProbeTimeout(TimeoutError):
    """A caller's readiness response budget elapsed."""


class ReadinessProbe:
    """Run dependency checks once per TTL instead of once per HTTP request."""

    def __init__(
        self,
        *,
        database_check: ReadinessCheck,
        model_check: ReadinessCheck | None = None,
        model_required: bool = False,
        success_ttl: float = 5.0,
        failure_ttl: float = 1.0,
        check_timeout: float = 3.0,
        wait_timeout: float = 4.0,
        clock: Callable[[], float] = time.monotonic,
        sync_executor: BoundedThreadExecutor | None = None,
    ) -> None:
        if success_ttl < 0 or failure_ttl < 0:
            raise ValueError("readiness TTL values cannot be negative")
        if check_timeout <= 0 or wait_timeout <= 0:
            raise ValueError("readiness timeout values must be positive")

        self._database_check = database_check
        self._model_check = model_check
        self._model_required = model_required
        self._success_ttl = success_ttl
        self._failure_ttl = failure_ttl
        self._check_timeout = check_timeout
        self._wait_timeout = wait_timeout
        self._clock = clock
        self._sync_executor = sync_executor or BoundedThreadExecutor(
            name="readiness-sync",
            max_workers=2,
            max_pending=2,
            default_timeout=check_timeout,
        )
        self._owns_sync_executor = sync_executor is None
        self._cache: dict[str, Any] | None = None
        self._cache_updated_at = 0.0
        self._refresh_task: asyncio.Task[dict[str, Any]] | None = None
        self._refresh_lock = asyncio.Lock()

    async def get(self) -> dict[str, Any]:
        """Return a cached snapshot or share one in-progress refresh."""

        cached = self._fresh_cache()
        if cached is not None:
            return cached

        async with self._refresh_lock:
            cached = self._fresh_cache()
            if cached is not None:
                return cached
            if self._refresh_task is None or self._refresh_task.done():
                self._refresh_task = asyncio.create_task(
                    self._refresh(),
                    name="readiness-refresh",
                )
            task = self._refresh_task

        try:
            snapshot = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._wait_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ReadinessProbeTimeout(
                f"readiness response exceeded {self._wait_timeout:.3f}s"
            ) from exc
        return copy.deepcopy(snapshot)

    def cached(self) -> dict[str, Any] | None:
        """Return the current cache regardless of age."""

        return copy.deepcopy(self._cache)

    def shutdown(self) -> None:
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
        if self._owns_sync_executor:
            self._sync_executor.shutdown(wait=False)

    def _fresh_cache(self) -> dict[str, Any] | None:
        if self._cache is None:
            return None
        ttl = self._success_ttl if self._cache["ready"] else self._failure_ttl
        if self._clock() - self._cache_updated_at >= ttl:
            return None
        return copy.deepcopy(self._cache)

    async def _refresh(self) -> dict[str, Any]:
        database_future = self._run_check(self._database_check)
        if self._model_required:
            model_future = self._run_check(self._model_check)
        else:
            model_future = self._not_required()

        database, model = await asyncio.gather(database_future, model_future)
        deps = {
            "database": database[0],
            "model": model[0],
        }
        snapshot = {
            "ready": all(deps.values()),
            "deps": deps,
            "details": {
                "database": database[1],
                "model": model[1],
                "rag": "not_probed",
            },
        }
        self._cache = snapshot
        self._cache_updated_at = self._clock()
        return snapshot

    async def _run_check(
        self,
        check: ReadinessCheck | None,
    ) -> tuple[bool, str]:
        if check is None:
            return False, "not_configured"
        try:
            if inspect.iscoroutinefunction(check):
                result = await asyncio.wait_for(
                    check(),
                    timeout=self._check_timeout,
                )
            else:
                result = await self._sync_executor.run(
                    check,
                    timeout=self._check_timeout,
                )
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(
                        result,
                        timeout=self._check_timeout,
                    )
            healthy = result if isinstance(result, bool) else True
            return bool(healthy), "ok" if healthy else "unavailable"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return False, type(exc).__name__

    @staticmethod
    async def _not_required() -> tuple[bool, str]:
        return True, "not_required"
