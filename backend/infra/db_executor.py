"""Shared bounded executor for synchronous DB calls from async routes."""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

from infra.bounded_executor import BoundedThreadExecutor

T = TypeVar("T")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, value)


db_executor = BoundedThreadExecutor(
    name="async-db",
    max_workers=_positive_int_env("ASYNC_DB_WORKERS", 8),
    max_pending=_positive_int_env("ASYNC_DB_MAX_PENDING", 32),
    default_timeout=float(os.getenv("ASYNC_DB_TIMEOUT_SECONDS", "30")),
)


async def run_db(
    func: Callable[..., T],
    /,
    *args: Any,
    timeout: float | None = None,
    **kwargs: Any,
) -> T:
    """Run one synchronous DB call on the shared bounded executor."""
    return await db_executor.run(func, *args, timeout=timeout, **kwargs)
