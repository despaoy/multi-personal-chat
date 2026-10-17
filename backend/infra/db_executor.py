"""Shared bounded executor for synchronous DB calls from async routes."""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

from infra.bounded_executor import BoundedThreadExecutor
from infra.environment import read_positive_int, read_positive_float

T = TypeVar("T")


db_executor = BoundedThreadExecutor(
    name="async-db",
    max_workers=read_positive_int(os.environ, "ASYNC_DB_WORKERS", 8),
    max_pending=read_positive_int(os.environ, "ASYNC_DB_MAX_PENDING", 32),
    default_timeout=read_positive_float(os.environ, "ASYNC_DB_TIMEOUT_SECONDS", 30.0),
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
