"""Dedicated bounded executors for authentication database work."""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

from infra.bounded_executor import BoundedThreadExecutor
from infra.environment import read_positive_int, read_positive_float

T = TypeVar("T")


auth_database_executor = BoundedThreadExecutor(
    name="auth-database",
    max_workers=read_positive_int(os.environ, "AUTH_DATABASE_WORKERS", 8),
    max_pending=read_positive_int(os.environ, "AUTH_DATABASE_MAX_PENDING", 32),
    default_timeout=read_positive_float(os.environ, "AUTH_DATABASE_TIMEOUT_SECONDS", 5.0),
)


async def run_auth_database(
    func: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    return await auth_database_executor.run(func, *args, **kwargs)
