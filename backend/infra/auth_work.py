"""Dedicated bounded executors for authentication database work."""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

from infra.bounded_executor import BoundedThreadExecutor

T = TypeVar("T")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


auth_database_executor = BoundedThreadExecutor(
    name="auth-database",
    max_workers=_positive_int_env("AUTH_DATABASE_WORKERS", 8),
    max_pending=_positive_int_env("AUTH_DATABASE_MAX_PENDING", 32),
    default_timeout=_positive_float_env("AUTH_DATABASE_TIMEOUT_SECONDS", 5.0),
)


async def run_auth_database(
    func: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    return await auth_database_executor.run(func, *args, **kwargs)
