"""Strict three-state circuit breakers: reject unavailable work explicitly."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Any

from interfaces import CircuitBreakerInterface

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """The circuit has rejected a request before its operation starts."""


class CircuitBreakerStats:
    def __init__(self) -> None:
        self.total_calls = 0
        self.success_calls = 0
        self.failure_calls = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None
        self.last_state_change_time = time.monotonic()

    def record_success(self) -> None:
        self.total_calls += 1
        self.success_calls += 1
        self.last_success_time = time.monotonic()

    def record_failure(self) -> None:
        self.total_calls += 1
        self.failure_calls += 1
        self.last_failure_time = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


class CircuitBreaker:
    """Reject OPEN work and bound concurrent HALF_OPEN recovery probes.

    Successful operations return their actual result; failed operations keep
    their original exception. State callbacks are observational, so callback
    failures are logged without replacing an operation's outcome.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        on_open: Callable[[CircuitBreaker], None] | None = None,
        on_close: Callable[[CircuitBreaker], None] | None = None,
        on_half_open: Callable[[CircuitBreaker], None] | None = None,
    ) -> None:
        if type(failure_threshold) is not int or failure_threshold < 1:
            raise ValueError("failure_threshold must be a positive integer")
        if type(half_open_max_calls) is not int or half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be a positive integer")
        if isinstance(recovery_timeout, bool) or not isfinite(recovery_timeout) or recovery_timeout < 0:
            raise ValueError("recovery_timeout must be finite and non-negative")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.on_open = on_open
        self.on_close = on_close
        self.on_half_open = on_half_open
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._generation = 0
        self._last_failure_time: float | None = None
        self._stats = CircuitBreakerStats()
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    def _transition(self, state: CircuitState) -> None:
        previous = self._state
        self._state = state
        # New OPEN/HALF_OPEN cycles invalidate earlier admissions. A recovery
        # success keeps its peer probes valid until they finish; reset below
        # separately invalidates outstanding work even when already CLOSED.
        if state != CircuitState.CLOSED:
            self._generation += 1
        self._half_open_calls = 0
        if state == CircuitState.OPEN:
            self._last_failure_time = time.monotonic()
        elif state == CircuitState.CLOSED:
            self._failure_count = 0
            self._last_failure_time = None
        if previous == state:
            return
        self._stats.last_state_change_time = time.monotonic()
        logger.info("Circuit [%s] %s -> %s", self.name, previous.value, state.value)
        callback = {
            CircuitState.OPEN: self.on_open,
            CircuitState.CLOSED: self.on_close,
            CircuitState.HALF_OPEN: self.on_half_open,
        }[state]
        if callback is not None:
            try:
                callback(self)
            except Exception as exc:
                logger.error("Circuit [%s] state callback failed: %s", self.name, type(exc).__name__)

    async def _record(self, success: bool, generation: int | None = None, *, probe: bool = False) -> None:
        async with self._get_lock():
            if success:
                self._stats.record_success()
            else:
                self._stats.record_failure()
            if generation is not None and generation != self._generation:
                return
            if success:
                if self._state == CircuitState.HALF_OPEN:
                    self._transition(CircuitState.CLOSED)
                elif self._state == CircuitState.CLOSED and not probe:
                    self._failure_count = 0
            else:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if probe or self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
                    self._transition(CircuitState.OPEN)

    async def record_success(self) -> None:
        """Record completed work admitted by a caller-owned streaming path."""
        await self._record(True)

    async def record_failure(self, error: Any = None) -> None:
        await self._record(False)
        if error is not None:
            logger.error("Circuit [%s] operation failed: %s", self.name, error)

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[CircuitBreaker]:
        """Own one admission, its outcome and cancellation cleanup."""
        async with self._get_lock():
            state = self.state
            if state == CircuitState.OPEN:
                raise CircuitOpenError(f"Circuit [{self.name}] is open")
            probe = state == CircuitState.HALF_OPEN
            if probe and self._half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(f"Circuit [{self.name}] recovery probes are at capacity")
            if probe:
                self._half_open_calls += 1
            generation = self._generation
        try:
            yield self
        except Exception as exc:
            await self._record(False, generation, probe=probe)
            logger.error("Circuit [%s] operation failed: %s", self.name, type(exc).__name__)
            raise
        else:
            await self._record(True, generation, probe=probe)
        finally:
            if probe:
                async with self._get_lock():
                    if generation == self._generation and self._state == CircuitState.HALF_OPEN:
                        self._half_open_calls -= 1

    async def call(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        async with self():
            return await func(*args, **kwargs)

    def protect(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.call(func, *args, **kwargs)
        return wrapper

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "current_failure_count": self._failure_count,
            "half_open_calls": self._half_open_calls,
            **self._stats.to_dict(),
        }

    async def reset(self) -> None:
        async with self._get_lock():
            self._generation += 1
            self._transition(CircuitState.CLOSED)
            self._stats = CircuitBreakerStats()

    async def trip(self) -> None:
        async with self._get_lock():
            self._transition(CircuitState.OPEN)


class CircuitBreakerRegistry:
    """Share named circuit instances with generation and monitoring."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        on_open: Callable[[CircuitBreaker], None] | None = None,
        on_close: Callable[[CircuitBreaker], None] | None = None,
        on_half_open: Callable[[CircuitBreaker], None] | None = None,
    ) -> CircuitBreaker:
        async with self._get_lock():
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    half_open_max_calls=half_open_max_calls,
                    on_open=on_open,
                    on_close=on_close,
                    on_half_open=on_half_open,
                )
            return self._breakers[name]

    async def get(self, name: str) -> CircuitBreaker | None:
        async with self._get_lock():
            return self._breakers.get(name)

    async def remove(self, name: str) -> bool:
        async with self._get_lock():
            return self._breakers.pop(name, None) is not None

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    async def reset_all(self) -> None:
        async with self._get_lock():
            breakers = tuple(self._breakers.values())
        for breaker in breakers:
            await breaker.reset()

    async def clear(self) -> None:
        async with self._get_lock():
            breakers = tuple(self._breakers.values())
            self._breakers.clear()
        for breaker in breakers:
            await breaker.reset()

    @property
    def names(self) -> list[str]:
        return list(self._breakers)


global_registry = CircuitBreakerRegistry()
assert isinstance(CircuitBreaker(), CircuitBreakerInterface), "CircuitBreaker must implement CircuitBreakerInterface"
