"""Bounded thread offloading for small, blocking workloads."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class BlockingWorkRejected(RuntimeError):
    """No bounded executor slot is available."""


class BlockingWorkTimeout(TimeoutError):
    """The caller stopped waiting for blocking work."""


class _WorkState:
    __slots__ = ("cancel_requested", "skipped")

    def __init__(self) -> None:
        self.cancel_requested = threading.Event()
        self.skipped = False

class BoundedThreadExecutor:
    """A dedicated thread pool with a hard running-plus-queued limit."""

    def __init__(
        self,
        *,
        name: str,
        max_workers: int,
        max_pending: int,
        default_timeout: float,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_pending < max_workers:
            raise ValueError("max_pending must be at least max_workers")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be positive")

        self.name = name
        self.max_workers = max_workers
        self.max_pending = max_pending
        self.default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=name,
        )
        self._slots = threading.BoundedSemaphore(max_pending)
        self._state_lock = threading.Lock()
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "timed_out": 0,
            "rejected": 0,
            "pending": 0,
        }

    async def run(
        self,
        func: Callable[..., T],
        /,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Run one function without allowing an unbounded executor queue."""

        wait_timeout = self.default_timeout if timeout is None else timeout
        if wait_timeout <= 0:
            raise ValueError("timeout must be positive")

        if not self._slots.acquire(blocking=False):
            self._increment("rejected")
            raise BlockingWorkRejected(f"{self.name} queue is full")

        state = _WorkState()
        work = partial(self._run_work, state, func, *args, **kwargs)
        try:
            future = self._executor.submit(work)
        except BaseException:
            self._slots.release()
            raise

        with self._state_lock:
            self._stats["submitted"] += 1
            self._stats["pending"] += 1
        future.add_done_callback(partial(self._work_finished, state=state))

        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(self._consume_future_exception)
        try:
            done, _ = await asyncio.wait({wrapped}, timeout=wait_timeout)
        except asyncio.CancelledError:
            state.cancel_requested.set()
            raise

        if not done:
            state.cancel_requested.set()
            self._increment("timed_out")
            raise BlockingWorkTimeout(
                f"{self.name} work exceeded {wait_timeout:.3f}s"
            )
        return await wrapped

    def stats(self) -> dict[str, int | str]:
        with self._state_lock:
            return {"name": self.name, **self._stats}

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _increment(self, key: str) -> None:
        with self._state_lock:
            self._stats[key] += 1

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()


    @staticmethod
    def _run_work(
        state: _WorkState,
        func: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T | None:
        if state.cancel_requested.is_set():
            state.skipped = True
            return None
        return func(*args, **kwargs)

    def _work_finished(self, future: Future[Any], *, state: _WorkState) -> None:
        with self._state_lock:
            self._stats["pending"] = max(0, self._stats["pending"] - 1)
            if future.cancelled() or state.skipped:
                self._stats["cancelled"] += 1
            elif future.exception() is not None:
                self._stats["failed"] += 1
            else:
                self._stats["completed"] += 1
        self._slots.release()
