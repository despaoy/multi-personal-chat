"""Unified admission control and in-process inference scheduling."""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from inference.request_context import detached_request_context
from infra.environment import read_positive_float, read_positive_int

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    def __init__(self, scope: str, retry_after: float = 1.0):
        super().__init__(scope)
        self.scope = scope
        self.retry_after = retry_after


class InferenceQueueFull(Exception):
    pass


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    def __init__(self, rate: float, capacity: int):
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("rate must be nonnegative and finite")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self.reset()

    def reset(self) -> None:
        """Reset inactive lifecycle state while retaining this limiter's configuration."""
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._allowed = 0
        self._rejected = 0

    async def acquire(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.capacity), updated_at=now)
                self._buckets[key] = bucket

            elapsed = max(0.0, now - bucket.updated_at)
            bucket.updated_at = now
            bucket.tokens = min(float(self.capacity), bucket.tokens + elapsed * self.rate)

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                self._allowed += 1
                return True, 0.0

            self._rejected += 1
            if self.rate <= 0:
                return False, 60.0
            return False, max(0.0, (cost - bucket.tokens) / self.rate)

    async def refund(self, key: str, cost: float = 1.0) -> None:
        """Refund a token when a later scope rejects the same composite check."""
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is not None:
                bucket.tokens = min(float(self.capacity), bucket.tokens + cost)
                self._allowed = max(0, self._allowed - 1)

    async def cleanup(self, max_age: float = 3600.0) -> int:
        cutoff = time.monotonic() - max_age
        async with self._lock:
            stale = [key for key, bucket in self._buckets.items() if bucket.updated_at < cutoff]
            for key in stale:
                self._buckets.pop(key, None)
            return len(stale)

    def stats(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "capacity": self.capacity,
            "active_buckets": len(self._buckets),
            "allowed": self._allowed,
            "rejected": self._rejected,
        }


@dataclass(order=True)
class _QueuedInference:
    priority: int
    sequence: int
    session_id: str = field(compare=False)
    factory: Callable[[], Awaitable[Any]] = field(compare=False)
    future: asyncio.Future = field(compare=False)
    running_task: asyncio.Task | None = field(default=None, compare=False)


class InferenceRuntime:
    """Single-process bounded priority scheduler shared by every chat entrypoint."""

    def __init__(self) -> None:
        self.max_queue_size = read_positive_int(os.environ, "INFERENCE_QUEUE_MAX_SIZE", 100)
        worker_key = next(
            (key for key in ("INFERENCE_WORKERS", "LLM_MAX_CONCURRENCY", "MODEL_MAX_CONCURRENCY") if key in os.environ),
            "INFERENCE_WORKERS",
        )
        if worker_key == "MODEL_MAX_CONCURRENCY":
            logger.warning("MODEL_MAX_CONCURRENCY 已弃用，请改用 INFERENCE_WORKERS")
        self.worker_count = read_positive_int(os.environ, worker_key, 2)
        self.queue_timeout = read_positive_float(os.environ, "INFERENCE_QUEUE_TIMEOUT", 180.0)
        self._pending: list[_QueuedInference] = []
        self._ready = asyncio.Condition()
        self._active_sessions: set[str] = set()
        self._sequence = itertools.count()
        self._workers_started = False
        self._workers: list[asyncio.Task] = []
        self._maintenance_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._rate_limit_guard = asyncio.Lock()
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "cancelled": 0,
        }

        self.global_limiter = TokenBucketLimiter(
            float(os.getenv("CHAT_GLOBAL_QPS", "20")),
            int(os.getenv("CHAT_GLOBAL_BURST", "40")),
        )
        self.conversation_limiter = TokenBucketLimiter(
            float(os.getenv("CHAT_CONVERSATION_QPS", "1")),
            int(os.getenv("CHAT_CONVERSATION_BURST", "5")),
        )
        self.sender_limiter = TokenBucketLimiter(
            float(os.getenv("CHAT_SENDER_QPS", "0.5")),
            int(os.getenv("CHAT_SENDER_BURST", "3")),
        )

    async def check_rate_limits(self, platform: str, conversation_id: str, sender_id: str) -> None:
        checks = [
            ("global", self.global_limiter, "global"),
            ("conversation", self.conversation_limiter, f"{platform}:{conversation_id}"),
        ]
        if sender_id:
            checks.append(("sender", self.sender_limiter, f"{platform}:{sender_id}"))

        async with self._rate_limit_guard:
            acquired: list[tuple[TokenBucketLimiter, str]] = []
            for scope, limiter, key in checks:
                allowed, retry_after = await limiter.acquire(key)
                if allowed:
                    acquired.append((limiter, key))
                    continue
                for acquired_limiter, acquired_key in acquired:
                    await acquired_limiter.refund(acquired_key)
                raise RateLimitExceeded(scope, retry_after)

    async def submit(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        session_id: str,
        priority: int,
        timeout: float | None = None,
    ) -> Any:
        wait_timeout = self.queue_timeout if timeout is None else timeout
        if not math.isfinite(wait_timeout) or wait_timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        await self._ensure_workers()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _QueuedInference(
            priority=priority,
            sequence=next(self._sequence),
            session_id=session_id or "default",
            factory=factory,
            future=future,
        )
        async with self._ready:
            if len(self._pending) >= self.max_queue_size:
                self._stats["rejected"] += 1
                raise InferenceQueueFull("inference queue is full")
            self._pending.append(item)
            self._stats["submitted"] += 1
            self._ready.notify_all()
        try:
            return await asyncio.wait_for(future, timeout=wait_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            # A job TimeoutError leaves a completed future, not a cancelled wait.
            if isinstance(exc, TimeoutError) and not future.cancelled():
                raise
            self._stats["cancelled"] += 1
            if not future.done():
                future.cancel()
            async with self._ready:
                if item in self._pending:
                    self._pending.remove(item)
                    self._ready.notify_all()
            if item.running_task is not None and not item.running_task.done():
                running_task = item.running_task
                running_task.cancel()
                # Do not release a gateway retry lease while the previous
                # generation coroutine is still unwinding its cancellation.
                await asyncio.gather(running_task, return_exceptions=True)
            raise

    async def _ensure_workers(self) -> None:
        if self._workers_started:
            return
        async with self._start_lock:
            if self._workers_started:
                return
            self._workers = [
                asyncio.create_task(
                    self._worker(idx), name=f"inference-worker-{idx}", context=detached_request_context()
                )
                for idx in range(self.worker_count)
            ]
            self._maintenance_task = asyncio.create_task(
                self._maintenance_loop(), name="inference-runtime-maintenance", context=detached_request_context()
            )
            self._workers_started = True

    async def _take_ready(self) -> _QueuedInference:
        async with self._ready:
            while True:
                item = min(
                    (
                        item
                        for item in self._pending
                        if item.session_id not in self._active_sessions and not item.future.cancelled()
                    ),
                    default=None,
                )
                if item is not None:
                    self._pending.remove(item)
                    self._active_sessions.add(item.session_id)
                    return item
                await self._ready.wait()

    async def _worker(self, worker_id: int) -> None:
        while True:
            item = await self._take_ready()
            try:
                # Selection and publishing the task have no suspension point;
                # submit() can always cancel work that has actually started.
                item.running_task = asyncio.create_task(item.factory())
                try:
                    result = await item.running_task
                except asyncio.CancelledError:
                    # A caller cancels its job; shutdown cancels the worker too.
                    if asyncio.current_task().cancelling():
                        raise
                    if not item.future.done():
                        item.future.cancel()
                    continue
                if not item.future.done():
                    item.future.set_result(result)
                    self._stats["completed"] += 1
            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                raise
            except Exception as exc:
                self._stats["failed"] += 1
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                item.running_task = None
                async with self._ready:
                    self._active_sessions.remove(item.session_id)
                    self._ready.notify_all()

    async def _maintenance_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(300)
                await self.cleanup()
        except asyncio.CancelledError:
            raise

    async def cleanup(self, max_age: float = 3600.0) -> dict[str, int]:
        return {
            "global_buckets": await self.global_limiter.cleanup(max_age),
            "conversation_buckets": await self.conversation_limiter.cleanup(max_age),
            "sender_buckets": await self.sender_limiter.cleanup(max_age),
        }

    async def shutdown(self) -> None:
        """Cancel workers, release queued callers, and make the scheduler restartable."""
        async with self._start_lock:
            tasks = [*self._workers]
            if self._maintenance_task is not None:
                tasks.append(self._maintenance_task)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            async with self._ready:
                for item in self._pending:
                    if not item.future.done():
                        item.future.cancel()
                self._pending.clear()

            self._workers.clear()
            self._maintenance_task = None
            self._workers_started = False

            # Rebuild loop-bound state so a fresh lifecycle can use a new loop.
            self._ready = asyncio.Condition()
            self._active_sessions.clear()
            self._rate_limit_guard = asyncio.Lock()
            for limiter in (self.global_limiter, self.conversation_limiter, self.sender_limiter):
                limiter.reset()

        # Replace this last: no await occurs after releasing the old lock, so
        # a new lifecycle cannot observe partially reset state on this loop.
        self._start_lock = asyncio.Lock()

    def priority_for(self, source: str, conversation_type: str) -> int:
        if source == "admin":
            return 0
        if conversation_type == "private":
            return 10
        if conversation_type == "channel":
            return 40
        return 50

    def rate_limit_stats(self) -> dict[str, Any]:
        return {
            "global": self.global_limiter.stats(),
            "conversation": self.conversation_limiter.stats(),
            "sender": self.sender_limiter.stats(),
        }

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "queue_size": len(self._pending),
            "max_queue_size": self.max_queue_size,
            "workers": self.worker_count,
            "active": len(self._active_sessions),
            "rate_limits": self.rate_limit_stats(),
        }


inference_runtime = InferenceRuntime()
