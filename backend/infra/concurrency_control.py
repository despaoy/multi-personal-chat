"""Unified admission control and in-process inference scheduling."""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

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
        self.rate = max(float(rate), 0.0)
        self.capacity = max(int(capacity), 1)
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
    enqueued_at: float = field(compare=False)
    session_id: str = field(compare=False)
    factory: Callable[[], Awaitable[Any]] = field(compare=False)
    future: asyncio.Future = field(compare=False)
    running_task: asyncio.Task | None = field(default=None, compare=False)


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    last_used: float = field(default_factory=time.monotonic)


class InferenceRuntime:
    """Single-process bounded priority scheduler shared by every chat entrypoint."""

    def __init__(self) -> None:
        self.max_queue_size = int(os.getenv("INFERENCE_QUEUE_MAX_SIZE", "100"))
        legacy_workers = os.getenv("MODEL_MAX_CONCURRENCY")
        worker_value = os.getenv("INFERENCE_WORKERS") or os.getenv("LLM_MAX_CONCURRENCY") or legacy_workers or "2"
        if legacy_workers and not os.getenv("INFERENCE_WORKERS") and not os.getenv("LLM_MAX_CONCURRENCY"):
            logger.warning("MODEL_MAX_CONCURRENCY 已弃用，请改用 INFERENCE_WORKERS")
        self.worker_count = max(1, int(worker_value))
        self.queue_timeout = float(os.getenv("INFERENCE_QUEUE_TIMEOUT", "180"))
        self._queue: asyncio.PriorityQueue[_QueuedInference] = asyncio.PriorityQueue(maxsize=self.max_queue_size)
        self._sequence = itertools.count()
        self._workers_started = False
        self._workers: list[asyncio.Task] = []
        self._maintenance_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._session_locks: dict[str, _SessionLockEntry] = {}
        self._session_locks_guard = asyncio.Lock()
        self._active_count = 0
        self._active_guard = asyncio.Lock()
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
        await self._ensure_workers()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _QueuedInference(
            priority=priority,
            sequence=next(self._sequence),
            enqueued_at=time.monotonic(),
            session_id=session_id or "default",
            factory=factory,
            future=future,
        )
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            self._stats["rejected"] += 1
            raise InferenceQueueFull("inference queue is full") from exc

        self._stats["submitted"] += 1
        try:
            return await asyncio.wait_for(future, timeout=timeout or self.queue_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._stats["cancelled"] += 1
            if not future.done():
                future.cancel()
            if item.running_task is not None and not item.running_task.done():
                item.running_task.cancel()
            raise

    async def _ensure_workers(self) -> None:
        if self._workers_started:
            return
        async with self._start_lock:
            if self._workers_started:
                return
            self._workers = [
                asyncio.create_task(self._worker(idx), name=f"inference-worker-{idx}")
                for idx in range(self.worker_count)
            ]
            self._maintenance_task = asyncio.create_task(
                self._maintenance_loop(), name="inference-runtime-maintenance"
            )
            self._workers_started = True

    async def _reserve_session_lock(self, session_id: str) -> _SessionLockEntry:
        async with self._session_locks_guard:
            entry = self._session_locks.get(session_id)
            if entry is None:
                entry = _SessionLockEntry()
                self._session_locks[session_id] = entry
            entry.users += 1
            entry.last_used = time.monotonic()
            return entry

    async def _release_session_lock(self, entry: _SessionLockEntry) -> None:
        async with self._session_locks_guard:
            entry.users = max(0, entry.users - 1)
            entry.last_used = time.monotonic()

    async def _worker(self, worker_id: int) -> None:
        while True:
            item = await self._queue.get()
            entry: _SessionLockEntry | None = None
            try:
                if item.future.cancelled():
                    continue
                entry = await self._reserve_session_lock(item.session_id)
                async with entry.lock:
                    if item.future.cancelled():
                        continue
                    async with self._active_guard:
                        if item.future.cancelled():
                            continue
                        # No await between the final cancellation check and
                        # publishing running_task: submit() can always cancel
                        # work that has actually started.
                        item.running_task = asyncio.create_task(item.factory())
                        self._active_count += 1
                    try:
                        result = await item.running_task
                    except asyncio.CancelledError:
                        # A timed-out caller cancels only its model task; the worker survives.
                        if asyncio.current_task().cancelling():
                            raise
                        if not item.future.done():
                            item.future.cancel()
                        continue
                    finally:
                        item.running_task = None
                        async with self._active_guard:
                            self._active_count = max(0, self._active_count - 1)
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
                if entry is not None:
                    await self._release_session_lock(entry)
                self._queue.task_done()

    async def _maintenance_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(300)
                await self.cleanup()
        except asyncio.CancelledError:
            raise

    async def cleanup(self, max_age: float = 3600.0) -> dict[str, int]:
        removed = {
            "global_buckets": await self.global_limiter.cleanup(max_age),
            "conversation_buckets": await self.conversation_limiter.cleanup(max_age),
            "sender_buckets": await self.sender_limiter.cleanup(max_age),
            "session_locks": 0,
        }
        cutoff = time.monotonic() - max_age
        async with self._session_locks_guard:
            stale = [
                key for key, entry in self._session_locks.items()
                if entry.users == 0 and not entry.lock.locked() and entry.last_used < cutoff
            ]
            for key in stale:
                self._session_locks.pop(key, None)
            removed["session_locks"] = len(stale)
        return removed

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

            while True:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not item.future.done():
                    item.future.cancel()
                self._queue.task_done()

            self._workers.clear()
            self._maintenance_task = None
            self._workers_started = False
            async with self._session_locks_guard:
                self._session_locks.clear()

            # asyncio queues and locks retain their owning event loop after
            # contention. Rebuild all loop-bound state so application/test
            # lifecycles can safely reuse this runtime on a fresh loop.
            self._queue = asyncio.PriorityQueue(maxsize=self.max_queue_size)
            self._session_locks = {}
            self._session_locks_guard = asyncio.Lock()
            self._active_guard = asyncio.Lock()
            self._rate_limit_guard = asyncio.Lock()
            self._active_count = 0
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
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "workers": self.worker_count,
            "active": self._active_count,
            "session_locks": len(self._session_locks),
            "rate_limits": self.rate_limit_stats(),
        }


inference_runtime = InferenceRuntime()
