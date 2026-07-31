from __future__ import annotations

import asyncio
import threading

import pytest

from infra.bounded_executor import (
    BlockingWorkRejected,
    BlockingWorkTimeout,
    BoundedThreadExecutor,
)


@pytest.mark.asyncio
async def test_rejects_when_running_and_pending_capacity_is_full():
    executor = BoundedThreadExecutor(
        name="test-bounded-reject",
        max_workers=1,
        max_pending=1,
        default_timeout=1.0,
    )
    started = threading.Event()
    release = threading.Event()

    def blocking_work():
        started.set()
        release.wait(timeout=1.0)
        return "done"

    task = asyncio.create_task(executor.run(blocking_work))
    try:
        assert await asyncio.to_thread(started.wait, 1.0)
        with pytest.raises(BlockingWorkRejected):
            await executor.run(lambda: "overflow")
    finally:
        release.set()
        assert await task == "done"
        executor.shutdown()


@pytest.mark.asyncio
async def test_timeout_keeps_slot_until_underlying_thread_finishes():
    executor = BoundedThreadExecutor(
        name="test-bounded-timeout",
        max_workers=1,
        max_pending=1,
        default_timeout=0.02,
    )
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_work():
        started.set()
        release.wait(timeout=1.0)
        finished.set()
        return "late"

    task = asyncio.create_task(executor.run(slow_work))
    try:
        assert await asyncio.to_thread(started.wait, 1.0)
        with pytest.raises(BlockingWorkTimeout):
            await task
        with pytest.raises(BlockingWorkRejected):
            await executor.run(lambda: "overflow")

        release.set()
        assert await asyncio.to_thread(finished.wait, 1.0)
        for _ in range(100):
            if executor.stats()["pending"] == 0:
                break
            await asyncio.sleep(0.001)

        assert await executor.run(lambda: "recovered") == "recovered"
        stats = executor.stats()
        assert stats["timed_out"] == 1

        assert stats["rejected"] == 1
    finally:
        release.set()
        executor.shutdown()

@pytest.mark.asyncio
async def test_queued_timeouts_do_not_release_slots_or_grow_physical_queue():
    executor = BoundedThreadExecutor(
        name="test-bounded-queued-timeout",
        max_workers=1,
        max_pending=2,
        default_timeout=1.0,
    )
    started = threading.Event()
    release = threading.Event()
    queued_work_ran = threading.Event()

    def blocking_work():
        started.set()
        release.wait(timeout=2.0)
        return "done"

    def queued_work():
        queued_work_ran.set()
        return "unexpected"

    running = asyncio.create_task(executor.run(blocking_work))
    try:
        assert await asyncio.to_thread(started.wait, 1.0)
        with pytest.raises(BlockingWorkTimeout):
            await executor.run(queued_work, timeout=0.02)

        assert executor.stats()["pending"] == 2
        assert executor._executor._work_queue.qsize() == 1
        for _ in range(100):
            with pytest.raises(BlockingWorkRejected):
                await executor.run(lambda: None)

        assert executor._executor._work_queue.qsize() == 1
        assert queued_work_ran.is_set() is False

        release.set()
        assert await running == "done"
        for _ in range(100):
            if executor.stats()["pending"] == 0:
                break
            await asyncio.sleep(0.001)

        assert executor.stats()["pending"] == 0
        assert executor.stats()["cancelled"] == 1
        assert queued_work_ran.is_set() is False
        assert executor._executor._work_queue.qsize() == 0
    finally:
        release.set()
        executor.shutdown()


@pytest.mark.asyncio
async def test_worker_exception_is_propagated_and_capacity_recovers():
    executor = BoundedThreadExecutor(
        name="test-bounded-failure",
        max_workers=1,
        max_pending=1,
        default_timeout=1.0,
    )

    def fail():
        raise RuntimeError("boom")

    try:
        with pytest.raises(RuntimeError, match="boom"):
            await executor.run(fail)
        assert await executor.run(lambda: 42) == 42
        assert executor.stats()["failed"] == 1
    finally:
        executor.shutdown()
