"""Runnable owners use worker capacity without overlapping one owner's history."""

import asyncio
from unittest.mock import Mock

import pytest

from infra.concurrency_control import InferenceQueueFull, InferenceRuntime


def runtime_for(monkeypatch, capacity=8):
    monkeypatch.setenv("INFERENCE_WORKERS", "2")
    monkeypatch.setenv("INFERENCE_QUEUE_MAX_SIZE", str(capacity))
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT", "5")
    return InferenceRuntime()


async def queued(runtime, size):
    while runtime.stats()["queue_size"] != size:
        await asyncio.sleep(0)


async def test_independent_owner_runs_while_earlier_owner_waits_and_history_stays_ordered(monkeypatch):
    runtime = runtime_for(monkeypatch)
    first_started, second_started, other_started, release_first, release_other = (asyncio.Event() for _ in range(5))
    history = []

    async def first():
        first_started.set()
        await release_first.wait()
        history.extend(["合成原话：尚未付款，仅周五执行，出差顺延。", "合成回复：已记录完整条件。"])
        return "first"

    async def second():
        second_started.set()
        return list(history)

    async def other():
        other_started.set()
        await release_other.wait()
        return "independent"

    tasks = [asyncio.create_task(runtime.submit(first, session_id="owner-a", priority=10))]
    try:
        await asyncio.wait_for(first_started.wait(), 1)
        tasks.append(asyncio.create_task(runtime.submit(second, session_id="owner-a", priority=10)))
        tasks.append(asyncio.create_task(runtime.submit(other, session_id="owner-b", priority=10)))
        await asyncio.wait_for(other_started.wait(), 1)
        assert not second_started.is_set()
        assert runtime.stats()["active"] == 2 and runtime.stats()["queue_size"] == 1
        release_other.set()
        assert await tasks[2] == "independent"
        assert not release_first.is_set() and not second_started.is_set()
        release_first.set()
        assert await tasks[0] == "first"
        assert await tasks[1] == ["合成原话：尚未付款，仅周五执行，出差顺延。", "合成回复：已记录完整条件。"]
    finally:
        release_first.set()
        release_other.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.shutdown()
    assert runtime.stats()["active"] == runtime.stats()["queue_size"] == 0


async def test_ready_priority_and_equal_priority_fifo_skip_a_busy_owner(monkeypatch):
    runtime = runtime_for(monkeypatch, capacity=5)
    started_a, started_x, release_a, release_x = (asyncio.Event() for _ in range(4))
    order = []

    async def block(started, release):
        started.set()
        await release.wait()

    async def record(label):
        order.append(label)
        return label

    tasks = [
        asyncio.create_task(runtime.submit(lambda: block(started_a, release_a), session_id="a", priority=10)),
        asyncio.create_task(runtime.submit(lambda: block(started_x, release_x), session_id="x", priority=10)),
    ]
    try:
        await asyncio.wait_for(asyncio.gather(started_a.wait(), started_x.wait()), 1)
        for label, scope, priority in [
            ("a2", "a", 0),
            ("low", "b", 50),
            ("high-first", "c", 10),
            ("high-second", "d", 10),
            ("a3", "a", 0),
        ]:
            tasks.append(
                asyncio.create_task(
                    runtime.submit(lambda value=label: record(value), session_id=scope, priority=priority)
                )
            )
        await asyncio.wait_for(queued(runtime, 5), 1)
        release_x.set()
        assert await asyncio.wait_for(asyncio.gather(tasks[3], tasks[4], tasks[5]), 1) == [
            "low",
            "high-first",
            "high-second",
        ]
        assert order == ["high-first", "high-second", "low"]
        assert runtime.stats()["active"] == 1 and runtime.stats()["queue_size"] == 2
        release_a.set()
        await asyncio.gather(*tasks)
        assert order == ["high-first", "high-second", "low", "a2", "a3"]
    finally:
        release_a.set()
        release_x.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.shutdown()


async def test_cancelled_busy_owner_wait_returns_bounded_capacity_without_running_it(monkeypatch):
    runtime = runtime_for(monkeypatch, capacity=1)
    started, release = asyncio.Event(), asyncio.Event()
    forbidden = Mock(side_effect=AssertionError("cancelled/overflow work must not start"))

    async def first():
        started.set()
        await release.wait()
        return "running"

    tasks = [asyncio.create_task(runtime.submit(first, session_id="a", priority=10))]
    try:
        await asyncio.wait_for(started.wait(), 1)
        tasks.append(asyncio.create_task(runtime.submit(forbidden, session_id="a", priority=10)))
        await asyncio.wait_for(queued(runtime, 1), 1)
        with pytest.raises(InferenceQueueFull):
            await runtime.submit(forbidden, session_id="b", priority=10)
        tasks[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[1]
        assert runtime.stats()["queue_size"] == 0 and runtime.stats()["active"] == 1
        assert (
            await runtime.submit(lambda: asyncio.sleep(0, result="independent"), session_id="b", priority=10)
            == "independent"
        )
        assert not release.is_set()
        forbidden.assert_not_called()
        assert runtime.stats()["rejected"] == runtime.stats()["cancelled"] == 1
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.shutdown()
