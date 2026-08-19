"""Same-session jobs must observe history written by the previous job."""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_same_session_job_sees_previous_reply_written_inside_lock(monkeypatch):
    monkeypatch.setenv("INFERENCE_WORKERS", "2")
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT", "5")
    from infra.concurrency_control import InferenceRuntime

    runtime = InferenceRuntime()
    history: list[str] = []

    async def job(label: str):
        snapshot = list(history)
        await asyncio.sleep(0.05)
        if label == "m1":
            history.append("user:m1")
            history.append("assistant:r1")
        return snapshot

    first = asyncio.create_task(runtime.submit(lambda: job("m1"), session_id="room", priority=10))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(runtime.submit(lambda: job("m2"), session_id="room", priority=10))
    first_snapshot, second_snapshot = await asyncio.gather(first, second)

    assert first_snapshot == []
    assert second_snapshot == ["user:m1", "assistant:r1"]
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_different_sessions_run_concurrently(monkeypatch):
    monkeypatch.setenv("INFERENCE_WORKERS", "2")
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT", "5")
    from infra.concurrency_control import InferenceRuntime

    runtime = InferenceRuntime()
    started: dict[str, asyncio.Event] = {"a": asyncio.Event(), "b": asyncio.Event()}
    release = asyncio.Event()

    async def job(session_id: str):
        started[session_id].set()
        await release.wait()
        return session_id

    first = asyncio.create_task(runtime.submit(lambda: job("a"), session_id="a", priority=10))
    second = asyncio.create_task(runtime.submit(lambda: job("b"), session_id="b", priority=10))

    await asyncio.wait_for(started["a"].wait(), timeout=1)
    await asyncio.wait_for(started["b"].wait(), timeout=1)
    release.set()
    results = await asyncio.gather(first, second)
    assert set(results) == {"a", "b"}
    await runtime.shutdown()
