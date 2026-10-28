"""Admission configuration and deadlines fail before queue owners start."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app import config
from infra import concurrency_control as controls

KEYS = (
    "INFERENCE_WORKERS",
    "LLM_MAX_CONCURRENCY",
    "MODEL_MAX_CONCURRENCY",
    "INFERENCE_QUEUE_MAX_SIZE",
    "INFERENCE_QUEUE_TIMEOUT",
)


@pytest.fixture(autouse=True)
def clean_queue_environment(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_missing_queue_configuration_preserves_existing_defaults():
    runtime = controls.InferenceRuntime()
    assert (runtime.max_queue_size, runtime.worker_count, runtime.queue_timeout) == (100, 2, 180.0)
    assert config._resolve_concurrency_limit() == 2


@pytest.mark.parametrize(
    "values,workers,local",
    [
        ({"INFERENCE_WORKERS": "3", "LLM_MAX_CONCURRENCY": "5", "MODEL_MAX_CONCURRENCY": "7"}, 3, 5),
        ({"LLM_MAX_CONCURRENCY": "5", "MODEL_MAX_CONCURRENCY": "7"}, 5, 5),
        ({"MODEL_MAX_CONCURRENCY": "7"}, 7, 7),
        ({"INFERENCE_WORKERS": "3", "LLM_MAX_CONCURRENCY": "5", "MODEL_MAX_CONCURRENCY": ""}, 3, 5),
    ],
)
def test_explicit_configuration_precedence_and_legacy_warning(monkeypatch, caplog, values, workers, local):
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    runtime = controls.InferenceRuntime()
    assert runtime.worker_count == workers and config._resolve_concurrency_limit() == local
    assert ("已弃用" in caplog.text) == (set(values) == {"MODEL_MAX_CONCURRENCY"})


@pytest.mark.parametrize(
    "name,value",
    [
        ("INFERENCE_QUEUE_MAX_SIZE", "0"),
        ("INFERENCE_WORKERS", "0"),
        ("INFERENCE_QUEUE_TIMEOUT", "nan"),
        ("INFERENCE_WORKERS", ""),
        ("LLM_MAX_CONCURRENCY", ""),
        ("MODEL_MAX_CONCURRENCY", ""),
    ],
)
def test_invalid_queue_configuration_never_allocates_a_queue(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    queue = Mock(side_effect=AssertionError("invalid configuration must fail before queue allocation"))
    monkeypatch.setattr(controls.asyncio, "PriorityQueue", queue)
    with pytest.raises(RuntimeError, match=name):
        controls.InferenceRuntime()
    queue.assert_not_called()


@pytest.mark.parametrize(
    "name,value", [("LLM_MAX_CONCURRENCY", "0"), ("LLM_MAX_CONCURRENCY", ""), ("MODEL_MAX_CONCURRENCY", "-2")]
)
def test_invalid_local_concurrency_is_not_corrected_or_replaced(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    if name == "LLM_MAX_CONCURRENCY":
        monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "7")
    with pytest.raises(RuntimeError, match=name):
        config._resolve_concurrency_limit()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
async def test_invalid_request_deadline_does_not_start_workers_or_submit_work(monkeypatch, value):
    runtime = controls.InferenceRuntime()
    ensure = AsyncMock(side_effect=AssertionError("invalid deadline must fail before worker startup"))
    factory = Mock(side_effect=AssertionError("invalid deadline must not execute work"))
    monkeypatch.setattr(runtime, "_ensure_workers", ensure)
    before = dict(runtime._stats)
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        await runtime.submit(factory, session_id="synthetic-session", priority=1, timeout=value)
    ensure.assert_not_awaited()
    factory.assert_not_called()
    assert runtime._queue.empty() and runtime._stats == before and not runtime._workers_started


@pytest.mark.parametrize("deadline", [None, 0.5])
async def test_valid_deadlines_complete_real_jobs(monkeypatch, deadline):
    monkeypatch.setenv("INFERENCE_WORKERS", "1")
    monkeypatch.setenv("INFERENCE_QUEUE_MAX_SIZE", "2")
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT", "1.25")
    runtime = controls.InferenceRuntime()
    try:
        assert (
            await runtime.submit(
                lambda: asyncio.sleep(0, result="complete synthetic source"),
                session_id="synthetic-session",
                priority=1,
                timeout=deadline,
            )
            == "complete synthetic source"
        )
        assert runtime._stats["submitted"] == runtime._stats["completed"] == 1
    finally:
        await runtime.shutdown()


async def test_positive_queue_capacity_rejects_overflow_and_preserves_admitted_jobs(monkeypatch):
    monkeypatch.setenv("INFERENCE_WORKERS", "1")
    monkeypatch.setenv("INFERENCE_QUEUE_MAX_SIZE", "1")
    runtime = controls.InferenceRuntime()
    started, release = asyncio.Event(), asyncio.Event()

    async def running():
        started.set()
        await release.wait()
        return "running-source"

    first = asyncio.create_task(runtime.submit(running, session_id="first", priority=1, timeout=2))
    admitted = [first]
    try:
        await asyncio.wait_for(started.wait(), 1)
        second = asyncio.create_task(
            runtime.submit(lambda: asyncio.sleep(0, result="queued-source"), session_id="second", priority=1, timeout=2)
        )
        admitted.append(second)

        async def wait_queued():
            while runtime._queue.empty():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_queued(), 1)
        forbidden = Mock(side_effect=AssertionError("overflow must not execute"))
        with pytest.raises(controls.InferenceQueueFull):
            await runtime.submit(forbidden, session_id="overflow", priority=1, timeout=1)
        forbidden.assert_not_called()
        release.set()
        assert await asyncio.gather(*admitted) == ["running-source", "queued-source"]
        assert runtime._stats["completed"] == 2 and runtime._stats["rejected"] == 1
    finally:
        release.set()
        await asyncio.gather(*admitted, return_exceptions=True)
        await runtime.shutdown()



@pytest.mark.parametrize("ending", ["job_timeout", "running_deadline", "queued_deadline", "cancel"])
async def test_job_timeout_is_failure_while_expired_or_cancelled_wait_is_cancellation(monkeypatch, ending):
    from fastapi import HTTPException
    from api import generate
    from db.schemas import MessageRequest
    from services.chat_generation import ChatGenerationService

    monkeypatch.setenv("INFERENCE_WORKERS", "1")
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT", "0.1")
    runtime = controls.InferenceRuntime()
    entered, stopped, release, occupied = (asyncio.Event() for _ in range(4))
    failure = TimeoutError("synthetic internal service deadline")
    facts = "完整虚构题设：周四14:00至15:00开会，尚未出发，下雨改线上。请求仅用于隔离调度测试。"
    calls = []
    pending = blocker = None

    async def job(request, current_user):
        assert request.message == facts and current_user["user_id"] == 41
        calls.append(request.traceId)
        entered.set()
        try:
            if ending == "job_timeout":
                raise failure
            await release.wait()
        finally:
            stopped.set()

    async def blocking_job():
        occupied.set()
        await release.wait()
        return "occupied-source"

    service = ChatGenerationService(generate_handler=job, inference_runtime=runtime,
        trace_id_factory=lambda: "fiction-queue-timeout")
    try:
        if ending == "queued_deadline":
            blocker = asyncio.create_task(runtime.submit(blocking_job, session_id="occupied", priority=1, timeout=2))
            await asyncio.wait_for(occupied.wait(), 1)
        pending = asyncio.create_task(generate.generate_reply(
            MessageRequest(message=facts, sessionId="fiction-session"),
            current_user={"username": "fiction-user", "user_id": 41}, service=service,
        ))
        if ending == "cancel":
            await asyncio.wait_for(entered.wait(), 1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises(HTTPException) as caught:
                await asyncio.wait_for(pending, 2)
            error = caught.value
            assert error.status_code == 503
            if ending == "job_timeout":
                assert error.__cause__ is failure
            else:
                assert isinstance(error.__cause__, TimeoutError)
        stats = runtime.stats()
        assert stats["failed"] == (ending == "job_timeout")
        assert stats["cancelled"] == (ending != "job_timeout")
        assert len(calls) == (ending != "queued_deadline")
        assert entered.is_set() == stopped.is_set() == (ending != "queued_deadline")
        assert stats["active"] == (ending == "queued_deadline")
        if ending != "cancel":
            assert error.detail == "请求处理超时，请检查队列和后端服务后重试"
        release.set()
        if blocker is not None:
            assert await blocker == "occupied-source"
        assert await runtime.submit(lambda: asyncio.sleep(0, result="complete-after-failure"),
            session_id="fiction-session", priority=1, timeout=1) == "complete-after-failure"
        stats = runtime.stats()
        assert stats["completed"] == 1 + (ending == "queued_deadline")
        assert stats["failed"] == (ending == "job_timeout")
        assert stats["cancelled"] == (ending != "job_timeout")
        assert stats["active"] == stats["queue_size"] == 0
        assert all(not worker.done() for worker in runtime._workers)
    finally:
        release.set()
        for task in (pending, blocker):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (pending, blocker) if task is not None), return_exceptions=True)
        await runtime.shutdown()
