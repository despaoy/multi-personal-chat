"""Invalid drain deadlines cannot mutate accepted jobs or discard their owner."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character import memory_llm as memory
from character.models import UserScope


def scheduler():
    completion = SimpleNamespace(
        complete=AsyncMock(side_effect=AssertionError("source-only job must not call a model")), close=AsyncMock()
    )
    return memory.MemoryEnrichmentScheduler(
        config=memory.MemoryLlmConfig(True, "unused", "fixture", idle_seconds=3600), completion=completion
    )


@pytest.mark.parametrize("operation", ["flush_memory", "shutdown"])
@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf"), float("-inf")])
async def test_invalid_deadline_preserves_buffered_work_timers_and_open_owner(operation, timeout):
    worker = scheduler()
    repository = SimpleNamespace(capture_source=AsyncMock(return_value="recorded"))
    receipt = asyncio.get_running_loop().create_future()
    assert worker.schedule(
        repository=repository,
        character_id="synthetic-role",
        user_scope=UserScope("web", "drain", "user", "user", "private"),
        message="请记住，我喜欢红茶。",
        rule_hints=[],
        source_message_id="synthetic-drain-source",
        immediate=False,
        source_only=True,
        receipt=receipt,
    )
    timers = dict(worker._idle_tasks)
    try:
        with pytest.raises(ValueError, match="timeout must be finite and nonnegative"):
            await getattr(worker, operation)(timeout=timeout)
        assert (
            not worker.status.closed
            and worker.status.buffered == 1
            and worker.status.queued == worker.status.processing == 0
        )
        assert worker._idle_tasks == timers and not any(task.done() for task in timers.values())
        assert not receipt.done() and worker._worker is None
        repository.capture_source.assert_not_awaited()
        worker._completion.complete.assert_not_awaited()
        worker._completion.close.assert_not_awaited()
    finally:
        await worker.shutdown(timeout=0)
    assert worker.status.closed and receipt.done()


@pytest.mark.parametrize("setting", ["synthetic-invalid-number", "nan", "-1"])
async def test_invalid_shutdown_setting_preserves_global_scheduler(monkeypatch, setting):
    worker = scheduler()
    monkeypatch.setattr(memory, "_default_scheduler", worker)
    monkeypatch.setenv("MEMORY_LLM_SHUTDOWN_TIMEOUT", setting)
    try:
        with pytest.raises(ValueError):
            await memory.shutdown_memory_enrichment()
        assert memory._default_scheduler is worker and not worker.status.closed
        worker._completion.close.assert_not_awaited()
    finally:
        await worker.shutdown(timeout=0)


async def test_client_close_failure_retains_owner_until_explicit_successful_cleanup(monkeypatch):
    worker = scheduler()
    failure = OSError("synthetic-close-failure")
    worker._completion.close.side_effect = [failure, None]
    monkeypatch.setattr(memory, "_default_scheduler", worker)
    monkeypatch.setenv("MEMORY_LLM_SHUTDOWN_TIMEOUT", "0")
    with pytest.raises(OSError) as caught:
        await memory.shutdown_memory_enrichment()
    assert caught.value is failure and memory._default_scheduler is worker
    assert worker.status.closed and worker._completion.close.await_count == 1
    await memory.shutdown_memory_enrichment()
    assert memory._default_scheduler is None and worker._completion.close.await_count == 2


@pytest.mark.parametrize("setting", ["0", "0.5"])
async def test_valid_shutdown_deadline_closes_and_releases_global_owner(monkeypatch, setting):
    worker = scheduler()
    monkeypatch.setattr(memory, "_default_scheduler", worker)
    monkeypatch.setenv("MEMORY_LLM_SHUTDOWN_TIMEOUT", setting)
    await memory.shutdown_memory_enrichment()
    assert worker.status.closed and memory._default_scheduler is None
    worker._completion.close.assert_awaited_once_with()
