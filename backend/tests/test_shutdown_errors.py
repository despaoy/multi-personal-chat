"""Cleanup always attempts later resources and never reports a failed close as success."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app import main
from cache import redis_client
from test_owned_startup_resources import lifecycle as owned_lifecycle

lifecycle = owned_lifecycle


def exception_chain(error):
    chain = []
    while error is not None:
        chain.append(error)
        error = error.__context__
    return chain


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_close_failure_propagates_original_error_without_private_log(asynchronous, caplog):
    failure = RuntimeError("synthetic-private-connection-detail")
    close = AsyncMock(side_effect=failure) if asynchronous else Mock(side_effect=failure)
    caplog.set_level(logging.ERROR, logger="main")
    with pytest.raises(RuntimeError) as raised:
        await main._close_resource("synthetic resource", SimpleNamespace(close=close), "close")
    assert raised.value is failure
    assert "RuntimeError" in caplog.text and "synthetic-private" not in caplog.text
    close.assert_called_once()


async def test_missing_cleanup_module_does_not_skip_remaining_resources(lifecycle, monkeypatch):
    real_import = main.import_module
    failure = ModuleNotFoundError("synthetic shutdown module missing")

    def import_module(name):
        if name == "evaluation.runtime_runner":
            raise failure
        return real_import(name)

    monkeypatch.setattr(main, "import_module", import_module)
    app, _ = lifecycle.application()
    with pytest.raises(ModuleNotFoundError) as raised:
        async with main.lifespan(app):
            pass
    assert raised.value is failure
    assert lifecycle.events == ["http", "backup"]
    assert all(getattr(app.state.runtime_container, name) is None for name in main._LIFESPAN_RESOURCE_NAMES)


@pytest.mark.parametrize("startup_failure", [False, True])
async def test_all_cleanup_runs_and_multiple_errors_keep_original_startup_failure(
    lifecycle, monkeypatch, caplog, startup_failure
):
    first = RuntimeError("synthetic first close failure")
    second = ValueError("synthetic second close failure")
    monkeypatch.setattr("evaluation.runtime_runner.shutdown_generation_evaluations", AsyncMock(side_effect=first))
    monkeypatch.setattr("inference.vllm_client.close_shared_vllm_client", AsyncMock(side_effect=second))
    if startup_failure:
        lifecycle.worker.side_effect = lifecycle.failure
    app, _ = lifecycle.application()
    with pytest.raises(ValueError) as raised:
        async with main.lifespan(app):
            if startup_failure:
                pytest.fail("Startup failure must not admit requests")
    assert raised.value is second
    chain = exception_chain(raised.value)
    assert first in chain
    if startup_failure:
        assert lifecycle.failure in chain
    assert lifecycle.events == ["http", "backup"]
    assert all(getattr(app.state.runtime_container, name) is None for name in main._LIFESPAN_RESOURCE_NAMES)
    assert "资源清理完成" not in caplog.text


async def test_cleanup_cancellation_propagates_after_attempting_later_resources(lifecycle, monkeypatch):
    failure = asyncio.CancelledError()
    monkeypatch.setattr("evaluation.runtime_runner.shutdown_generation_evaluations", AsyncMock(side_effect=failure))
    app, _ = lifecycle.application()
    with pytest.raises(asyncio.CancelledError) as raised:
        async with main.lifespan(app):
            pass
    assert raised.value is failure
    assert lifecycle.events == ["http", "backup"]


async def test_normal_shutdown_keeps_dependency_order(lifecycle, monkeypatch, caplog):
    order = []
    operations = [
        ("evaluation.runtime_runner.shutdown_generation_evaluations", "evaluation"),
        ("training.task_manager.shutdown_simple_lora_trainer", "training"),
        ("api.knowledge.shutdown_intent_tasks", "rag"),
        ("services.delivery_memory.shutdown_delivery_memory", "delivery"),
        ("services.turn_completion.shutdown_turn_completions", "completions"),
        ("character.memory_llm.shutdown_memory_enrichment", "memory"),
        ("inference.vllm_client.close_shared_vllm_client", "vllm"),
        ("cache.redis_client.close_async_redis", "redis"),
    ]
    for name, label in operations:

        async def close(*_resource_owner, label=label):
            order.append(label)

        monkeypatch.setattr(name, close)
    app, _ = lifecycle.application(postgres=True, db_path=False)

    async def close_database():
        order.append("database")

    app.state.runtime_container.db.close = close_database
    caplog.set_level(logging.INFO, logger="main")
    async with main.lifespan(app):
        pass
    assert order == ["evaluation", "training", "rag", "delivery", "completions", "memory", "database", "vllm", "redis"]
    assert "资源清理完成" in caplog.text


@pytest.mark.parametrize("fail_at", [None, "client", "pool", "both", "cancel"])
async def test_redis_cleanup_releases_both_resources_and_reports_failure(monkeypatch, fail_at):
    first = asyncio.CancelledError() if fail_at == "cancel" else RuntimeError("synthetic client close failure")
    second = ValueError("synthetic pool close failure")
    client = SimpleNamespace(aclose=AsyncMock(side_effect=first if fail_at in {"client", "both", "cancel"} else None))
    pool = SimpleNamespace(aclose=AsyncMock(side_effect=second if fail_at in {"pool", "both"} else None))
    monkeypatch.setattr(redis_client, "_async_client", client)
    monkeypatch.setattr(redis_client, "_async_pool", pool)
    monkeypatch.setattr(redis_client, "_async_lock", asyncio.Lock())
    if fail_at is None:
        await redis_client.close_async_redis()
    else:
        expected = second if fail_at in {"pool", "both"} else first
        with pytest.raises(type(expected)) as raised:
            await redis_client.close_async_redis()
        assert raised.value is expected
        if fail_at == "both":
            assert first in exception_chain(raised.value)
    client.aclose.assert_awaited_once()
    pool.aclose.assert_awaited_once()
    assert redis_client._async_client is redis_client._async_pool is redis_client._async_lock is None
    await redis_client.close_async_redis()
    client.aclose.assert_awaited_once()
    pool.aclose.assert_awaited_once()


async def test_shutdown_closes_model_first_created_while_draining_background_work(lifecycle, monkeypatch):
    from inference import model_manager

    model = SimpleNamespace(shutdown=AsyncMock())

    async def finish_memory_work(_resource_owner):
        database._model_manager = model

    monkeypatch.setattr("character.memory_llm.shutdown_memory_enrichment", finish_memory_work)
    app, database = lifecycle.application()
    async with main.lifespan(app):
        assert database._model_manager is None
    model.shutdown.assert_awaited_once()
    lifecycle.getter.assert_not_called()


@pytest.mark.parametrize("startup_failure", [False, True])
async def test_real_sync_adapter_close_error_reaches_lifespan_after_other_cleanup(lifecycle, monkeypatch, startup_failure):
    from db.pg_database import SyncPgAdapter

    failure = OSError("synthetic-private-postgresql-close")
    backend = SimpleNamespace(init=AsyncMock(), execute_sql=AsyncMock(return_value=[{"value": 1}]),
                              close=AsyncMock(side_effect=failure))
    adapter = SyncPgAdapter(backend)
    app, _ = lifecycle.application(postgres=True, db_path=False)
    app.state.runtime_container.db = adapter
    if startup_failure:
        lifecycle.worker.side_effect = lifecycle.failure
    with pytest.raises(OSError) as caught:
        async with main.lifespan(app):
            assert not startup_failure
    assert caught.value is failure
    if startup_failure:
        assert lifecycle.failure in exception_chain(caught.value)
    assert lifecycle.events == ["http"]
    backend.close.assert_awaited_once()
    assert adapter._loop is adapter._thread is None

async def test_live_generation_exits_before_its_memory_dependencies_close(lifecycle, monkeypatch):
    from infra.concurrency_control import InferenceRuntime

    runtime = InferenceRuntime()
    started, cancelled = asyncio.Event(), asyncio.Event()
    blocked = asyncio.Event()

    async def generate():
        started.set()
        try:
            await blocked.wait()
        finally:
            cancelled.set()

    async def close_memory(_owner_database):
        assert cancelled.is_set(), "memory closed while generation still running"
        assert runtime.stats()["active"] == 0

    monkeypatch.setattr("services.delivery_memory.shutdown_delivery_memory", close_memory)
    app, _ = lifecycle.application()
    app.state.runtime_container.inference_runtime = runtime
    caller = None
    try:
        async with main.lifespan(app):
            caller = asyncio.create_task(runtime.submit(generate, session_id="isolated", priority=0, timeout=5))
            await asyncio.wait_for(started.wait(), 1)
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not runtime._workers and runtime.stats()["queue_size"] == 0
    finally:
        await runtime.shutdown()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
