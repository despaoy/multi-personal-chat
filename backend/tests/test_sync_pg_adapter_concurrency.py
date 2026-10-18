from __future__ import annotations

import asyncio
import sys
import threading
import time
from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from types import ModuleType

import pytest


@pytest.fixture
def sync_adapter_class(monkeypatch):
    monkeypatch.setitem(sys.modules, "asyncpg", ModuleType("asyncpg"))
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:password@localhost/test_database",
    )
    cached_module = sys.modules.get("db.pg_database")
    if cached_module is not None and not hasattr(cached_module, "SyncPgAdapter"):
        import db as db_package

        monkeypatch.delitem(sys.modules, "db.pg_database")
        monkeypatch.delattr(db_package, "pg_database", raising=False)

    from db.pg_database import SyncPgAdapter

    return SyncPgAdapter


class RecordingPgDatabase:
    def __init__(self) -> None:
        self.init_calls = 0
        self.close_calls = 0
        self.loop_ids: set[int] = set()
        self.thread_ids: set[int] = set()
        self.operation_started = threading.Event()
        self.operation_cancelled = threading.Event()

    async def init(self) -> None:
        self.init_calls += 1
        self._record_execution_context()
        await asyncio.sleep(0.02)

    async def echo(self, value: int) -> int:
        self._record_execution_context()
        await asyncio.sleep(0.002)
        return value

    async def blocking_operation(self) -> None:
        self._record_execution_context()
        self.operation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.operation_cancelled.set()
            raise

    async def close(self) -> None:
        self.close_calls += 1
        self._record_execution_context()

    def _record_execution_context(self) -> None:
        self.loop_ids.add(id(asyncio.get_running_loop()))
        self.thread_ids.add(threading.get_ident())


def _wait_until_set(event: threading.Event, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while not event.is_set() and time.monotonic() < deadline:
        time.sleep(0.005)
    return event.is_set()


def test_concurrent_first_use_creates_one_loop_and_initializes_once(
    sync_adapter_class,
):
    backend = RecordingPgDatabase()
    adapter = sync_adapter_class(
        backend,
        init_timeout=2,
        operation_timeout=2,
        close_timeout=2,
    )
    callers = 24
    barrier = threading.Barrier(callers)

    def invoke(value: int) -> int:
        barrier.wait(timeout=2)
        return adapter._run(backend.echo(value))

    with ThreadPoolExecutor(max_workers=callers) as executor:
        results = list(executor.map(invoke, range(callers)))

    assert results == list(range(callers))
    assert backend.init_calls == 1
    assert len(backend.loop_ids) == 1
    assert len(backend.thread_ids) == 1

    adapter.close()


def test_operation_timeout_cancels_background_coroutine(sync_adapter_class):
    backend = RecordingPgDatabase()
    adapter = sync_adapter_class(
        backend,
        init_timeout=2,
        operation_timeout=0.02,
        close_timeout=2,
    )

    with pytest.raises(FutureTimeoutError):
        adapter._run(backend.blocking_operation())

    assert _wait_until_set(backend.operation_cancelled)
    adapter.close()


def test_close_cancels_inflight_work_and_is_idempotent(sync_adapter_class):
    backend = RecordingPgDatabase()
    adapter = sync_adapter_class(
        backend,
        init_timeout=2,
        operation_timeout=5,
        close_timeout=2,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        operation = executor.submit(adapter._run, backend.blocking_operation())
        assert backend.operation_started.wait(timeout=1)
        adapter.close()
        with pytest.raises(CancelledError):
            operation.result(timeout=1)

    assert _wait_until_set(backend.operation_cancelled)
    assert backend.close_calls == 1
    assert adapter._loop is None
    assert adapter._thread is None

    adapter.close()
    assert backend.close_calls == 1

def test_explicit_init_does_not_initialize_backend_twice(sync_adapter_class):
    backend = RecordingPgDatabase()
    adapter = sync_adapter_class(
        backend,
        init_timeout=2,
        operation_timeout=2,
        close_timeout=2,
    )

    adapter.init()

    assert backend.init_calls == 1
    adapter.close()


def test_closed_adapter_rejects_new_work(sync_adapter_class):
    backend = RecordingPgDatabase()
    adapter = sync_adapter_class(
        backend,
        init_timeout=2,
        operation_timeout=2,
        close_timeout=2,
    )
    adapter.init()
    adapter.close()

    with pytest.raises(RuntimeError, match="closed"):
        adapter._run(backend.echo(1))


@pytest.mark.parametrize("deadline", [False, True])
def test_close_failure_is_explicit_after_cancellation_and_loop_cleanup(sync_adapter_class, monkeypatch, caplog, deadline):
    backend = RecordingPgDatabase()
    error = OSError("synthetic-private-close-detail")
    finished = threading.Event()
    async def close():
        backend.close_calls += 1
        try:
            if deadline:
                await asyncio.Event().wait()
            raise error
        finally:
            finished.set()
    monkeypatch.setattr(backend, "close", close)
    adapter = sync_adapter_class(backend, close_timeout=.05)
    adapter.init()
    loop, thread = adapter._loop, adapter._thread
    with pytest.raises(FutureTimeoutError if deadline else OSError) as caught:
        adapter.close()
    if not deadline:
        assert caught.value is error
    assert finished.is_set() and loop.is_closed() and not thread.is_alive()
    assert adapter._loop is adapter._thread is None and backend.close_calls == 1
    assert "synthetic-private" not in caplog.text
    adapter.close()
    assert backend.close_calls == 1


def test_reader_failures_attempt_every_reader_and_engine_and_preserve_all_causes(sync_adapter_class, monkeypatch):
    backend = RecordingPgDatabase()
    engine_error = ValueError("synthetic engine close failure")
    errors = [OSError("synthetic first reader"), RuntimeError("synthetic second reader")]
    closed = []
    async def close():
        backend.close_calls += 1
        raise engine_error
    async def reader(index):
        try:
            yield {"complete": "synthetic snapshot"}
        finally:
            closed.append(index)
            raise errors[index]
    monkeypatch.setattr(backend, "close", close)
    adapter = sync_adapter_class(backend)
    for index in range(2):
        stream = reader(index)
        adapter._run(anext(stream))
        adapter._knowledge_readers.add(stream)
    with pytest.raises(ValueError) as caught:
        adapter.close()
    assert caught.value is engine_error and set(closed) == {0, 1}
    chain, pending = [], [caught.value]
    while pending:
        item = pending.pop()
        if item is not None and item not in chain:
            chain.append(item)
            pending.extend([item.__cause__, item.__context__])
    assert all(error in chain for error in errors)
    assert backend.close_calls == 1 and not adapter._knowledge_readers
    assert adapter._loop is adapter._thread is None


def test_failed_initialization_keeps_its_cause_when_engine_cleanup_also_fails(sync_adapter_class, monkeypatch):
    backend = RecordingPgDatabase()
    primary, cleanup = ValueError("synthetic init failure"), OSError("synthetic cleanup failure")
    async def init():
        raise primary
    async def close():
        raise cleanup
    monkeypatch.setattr(backend, "init", init)
    monkeypatch.setattr(backend, "close", close)
    adapter = sync_adapter_class(backend)
    with pytest.raises(OSError) as caught:
        adapter.init()
    assert caught.value is cleanup and caught.value.__context__ is primary
    assert adapter._loop is adapter._thread is None


def test_thread_stop_timeout_preserves_ownership_until_joined(sync_adapter_class):
    backend = RecordingPgDatabase()
    adapter = sync_adapter_class(backend, thread_join_timeout=.02)
    adapter.init()
    loop, thread = adapter._loop, adapter._thread
    entered, release = threading.Event(), threading.Event()
    def block_loop():
        entered.set()
        release.wait(timeout=2)
    loop.call_soon_threadsafe(block_loop)
    try:
        assert entered.wait(timeout=1)
        with pytest.raises(TimeoutError, match="did not stop"):
            with adapter._state_lock:
                adapter._stop_loop_locked(close_backend=False)
        assert adapter._loop is loop and adapter._thread is thread and thread.is_alive()
    finally:
        release.set()
        thread.join(timeout=2)
        with adapter._state_lock:
            adapter._stop_loop_locked(close_backend=False)
        adapter.close()
    assert loop.is_closed() and not thread.is_alive()
