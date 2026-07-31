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
