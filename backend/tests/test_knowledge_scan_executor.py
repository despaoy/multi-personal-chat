"""Scans yield the event loop and keep finite admission and explicit errors."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api import knowledge
from app.dependencies import get_current_admin
from app.runtime import RuntimeContainer
from infra.bounded_executor import BoundedThreadExecutor
from test_owned_startup_resources import lifecycle as owned_lifecycle

lifecycle = owned_lifecycle
TEXT = "完整合成规则：12月5日14点在二楼办理，须本人携身份证和书面确认，未确认不得付款。"


@pytest.fixture
def scans(tmp_path, monkeypatch, request):
    root = tmp_path / "sources"
    target = root / "rules"
    target.mkdir(parents=True)
    (target / "complete.txt").write_text(TEXT, encoding="utf-8")
    monkeypatch.setattr(knowledge, "KNOWLEDGE_BASES_DIR", root)
    pool = BoundedThreadExecutor(
        name="controlled-scan", max_workers=1, max_pending=getattr(request, "param", 1), default_timeout=1
    )
    runtime = RuntimeContainer(db=object(), is_pg_mode=lambda: False, knowledge_scan_executor=pool)
    app = FastAPI()
    app.state.runtime_container = runtime
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    errors = []

    @app.exception_handler(HTTPException)
    async def error_response(request, error):
        errors.append(error)
        return JSONResponse({"detail": error.detail}, status_code=error.status_code)

    @app.get("/heartbeat")
    async def heartbeat():
        return {"alive": True}

    try:
        yield SimpleNamespace(app=app, runtime=runtime, pool=pool, errors=errors)
    finally:
        pool.shutdown()


def blocking_scan(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = knowledge._scan_knowledge_dirs
    calls = []

    def scan():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(3), "Test must release controlled filesystem work"
        return original()

    monkeypatch.setattr(knowledge, "_scan_knowledge_dirs", scan)
    return entered, release, calls


async def started(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 1), timeout=2)


async def idle(pool):
    for _ in range(100):
        if pool.stats()["pending"] == 0:
            return
        await asyncio.sleep(0.01)
    pytest.fail("Released work did not leave executor")


async def test_scan_runs_in_worker_while_unrelated_request_completes(scans, monkeypatch):
    entered, release, calls = blocking_scan(monkeypatch)
    event_loop_thread = threading.get_ident()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=scans.app), base_url="http://test") as client:
        task = asyncio.create_task(client.get("/api/knowledge/scan"))
        try:
            await started(entered)
            assert (await asyncio.wait_for(client.get("/heartbeat"), timeout=0.5)).json() == {"alive": True}
            assert calls == [calls[0]] and calls[0] != event_loop_thread
        finally:
            release.set()
        response = await task
    assert response.status_code == 200 and response.json()["success"]
    tree = response.json()["directories"][0]
    assert tree["fileCount"] == 1 and tree["totalSize"] == len(TEXT.encode())
    assert tree["children"][0]["name"] == "complete.txt"


async def test_timeout_reports_failure_and_keeps_slot_until_actual_work_finishes(scans, monkeypatch, caplog):
    entered, release, calls = blocking_scan(monkeypatch)
    scans.pool.default_timeout = 0.05
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=scans.app), base_url="http://test") as client:
        task = asyncio.create_task(client.get("/api/knowledge/scan"))
        try:
            await started(entered)
            response = await task
            assert response.status_code == 503 and "BlockingWorkTimeout" in response.text
            assert scans.pool.stats()["pending"] == 1
            rejected = await client.get("/api/knowledge/scan")
            assert rejected.status_code == 503 and "BlockingWorkRejected" in rejected.text
            assert len(calls) == 1 and "directories" not in response.json()
        finally:
            release.set()
        await idle(scans.pool)
        recovered = await client.get("/api/knowledge/scan")
    assert recovered.status_code == 200 and recovered.json()["directories"][0]["fileCount"] == 1
    assert "知识库目录扫描失败" in caplog.text


@pytest.mark.parametrize("scans", [1, 2], indirect=True)
async def test_cancelled_running_or_queued_scan_never_frees_a_running_slot_early(scans, monkeypatch):
    entered, release, calls = blocking_scan(monkeypatch)
    queued = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=scans.app), base_url="http://test") as client:
        running = asyncio.create_task(client.get("/api/knowledge/scan"))
        try:
            await started(entered)
            if scans.pool.max_pending == 2:
                queued = asyncio.create_task(client.get("/api/knowledge/scan"))
                for _ in range(100):
                    if scans.pool.stats()["pending"] == 2:
                        break
                    await asyncio.sleep(0.001)
                assert scans.pool.stats()["pending"] == 2
                cancelled = queued
            else:
                cancelled = running
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            rejected = await client.get("/api/knowledge/scan")
            assert rejected.status_code == 503 and "BlockingWorkRejected" in rejected.text
            assert len(calls) == 1
        finally:
            release.set()
        if queued is not None:
            assert (await running).status_code == 200
        await idle(scans.pool)
    assert len(calls) == 1
    assert scans.pool.stats()["cancelled"] == (1 if queued is not None else 0)


@pytest.mark.parametrize("error_type", [OSError, ValueError])
async def test_scan_failure_preserves_cause_without_empty_success_or_private_text(
    scans, monkeypatch, caplog, error_type
):
    failure = error_type("synthetic-private-filesystem-detail")
    scan = Mock(side_effect=failure)
    monkeypatch.setattr(knowledge, "_scan_knowledge_dirs", scan)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=scans.app), base_url="http://test") as client:
        response = await client.get("/api/knowledge/scan")
    assert response.status_code == 503 and scans.errors[-1].__cause__ is failure
    assert "synthetic-private" not in response.text + caplog.text
    assert "directories" not in response.json()
    scan.assert_called_once_with()


@pytest.mark.parametrize("state", ["absent", "closed"])
async def test_unavailable_executor_fails_without_allocating_a_replacement(scans, monkeypatch, state):
    scan = Mock(side_effect=AssertionError("Unavailable executor must not run the scan"))
    monkeypatch.setattr(knowledge, "_scan_knowledge_dirs", scan)
    if state == "absent":
        scans.runtime.knowledge_scan_executor = None
    else:
        scans.pool.shutdown()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=scans.app), base_url="http://test") as client:
        response = await client.get("/api/knowledge/scan")
    assert response.status_code == 503 and isinstance(scans.errors[-1].__cause__, RuntimeError)
    assert scans.runtime.knowledge_scan_executor is (None if state == "absent" else scans.pool)
    scan.assert_not_called()


async def test_each_application_lifespan_gets_a_fresh_executor_and_closes_it(lifecycle):
    app, _ = lifecycle.application()
    runtime = app.state.runtime_container
    pools = []
    for _ in range(2):
        async with lifecycle.main.lifespan(app):
            pool = runtime.knowledge_scan_executor
            pools.append(pool)
            assert pool.max_workers == 2 and pool.max_pending == 4 and pool.default_timeout == 30
            assert await pool.run(lambda: 42) == 42
        assert runtime.knowledge_scan_executor is None
        with pytest.raises(RuntimeError):
            await pool.run(lambda: pytest.fail("Closed pool accepted work"))
    assert pools[0] is not pools[1]


async def test_failed_startup_closes_scan_executor_and_preserves_startup_error(lifecycle, monkeypatch):
    created = []
    constructor = lifecycle.main.BoundedThreadExecutor

    def allocate(**kwargs):
        pool = constructor(**kwargs)
        created.append(pool)
        return pool

    monkeypatch.setattr(lifecycle.main, "BoundedThreadExecutor", allocate)
    lifecycle.worker.side_effect = lifecycle.failure
    app, _ = lifecycle.application()
    with pytest.raises(RuntimeError) as caught:
        async with lifecycle.main.lifespan(app):
            pytest.fail("Failed startup admitted requests")
    assert caught.value is lifecycle.failure and app.state.runtime_container.knowledge_scan_executor is None
    assert len(created) == 1
    with pytest.raises(RuntimeError):
        await created[0].run(lambda: pytest.fail("Startup failure leaked executor"))


async def test_scan_close_failure_still_closes_later_resources_and_remains_visible(lifecycle, monkeypatch, caplog):
    app, _ = lifecycle.application()
    failure = RuntimeError("synthetic-private-scan-close-detail")
    with pytest.raises(RuntimeError) as caught:
        async with lifecycle.main.lifespan(app):
            pool = app.state.runtime_container.knowledge_scan_executor
            close = pool.shutdown

            def fail_close():
                close()
                raise failure

            monkeypatch.setattr(pool, "shutdown", fail_close)
    assert caught.value is failure and lifecycle.events == ["sqlite", "http", "backup"]
    assert app.state.runtime_container.knowledge_scan_executor is None
    assert "synthetic-private" not in caplog.text and "资源清理完成" not in caplog.text
