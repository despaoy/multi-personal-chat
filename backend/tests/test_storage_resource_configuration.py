"""Configured storage resources must not silently clamp or disable themselves."""

import builtins
import importlib.util
from pathlib import Path

import pytest

from cache.ttl_value_cache import BoundedTTLCache


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), True, "1"])
def test_ttl_cache_rejects_invalid_duration(ttl):
    with pytest.raises(ValueError, match="finite positive"):
        BoundedTTLCache(ttl=ttl)


@pytest.mark.parametrize("max_size", [0, -1, 1.5, True, "1"])
def test_ttl_cache_rejects_invalid_capacity(max_size):
    with pytest.raises(ValueError, match="positive integer"):
        BoundedTTLCache(max_size=max_size)


def test_required_http_dependency_failure_propagates(monkeypatch):
    from infra import resource_pool

    original = builtins.__import__
    failure = ImportError("synthetic missing httpx")

    def blocked(name, *args, **kwargs):
        if name == "httpx":
            raise failure
        return original(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("infra.resource_pool", Path(resource_pool.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError) as caught:
        spec.loader.exec_module(module)
    assert caught.value is failure


def test_postgres_mapping_errors_do_not_switch_record_protocol(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/test_database")
    from db.pg_database import _row_to_dict

    failure = OSError("synthetic row mapping unavailable")

    class BrokenRow(dict):
        @property
        def _mapping(self):
            raise failure

    with pytest.raises(OSError) as caught:
        _row_to_dict(BrokenRow(hidden="value"))
    assert caught.value is failure
    assert _row_to_dict({"value": 1}) == {"value": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sqlite", "http"])
async def test_pool_close_attempts_every_resource_and_propagates_failure(tmp_path, kind):
    from types import SimpleNamespace

    from infra.resource_pool import ConnectionPool, HttpClientPool

    closed = []
    failure = OSError("synthetic close failure")

    def close(index):
        closed.append(index)
        if index == 1:
            raise failure

    async def aclose(index):
        close(index)

    if kind == "sqlite":
        pool = ConnectionPool(str(tmp_path / "pool.db"))
        pool._pool = [SimpleNamespace(conn=SimpleNamespace(close=lambda i=i: close(i))) for i in range(3)]
    else:
        pool = HttpClientPool()
        pool._client = SimpleNamespace(aclose=lambda: aclose(1))
    with pytest.raises(OSError) as caught:
        await pool.close()
    assert caught.value is failure
    assert closed == ([0, 1, 2] if kind == "sqlite" else [1])
    assert pool._closed
    assert not (pool._pool if kind == "sqlite" else pool._client)
    await pool.close()



@pytest.mark.asyncio
@pytest.mark.parametrize("transport_fails", [False, True])
async def test_http_pool_shares_one_client_and_releases_bounded_requests(monkeypatch, transport_fails):
    import asyncio
    import json
    from unittest.mock import Mock

    import httpx
    from infra import resource_pool

    entered = []
    ready, fourth = asyncio.Event(), asyncio.Event()
    release = [asyncio.Event() for _ in range(4)]
    failure = httpx.ConnectError("controlled transport failure")

    async def transport(request):
        payload = json.loads(request.content)
        index = payload["index"]
        assert payload["message"] == "周三下班后喝茶，工作日早晨不喝；出差时只喝白水。"
        assert request.headers["authorization"] == f"Bearer synthetic-{index}"
        entered.append(index)
        if len(entered) == 3:
            ready.set()
        if index == 3:
            fourth.set()
        await release[index].wait()
        if transport_fails and index == 1:
            raise failure
        return httpx.Response(200, json=payload)

    original = httpx.AsyncClient
    created = []

    def client(**kwargs):
        instance = original(transport=httpx.MockTransport(transport), **kwargs)
        created.append(instance)
        return instance

    factory = Mock(side_effect=client)
    monkeypatch.setattr(resource_pool.httpx, "AsyncClient", factory)
    pool = resource_pool.HttpClientPool(max_connections=3, request_timeout=7)

    async def post(index):
        async with pool.acquire() as borrowed:
            response = await borrowed.post(
                "https://synthetic.invalid/completion",
                headers={"authorization": f"Bearer synthetic-{index}"},
                json={"index": index, "message": "周三下班后喝茶，工作日早晨不喝；出差时只喝白水。"},
            )
            return response.json()["index"]

    tasks = [asyncio.create_task(post(index)) for index in range(4)]
    try:
        await asyncio.wait_for(ready.wait(), 5)
        assert len(created) == factory.call_count == 1
        assert entered == [0, 1, 2] and not tasks[3].done()
        assert pool.get_pool_stats()["active_requests"] == 3
        release[0].set()
        await asyncio.wait_for(fourth.wait(), 5)
        assert pool.get_pool_stats()["active_requests"] == 3
        for gate in release:
            gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert results == ([0, failure, 2, 3] if transport_fails else [0, 1, 2, 3])
        assert factory.call_count == 1
        stats = pool.get_pool_stats()
        assert stats["active_requests"] == 0 and stats["total_acquired"] == stats["total_released"] == 4
        assert stats["client_initialized"] is True and not stats["closed"]
    finally:
        for gate in release:
            gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await pool.close()
    assert all(instance.is_closed for instance in created)
    assert pool.get_pool_stats()["client_initialized"] is False


@pytest.mark.asyncio
async def test_http_pool_close_rejects_waiter_without_reopening_client(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from infra import resource_pool

    owned = SimpleNamespace(aclose=AsyncMock())
    factory = Mock(return_value=owned)
    monkeypatch.setattr(resource_pool.httpx, "AsyncClient", factory)
    pool = resource_pool.HttpClientPool(max_connections=1)
    waiting = asyncio.Event()

    async def borrower():
        waiting.set()
        async with pool.acquire():
            return "incorrectly reopened"

    async with pool.acquire() as first:
        assert first is owned
        task = asyncio.create_task(borrower())
        await waiting.wait()
        assert not task.done()
        await pool.close()
        with pytest.raises(RuntimeError, match="已关闭"):
            async with pool.acquire():
                pytest.fail("closed pool must not admit another borrower")
    try:
        with pytest.raises(RuntimeError, match="已关闭"):
            await task
        owned.aclose.assert_awaited_once()
        assert factory.call_count == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_http_pool_cancelled_request_releases_slot_and_keeps_client(monkeypatch):
    import asyncio
    from unittest.mock import Mock

    import httpx
    from infra import resource_pool

    entered = asyncio.Event()
    stop = asyncio.Event()

    async def transport(request):
        if request.url.path == "/cancel":
            entered.set()
            await stop.wait()
        return httpx.Response(200, json={"ok": True})

    owned = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    factory = Mock(return_value=owned)
    monkeypatch.setattr(resource_pool.httpx, "AsyncClient", factory)
    pool = resource_pool.HttpClientPool(max_connections=1)

    async def request():
        async with pool.acquire() as client:
            await client.get("https://synthetic.invalid/cancel")

    task = asyncio.create_task(request())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.get_pool_stats()["active_requests"] == 0
        async with pool.acquire() as client:
            assert client is owned
            response = await client.get("https://synthetic.invalid/healthy")
            assert response.json() == {"ok": True}
        stats = pool.get_pool_stats()
        assert stats["active_requests"] == 0 and stats["total_acquired"] == stats["total_released"] == 2
        assert factory.call_count == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()
    assert owned.is_closed
