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
        pool._clients = [SimpleNamespace(aclose=lambda i=i: aclose(i)) for i in range(3)]
    with pytest.raises(OSError) as caught:
        await pool.close()
    assert caught.value is failure
    assert closed == [0, 1, 2]
    assert pool._closed
    assert not (pool._pool if kind == "sqlite" else pool._clients)
    await pool.close()
