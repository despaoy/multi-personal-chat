"""Exact cache behavior and bounded resources, without model or production data."""

import asyncio
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from cache import response_cache
from cache.response_cache import ResponseCache


@pytest.mark.parametrize("size", [0, -1, 1.5, True, None])
def test_invalid_capacity_fails_before_cache_operations(size):
    with pytest.raises(ValueError, match="capacity"):
        ResponseCache(max_size=size)


@pytest.mark.parametrize("ttl", [-1, float("nan"), float("inf"), True, "60"])
def test_invalid_default_lifetime_is_not_replaced_with_a_default(ttl):
    with pytest.raises(ValueError, match="TTL"):
        ResponseCache(default_ttl=ttl)


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(response_cache.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(response_cache.random, "uniform", lambda low, high: 0.0)
    return now


@pytest.mark.asyncio
async def test_model_identity_and_literal_values_never_collide(clock):
    cache = ResponseCache(max_size=8)
    values = ["__NULL__", False, 0, ""]
    for index, value in enumerate(values):
        await cache.set("same-prompt", str(index), value)
    assert [await cache.get("same-prompt", str(i)) for i in range(4)] == values
    await cache.set("a::b", "c", "first")
    await cache.set("a", "b::c", "second")
    assert await cache.get("a::b", "c") == "first" and await cache.get("a", "b::c") == "second"
    assert await cache.get("same-prompt", "unknown-model") is None
    assert cache.stats == {"hits": 6, "misses": 1, "hit_rate": 6 / 7, "size": 6, "max_size": 8}


@pytest.mark.asyncio
async def test_lru_hits_and_replacement_do_not_evict_an_unrelated_response(clock):
    cache = ResponseCache(max_size=2)
    await cache.set("a", "model", "A")
    await cache.set("b", "model", "B")
    assert await cache.get("a", "model") == "A"
    await cache.set("c", "model", "C")
    assert await cache.get("b", "model") is None
    await cache.set("a", "model", "updated-A")
    assert await cache.get("c", "model") == "C" and await cache.get("a", "model") == "updated-A"
    assert cache.stats["size"] == 2


@pytest.mark.asyncio
async def test_zero_lifetime_explicitly_disables_storage_and_invalidates_old_value(clock):
    cache = ResponseCache(max_size=2)
    await cache.set("a", "model", "old", ttl=60)
    await cache.set("a", "model", "new", ttl=0)
    assert await cache.get("a", "model") is None and cache.stats["size"] == 0
    disabled = ResponseCache(default_ttl=0)
    await disabled.set("a", "model", "new")
    assert await disabled.get("a", "model") is None and disabled.stats["size"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [-1, float("nan"), float("inf"), True, "60"])
async def test_bad_write_lifetime_keeps_the_existing_complete_response(clock, ttl):
    cache = ResponseCache()
    await cache.set("a", "model", {"reply": "完整有效的旧结果"}, ttl=60)
    with pytest.raises(ValueError, match="TTL"):
        await cache.set("a", "model", {"reply": "应拒绝的新写入"}, ttl=ttl)
    assert await cache.get("a", "model") == {"reply": "完整有效的旧结果"}


@pytest.mark.asyncio
async def test_missing_response_never_becomes_a_negative_cache_or_deletes_good_data(clock):
    cache = ResponseCache()
    await cache.set("a", "model", "complete-response")
    with pytest.raises(ValueError, match="missing response"):
        await cache.set("a", "model", None)
    assert await cache.get("a", "model") == "complete-response"


@pytest.mark.asyncio
async def test_expiry_uses_elapsed_time_and_cache_hits_do_not_extend_it(clock, monkeypatch):
    cache = ResponseCache(default_ttl=10)
    await cache.set("a", "model", "complete-response")
    clock[0] = 109
    assert await cache.get("a", "model") == "complete-response"
    monkeypatch.setattr(response_cache.time, "time", lambda: -1_000_000)
    clock[0] = 110
    assert await cache.get("a", "model") is None and cache.stats["size"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("jitter,deadline", [(-0.1, 109), (0.1, 111)])
async def test_existing_jitter_range_remains_bounded(clock, monkeypatch, jitter, deadline):
    monkeypatch.setattr(response_cache.random, "uniform", lambda low, high: jitter)
    cache = ResponseCache(default_ttl=10)
    await cache.set("a", "model", "complete-response")
    clock[0] = deadline - 0.01
    assert await cache.get("a", "model") == "complete-response"
    clock[0] = deadline
    assert await cache.get("a", "model") is None


@pytest.mark.asyncio
async def test_cleanup_and_pattern_invalidation_keep_other_models_and_prompts(clock):
    cache = ResponseCache(max_size=8)
    await cache.set("matching-prompt", "model-A", "A", ttl=10)
    await cache.set("matching-prompt", "model-B", "B", ttl=60)
    await cache.set("other-prompt", "matching-model", "C", ttl=60)
    clock[0] = 110
    assert await cache.cleanup_expired() == 1 and await cache.cleanup_expired() == 0
    assert await cache.invalidate("model") == 0
    assert await cache.invalidate("matching-prompt") == 1
    assert await cache.get("other-prompt", "matching-model") == "C"
    assert await cache.invalidate() == 1 and await cache.invalidate() == 0


@pytest.mark.asyncio
async def test_many_misses_do_not_retain_a_resource_for_each_query():
    cache = ResponseCache(max_size=4)
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        await cache.get("warmup", "model")
        before = tracemalloc.get_traced_memory()[0]
        for i in range(2000):
            assert await cache.get("fiction-miss-" + str(i), "model") is None
        growth = tracemalloc.get_traced_memory()[0] - before
    finally:
        if not already_tracing:
            tracemalloc.stop()
    assert growth < 100_000, f"Query misses retained {growth} bytes without any response"
    assert cache.stats["size"] == 0 and cache.stats["misses"] == 2001


@pytest.mark.asyncio
async def test_independent_cache_operations_keep_capacity_and_expiry_coherent(clock):
    cache = ResponseCache(max_size=8, default_ttl=10)
    await asyncio.gather(*(cache.set(str(i), "model", {"value": i}) for i in range(64)))
    results = await asyncio.gather(*(cache.get(str(i), "model") for i in range(64)))
    assert [row["value"] for row in results if row is not None] == list(range(56, 64))
    assert cache.stats["size"] == 8 and cache.stats["hits"] == 8 and cache.stats["misses"] == 56
    clock[0] = 110
    assert await cache.cleanup_expired() == 8 and cache.stats["size"] == 0


@pytest.mark.asyncio
async def test_admin_cache_operations_keep_the_existing_response_contract(clock, monkeypatch):
    from api import enhanced
    cache = ResponseCache(max_size=2)
    monkeypatch.setattr(enhanced, "response_cache", cache)
    await cache.set("fiction-prompt", "model", {"reply": "完整的管理接口控制"})
    result = await enhanced.get_cache_stats(current_user={"isAdmin": True})
    assert result["success"] and result["stats"] == cache.stats
    removed = await enhanced.invalidate_cache("fiction", current_user={"isAdmin": True})
    assert removed["success"] and removed["removed"] == 1 and cache.stats["size"] == 0


def test_required_cache_import_fault_does_not_disable_caching_and_start_successfully(tmp_path):
    code = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "cache.response_cache":
        raise ImportError("fiction-required-cache-unavailable")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
try:
    import app.config
except ImportError as exc:
    if str(exc) != "fiction-required-cache-unavailable":
        raise
    print("required-cache-import-failed")
else:
    raise AssertionError("Cache import failure was hidden during application configuration")
'''
    env = dict(os.environ, ENVIRONMENT="production", USE_POSTGRESQL="false", DATABASE_PATH=str(tmp_path / "boot.sqlite"),
        JWT_SECRET="fiction-required-cache-test-long-enough-secret", DATABASE_URL="postgresql+asyncpg://unused@127.0.0.1:1/unused",
        MODEL_PROVIDER="openai_compat", OPENAI_COMPAT_BASE_URL="https://fiction.example", OPENAI_COMPAT_API_KEY="fiction-key")
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "required-cache-import-failed" in result.stdout
