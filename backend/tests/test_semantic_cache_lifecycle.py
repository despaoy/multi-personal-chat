from __future__ import annotations

import asyncio


def test_semantic_cache_honors_ttl_contract_and_restarts_between_event_loops(monkeypatch) -> None:
    from cache import semantic_cache as module

    async def no_redis(_self):
        return None

    monkeypatch.setattr(module.L2RedisCache, "_ensure_client", no_redis)

    async def use_cache(value: str):
        cache = await module.get_semantic_cache()
        # The third positional argument is CacheInterface.ttl, not semantic context.
        await cache.set("prompt", value, 30.0)
        assert await cache.get("prompt") == value
        await module.reset_semantic_cache()
        return cache

    first = asyncio.run(use_cache("first"))
    second = asyncio.run(use_cache("second"))

    assert first is not second