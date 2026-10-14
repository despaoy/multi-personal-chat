"""Bounded exact-response cache; callers own the complete request identity."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheEntry:
    result: Any
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at


def _validated_ttl(ttl: float) -> float:
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or ttl < 0:
        raise ValueError("Response cache TTL must be a finite non-negative number")
    return float(ttl)


class ResponseCache:
    """Exact LRU results with one mutex per application event loop.

    TTL uses monotonic time with the existing ±10% jitter; zero explicitly
    disables storage. None denotes a miss and is not a cacheable response.
    The lock covers cache operations, not model generation or coalescing.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("Response cache capacity must be a positive integer")
        self._max_size = max_size
        self._default_ttl = _validated_ttl(default_ttl)
        self._cache: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._hits = 0
        self._misses = 0

    def _get_lock(self) -> asyncio.Lock:
        """A stopped app may reuse this cache from a fresh event loop."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def get(self, prompt_hash: str, cache_key: str) -> Any | None:
        key = (prompt_hash, cache_key)
        async with self._get_lock():
            entry = self._cache.get(key)
            if entry is None or entry.is_expired:
                self._cache.pop(key, None)
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.result

    async def set(self, prompt_hash: str, cache_key: str, result: Any, ttl: float | None = None) -> None:
        if result is None:
            raise ValueError("Response cache cannot store a missing response")
        effective_ttl = self._default_ttl if ttl is None else _validated_ttl(ttl)
        expires_at = time.monotonic() + effective_ttl * (1 + random.uniform(-0.1, 0.1))
        if not math.isfinite(expires_at):
            raise ValueError("Response cache expiry must be finite")
        key = (prompt_hash, cache_key)
        async with self._get_lock():
            self._cache.pop(key, None)
            if effective_ttl == 0:
                return
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = CacheEntry(result=result, expires_at=expires_at)

    async def invalidate(self, pattern: str | None = None) -> int:
        """Clear all entries or only matching prompt identities."""
        async with self._get_lock():
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            keys = [key for key in self._cache if pattern in key[0]]
            for key in keys:
                del self._cache[key]
            return len(keys)

    async def cleanup_expired(self) -> int:
        async with self._get_lock():
            keys = [key for key, entry in self._cache.items() if entry.is_expired]
            for key in keys:
                del self._cache[key]
            if keys:
                logger.info("清理过期缓存条目: %d 条", len(keys))
            return len(keys)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "size": len(self._cache),
            "max_size": self._max_size,
        }
