"""Small bounded TTL cache for frequently-read scalar values."""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import RLock
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BoundedTTLCache(Generic[K, V]):
    """Thread-safe in-process TTL cache with deterministic LRU eviction."""

    def __init__(self, *, ttl: float = 60.0, max_size: int = 4096) -> None:
        self._ttl = max(0.001, float(ttl))
        self._max_size = max(1, int(max_size))
        self._values: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: K) -> V | None:
        now = time.monotonic()
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._values.pop(key, None)
                return None
            self._values.move_to_end(key)
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._values[key] = (time.monotonic() + self._ttl, value)
            self._values.move_to_end(key)
            while len(self._values) > self._max_size:
                self._values.popitem(last=False)

    def invalidate(self, key: K) -> None:
        with self._lock:
            self._values.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)
