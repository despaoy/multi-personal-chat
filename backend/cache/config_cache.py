"""Process-local configuration snapshots and Redis knowledge statistics."""

import random
import threading
from time import monotonic
from typing import Dict, Optional

from cache import redis_client
from db.config_utils import coerce_config_value

CONFIG_CACHE_TTL = 60
KNOWLEDGE_STATS_KEY = "cache:knowledge_stats"
KNOWLEDGE_STATS_TTL = 30
_CONFIG_CACHE: tuple[object, float, dict] | None = None
_config_lock = threading.Lock()


def _ttl_with_jitter(ttl: int) -> int:
    jitter = int(ttl * random.uniform(-0.1, 0.1))
    return max(1, ttl + jitter)


def read_config(database) -> tuple[dict, bool]:
    """Return a copied snapshot and cache-hit status for this database."""
    global _CONFIG_CACHE
    with _config_lock:
        cached = _CONFIG_CACHE
        if cached is not None and cached[0] is database and monotonic() < cached[1]:
            return dict(cached[2]), True
        _CONFIG_CACHE = None
        stored = database.config
        if not isinstance(stored, dict):
            raise TypeError("Configuration database must return a mapping")
        value = {key: coerce_config_value(item) for key, item in stored.items()}
        _CONFIG_CACHE = (database, monotonic() + _ttl_with_jitter(CONFIG_CACHE_TTL), value)
        return dict(value), False


def invalidate_config_cache() -> None:
    """Wait for an in-flight load before invalidating its snapshot."""
    global _CONFIG_CACHE
    with _config_lock:
        _CONFIG_CACHE = None


def get_cached_knowledge_stats() -> Optional[Dict]:
    """获取缓存的知识库统计"""
    return redis_client.cache_get(KNOWLEDGE_STATS_KEY)


def set_cached_knowledge_stats(stats: Dict) -> bool:
    """缓存知识库统计"""
    return redis_client.cache_set(KNOWLEDGE_STATS_KEY, stats, _ttl_with_jitter(KNOWLEDGE_STATS_TTL))


def invalidate_knowledge_cache() -> None:
    """使知识库相关缓存全部失效"""
    redis_client.cache_delete(KNOWLEDGE_STATS_KEY)
    # 清空知识库列表缓存
    for key in redis_client.cache_keys("cache:kb:*"):
        redis_client.cache_delete(key)
