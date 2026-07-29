"""
语义缓存 - L1进程内LRU + L2 Redis缓存

多级缓存架构:
  请求 → L1 进程内LRU (命中~10-15%, <1ms)
       → L2 Redis语义缓存 (命中~40-60%, ~5ms)
       → LLM推理 (剩余, 1-5s)

核心策略:
  - L1: 精确匹配 (normalized hash)
  - L2: Redis缓存，精确匹配 + 相似度前缀分组
  - 语义相似度通过 normalized text hash 分桶实现
  - 缓存命中时直接返回，不调用LLM
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from interfaces import CacheInterface

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """文本归一化：小写、去空白、去标点"""
    text = text.lower().strip()
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除标点（保留中文、字母、数字）
    text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text)
    return text


def text_hash(text: str) -> str:
    """计算文本的SHA256哈希"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]


def text_prefix_hash(text: str, prefix_len: int = 8) -> str:
    """计算文本前缀哈希（用于L2分桶，相似文本归入同一桶）"""
    normalized = normalize_text(text)
    # 取前N个字符作为桶键
    prefix = normalized[:prefix_len] if len(normalized) >= prefix_len else normalized
    return f"sc:bucket:{text_hash(prefix)}"


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    ttl: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class L1LRUCache:
    """L1 进程内LRU缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._max_size:
                # Evict oldest
                self._cache.popitem(last=False)
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl or self._default_ttl,
            )

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "level": "L1",
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0,
        }


class L2RedisCache:
    """L2 Redis缓存（语义分桶）"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", default_ttl: float = 3600.0):
        self._redis_url = redis_url
        self._default_ttl = default_ttl
        self._client = None
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._available = False

    async def _ensure_client(self):
        """Lazy init Redis async client（复用全局共享连接池）"""
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            try:
                # 复用 cache.redis_client 的共享连接池，避免独立创建连接池
                from cache.redis_client import get_async_redis
                self._client = await get_async_redis()
                if self._client is None:
                    self._available = False
                    return None
                self._available = True
                logger.info(f"L2语义缓存已连接(共享连接池): {self._redis_url}")
                return self._client
            except Exception as e:
                logger.warning(f"L2 Redis不可用，仅使用L1缓存: {e}")
                self._available = False
                return None

    async def get(self, key: str) -> Optional[Any]:
        """通过精确hash key获取缓存"""
        client = await self._ensure_client()
        if not client or not self._available:
            self._misses += 1
            return None
        try:
            cache_key = f"sc:exact:{key}"
            raw = await client.get(cache_key)
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"L2 cache get error: {e}")
            self._misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存，同时写入分桶索引"""
        client = await self._ensure_client()
        if not client or not self._available:
            return
        try:
            ttl = ttl or self._default_ttl
            # 精确匹配缓存
            cache_key = f"sc:exact:{key}"
            await client.setex(cache_key, int(ttl), json.dumps(value, ensure_ascii=False, default=str))
        except Exception as e:
            logger.debug(f"L2 cache set error: {e}")

    async def delete(self, key: str) -> bool:
        client = await self._ensure_client()
        if not client or not self._available:
            return False
        try:
            cache_key = f"sc:exact:{key}"
            result = await client.delete(cache_key)
            return result > 0
        except Exception:
            return False

    async def clear(self) -> None:
        """清除所有语义缓存键"""
        client = await self._ensure_client()
        if not client or not self._available:
            return
        try:
            # Use SCAN instead of KEYS for production safety
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match="sc:*", count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
            self._hits = 0
            self._misses = 0
        except Exception as e:
            logger.warning(f"L2 cache clear error: {e}")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            self._available = False

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "level": "L2",
            "available": self._available,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0,
        }


class SemanticCache:
    """两级语义缓存：L1进程内LRU + L2 Redis"""

    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_ttl: float = 300.0,
        l2_ttl: float = 3600.0,
        redis_url: Optional[str] = None,
    ):
        self._l1 = L1LRUCache(max_size=l1_max_size, default_ttl=l1_ttl)
        self._l2 = L2RedisCache(
            redis_url=redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            default_ttl=l2_ttl,
        )

    def _compute_key(self, text: str, context: Optional[str] = None) -> str:
        """计算缓存键 = hash(normalized_text + optional_context)"""
        normalized = normalize_text(text)
        if context:
            normalized = f"{normalized}:{normalize_text(context)}"
        return text_hash(normalized)

    async def get(self, prompt: str, context: Optional[str] = None) -> Optional[Any]:
        """查询缓存：L1 → L2 → Miss"""
        key = self._compute_key(prompt, context)

        # L1 lookup
        result = await self._l1.get(key)
        if result is not None:
            logger.debug(f"语义缓存L1命中: {key[:8]}")
            return result

        # L2 lookup
        result = await self._l2.get(key)
        if result is not None:
            # Promote to L1
            await self._l1.set(key, result)
            logger.debug(f"语义缓存L2命中: {key[:8]}")
            return result

        return None

    async def set(self, prompt: str, value: Any, context: Optional[str] = None) -> None:
        """写入缓存：同时写L1和L2"""
        key = self._compute_key(prompt, context)
        await self._l1.set(key, value)
        await self._l2.set(key, value)

    async def delete(self, prompt: str, context: Optional[str] = None) -> bool:
        """删除缓存"""
        key = self._compute_key(prompt, context)
        l1_deleted = await self._l1.delete(key)
        l2_deleted = await self._l2.delete(key)
        return l1_deleted or l2_deleted

    async def clear(self) -> None:
        """清除所有缓存"""
        await self._l1.clear()
        await self._l2.clear()
        logger.info("语义缓存已清除")

    async def close(self) -> None:
        """关闭连接"""
        await self._l2.close()

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "l1": self._l1.stats(),
            "l2": self._l2.stats(),
        }


# 全局实例
_semantic_cache: Optional[SemanticCache] = None


async def get_semantic_cache() -> SemanticCache:
    """获取全局语义缓存实例"""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache


# 接口契约验证：确保 SemanticCache 实现 CacheInterface 接口
assert isinstance(SemanticCache(), CacheInterface), f"SemanticCache must implement CacheInterface"
