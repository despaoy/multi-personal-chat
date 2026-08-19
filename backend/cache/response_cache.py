"""Response cache (exact-match LRU cache) moved out of the legacy optimizer module.

Only the production-used ``ResponseCache`` is retained here.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    result: Any
    created_at: float
    ttl: float
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class ResponseCache:
    """基于语义相似度的响应缓存

    支持：
    - 精确匹配缓存：query文本hash完全一致时直接返回
    - 缓存TTL：默认300秒（5分钟），可配置
    - LRU淘汰：默认max_size=1000
    - 缓存统计：命中次数、未命中次数、命中率
    - 缓存失效：支持手动invalidate(pattern)和自动过期
    - 线程安全：使用asyncio.Lock
    - 缓存穿透防护：空值缓存（__NULL__标记）+ 互斥锁防击穿
    """

    _NULL_MARKER = "__NULL__"

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,
    ) -> None:
        """初始化响应缓存

        Args:
            max_size: 缓存最大容量（LRU淘汰）
            default_ttl: 默认缓存TTL（秒）
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        # prompt_hash -> set of cache_keys，用于按pattern失效
        self._prompt_index: dict[str, set[str]] = {}
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        # 互斥锁：防止缓存击穿，每个key一个锁
        self._locks: dict[str, asyncio.Lock] = {}

        # 统计
        self._hits = 0
        self._misses = 0

    def _get_lock(self) -> asyncio.Lock:
        """Rebuild loop-bound mutexes when an application lifecycle restarts."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
            self._locks = {}
        return self._lock

    @staticmethod
    def compute_prompt_hash(prompt: str) -> str:
        """计算prompt的hash值

        Args:
            prompt: prompt文本

        Returns:
            SHA256哈希值的十六进制字符串
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def build_cache_key(
        model_name: str = "",
        lora_name: str = "",
        temperature: float = 0.7,
    ) -> str:
        """构建缓存复合键

        基于 (model_name, lora_name, temperature) 生成复合键。

        Args:
            model_name: 模型名称
            lora_name: LoRA名称
            temperature: 温度参数

        Returns:
            复合键字符串
        """
        raw = f"{model_name}|{lora_name}|{temperature}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _full_key(self, prompt_hash: str, cache_key: str) -> str:
        """生成完整缓存键"""
        return f"{prompt_hash}::{cache_key}"

    async def get(self, prompt_hash: str, cache_key: str) -> Optional[Any]:
        """查询缓存

        精确匹配缓存：prompt_hash + cache_key 完全一致时返回缓存结果。
        过期的条目会被自动移除。
        支持空值缓存（__NULL__标记），防止缓存穿透。
        支持互斥锁机制，防止缓存击穿。

        Args:
            prompt_hash: prompt的hash值
            cache_key: 缓存复合键

        Returns:
            缓存的结果，未命中返回None
        """
        key = self._full_key(prompt_hash, cache_key)

        # 获取该key对应的互斥锁
        if key not in self._locks:
            async with self._get_lock():
                # double-check: 可能在等待锁时已被其他协程创建
                if key not in self._locks:
                    self._locks[key] = asyncio.Lock()
        key_lock = self._locks[key]

        async with key_lock:
            async with self._get_lock():
                entry = self._cache.get(key)

                if entry is None:
                    self._misses += 1
                    return None

                if entry.is_expired:
                    self._remove_entry(key, prompt_hash)
                    self._misses += 1
                    return None

                # LRU：移到末尾
                self._cache.move_to_end(key)
                entry.access_count += 1
                self._hits += 1

                # 空值缓存：返回None表示之前查询过但无结果，避免穿透
                if entry.result is self._NULL_MARKER:
                    return None

                return entry.result

    async def set(
        self,
        prompt_hash: str,
        cache_key: str,
        result: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """写入缓存

        Args:
            prompt_hash: prompt的hash值
            cache_key: 缓存复合键
            result: 要缓存的结果，None时存储空值标记防止穿透
            ttl: 缓存TTL（秒），None则使用默认值
        """
        async with self._get_lock():
            key = self._full_key(prompt_hash, cache_key)
            effective_ttl = ttl if ttl is not None else self._default_ttl

            # 空值缓存：value为None时存储NULL_MARKER，TTL为正常值的1/5
            if result is None:
                result = self._NULL_MARKER
                effective_ttl = effective_ttl / 5

            # TTL抖动：±10%随机偏移，防止大量缓存同时过期（雪崩）
            jitter = effective_ttl * random.uniform(-0.1, 0.1)
            effective_ttl = effective_ttl + jitter

            # 如果已存在，先移除旧条目
            if key in self._cache:
                self._remove_entry(key, prompt_hash)

            # LRU淘汰
            while len(self._cache) >= self._max_size:
                self._evict_one()

            entry = CacheEntry(
                result=result,
                created_at=time.time(),
                ttl=effective_ttl,
            )
            self._cache[key] = entry

            # 更新prompt索引
            if prompt_hash not in self._prompt_index:
                self._prompt_index[prompt_hash] = set()
            self._prompt_index[prompt_hash].add(key)

    def _remove_entry(self, key: str, prompt_hash: str) -> None:
        """移除缓存条目（内部方法，调用方需持有锁）"""
        self._cache.pop(key, None)
        # 清理该key对应的互斥锁
        self._locks.pop(key, None)
        if prompt_hash in self._prompt_index:
            self._prompt_index[prompt_hash].discard(key)
            if not self._prompt_index[prompt_hash]:
                del self._prompt_index[prompt_hash]

    def _evict_one(self) -> None:
        """LRU淘汰一个最旧的条目（内部方法，调用方需持有锁）"""
        if not self._cache:
            return
        # OrderedDict中第一个元素是最旧的
        oldest_key, _ = next(iter(self._cache.items()))
        # 解析prompt_hash
        prompt_hash = oldest_key.split("::")[0]
        self._remove_entry(oldest_key, prompt_hash)

    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """手动失效缓存

        Args:
            pattern: 失效模式。如果提供，则移除prompt_hash包含该模式的条目；
                     如果为None，则清空所有缓存。

        Returns:
            被移除的条目数
        """
        async with self._get_lock():
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                self._prompt_index.clear()
                self._locks.clear()
                return count

            # 按pattern匹配prompt_hash
            keys_to_remove: list[tuple[str, str]] = []
            for prompt_hash, keys in self._prompt_index.items():
                if pattern in prompt_hash:
                    for key in keys:
                        keys_to_remove.append((key, prompt_hash))

            for key, prompt_hash in keys_to_remove:
                self._remove_entry(key, prompt_hash)

            return len(keys_to_remove)

    async def cleanup_expired(self) -> int:
        """清理过期的缓存条目

        Returns:
            被清理的条目数
        """
        async with self._get_lock():
            keys_to_remove: list[tuple[str, str]] = []
            for key, entry in self._cache.items():
                if entry.is_expired:
                    prompt_hash = key.split("::")[0]
                    keys_to_remove.append((key, prompt_hash))

            for key, prompt_hash in keys_to_remove:
                self._remove_entry(key, prompt_hash)

            if keys_to_remove:
                logger.info("清理过期缓存条目: %d 条", len(keys_to_remove))

            return len(keys_to_remove)

    @property
    def stats(self) -> dict[str, Any]:
        """获取缓存统计信息

        Returns:
            包含命中次数、未命中次数、命中率、当前大小的字典
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self._max_size,
        }
