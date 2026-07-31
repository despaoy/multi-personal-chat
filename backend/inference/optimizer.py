"""
LLM调用优化模块 - API优化、响应缓存、请求限流一体化

提供以下核心组件：
- LLMCallOptimizer: API调用优化（重试、连接池、超时、并发控制）
- ResponseCache: 基于语义相似度的响应缓存
- RateLimiter: 令牌桶算法请求限流
- PromptOptimizer: Prompt优化与对话历史压缩
"""

import asyncio
import hashlib
import heapq
import logging
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类与枚举
# ---------------------------------------------------------------------------

class TimeoutLevel(Enum):
    """请求超时分级"""
    FAST = 5        # 快速请求（如简单问答）
    NORMAL = 30     # 普通请求
    LONG_TEXT = 120  # 长文本生成


class UserRole(Enum):
    """用户角色，不同角色对应不同限流策略"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_USER = "api_user"


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True


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


@dataclass
class TokenBucket:
    """令牌桶"""
    rate: float           # 每秒补充令牌数
    capacity: float       # 桶容量
    tokens: float = 0.0   # 当前令牌数
    last_refill: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def refill(self) -> None:
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        self.last_access = now

    def consume(self, tokens: int = 1) -> bool:
        """尝试消费令牌，成功返回True"""
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def can_consume(self, tokens: int = 1) -> bool:
        """检查是否有足够令牌可消费（不实际消耗）"""
        self.refill()
        return self.tokens >= tokens

    @property
    def wait_time(self) -> float:
        """等待一个令牌所需的时间（秒）"""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.rate


# ---------------------------------------------------------------------------
# 角色限流配置
# ---------------------------------------------------------------------------

ROLE_RATE_LIMITS: dict[UserRole, dict[str, float]] = {
    UserRole.ADMIN: {"rate": 30, "capacity": 180},
    UserRole.OPERATOR: {"rate": 20, "capacity": 120},
    UserRole.VIEWER: {"rate": 10, "capacity": 60},
    UserRole.API_USER: {"rate": 5, "capacity": 30},
}


# ---------------------------------------------------------------------------
# PromptOptimizer - Prompt优化器
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """Prompt优化器：自动压缩对话历史、优化system prompt、估算token消耗"""

    # 中文约1.5字/token，英文约4字符/token，取混合估算系数
    _CN_CHARS_PER_TOKEN = 1.5
    _EN_CHARS_PER_TOKEN = 4.0

    # 任务类型到system prompt的映射
    _TASK_SYSTEM_PROMPTS: dict[str, str] = {
        "chat": "你是一个友好的AI助手，请用简洁准确的方式回答用户问题。",
        "qa": "你是一个知识问答助手，请基于已知信息准确回答问题。如果不确定，请如实说明。",
        "creative": "你是一个创意写作助手，请发挥想象力，生成富有创意的内容。",
        "code": "你是一个编程助手，请提供准确、高效、可运行的代码解决方案。",
        "summary": "你是一个摘要助手，请提取关键信息，生成简洁的摘要。",
    }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的token消耗

        基于字符数估算：中文约1.5字/token，英文约4字符/token。
        通过统计中文字符占比进行加权估算。

        Args:
            text: 待估算的文本

        Returns:
            估算的token数量
        """
        if not text:
            return 0

        cn_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        en_chars = len(text) - cn_chars

        cn_tokens = cn_chars / PromptOptimizer._CN_CHARS_PER_TOKEN
        en_tokens = en_chars / PromptOptimizer._EN_CHARS_PER_TOKEN

        return max(1, int(cn_tokens + en_tokens))

    @classmethod
    def get_system_prompt(cls, task_type: str) -> str:
        """根据任务类型获取最优system prompt

        Args:
            task_type: 任务类型（chat/qa/creative/code/summary）

        Returns:
            对应的system prompt，未知类型返回通用prompt
        """
        return cls._TASK_SYSTEM_PROMPTS.get(
            task_type,
            "你是一个AI助手，请尽力帮助用户解决问题。"
        )

    @staticmethod
    def compress_history(
        messages: list[dict[str, str]],
        keep_recent: int = 4,
        max_summary_tokens: int = 200,
    ) -> list[dict[str, str]]:
        """压缩过长的对话历史

        保留最近N轮对话，将更早的对话压缩为摘要。
        如果没有system消息则插入一条通用system prompt。

        Args:
            messages: 消息列表，每条包含 role 和 content
            keep_recent: 保留最近N轮对话（一轮=一条user+一条assistant）
            max_summary_tokens: 摘要的最大token数估算

        Returns:
            压缩后的消息列表
        """
        if len(messages) <= keep_recent * 2 + 1:
            return messages

        # 分离system消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # 保留最近N轮
        recent_count = keep_recent * 2
        older = non_system[:-recent_count] if len(non_system) > recent_count else []
        recent = non_system[-recent_count:] if len(non_system) > recent_count else non_system

        if not older:
            return messages

        # 将更早的对话压缩为摘要
        summary_parts: list[str] = []
        for msg in older:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = "用户" if role == "user" else "助手"
            summary_parts.append(f"{prefix}: {content}")

        summary_text = " ".join(summary_parts)

        # 按估算token数截断摘要
        estimated = PromptOptimizer.estimate_tokens(summary_text)
        if estimated > max_summary_tokens:
            ratio = max_summary_tokens / estimated
            cut_len = int(len(summary_text) * ratio)
            summary_text = summary_text[:cut_len] + "..."

        summary_msg = {
            "role": "system",
            "content": f"[对话历史摘要] {summary_text}",
        }

        result = system_msgs + [summary_msg] + recent
        return result

    @staticmethod
    def truncate_prompt(
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """自动截断超长prompt，确保不超过模型context window

        从最早的非system消息开始截断，优先保留system消息和最近的消息。

        Args:
            messages: 消息列表
            max_tokens: 模型context window的token上限

        Returns:
            截断后的消息列表
        """
        total_tokens = sum(PromptOptimizer.estimate_tokens(m.get("content", "")) for m in messages)
        if total_tokens <= max_tokens:
            return messages

        # 分离system和非system消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # 保留system消息的token预算
        system_tokens = sum(PromptOptimizer.estimate_tokens(m.get("content", "")) for m in system_msgs)
        remaining = max_tokens - system_tokens

        # 从最早的非system消息开始移除
        while non_system and remaining <= 0:
            removed = non_system.pop(0)
            remaining += PromptOptimizer.estimate_tokens(removed.get("content", ""))

        # 重新计算非system消息的token
        ns_tokens = sum(PromptOptimizer.estimate_tokens(m.get("content", "")) for m in non_system)
        while non_system and ns_tokens > remaining:
            removed = non_system.pop(0)
            ns_tokens -= PromptOptimizer.estimate_tokens(removed.get("content", ""))

        return system_msgs + non_system

    @classmethod
    def optimize_prompt(
        cls,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        task_type: str = "chat",
        keep_recent: int = 4,
    ) -> list[dict[str, str]]:
        """优化prompt：压缩历史、截断超长内容、优化system prompt

        Args:
            messages: 原始消息列表
            max_tokens: 模型context window的token上限
            task_type: 任务类型，用于选择最优system prompt
            keep_recent: 压缩历史时保留的最近轮数

        Returns:
            优化后的消息列表
        """
        # 如果没有system消息，根据任务类型添加
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": cls.get_system_prompt(task_type)}] + messages

        # 压缩对话历史
        messages = cls.compress_history(messages, keep_recent=keep_recent)

        # 截断超长prompt
        messages = cls.truncate_prompt(messages, max_tokens=max_tokens)

        return messages


# ---------------------------------------------------------------------------
# ResponseCache - 响应缓存
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RateLimiter - 请求限流器
# ---------------------------------------------------------------------------

class RateLimiter:
    """令牌桶算法请求限流器

    支持：
    - 每个用户/API Key独立的令牌桶
    - 不同角色的不同限流策略
    - 全局限流：保护后端总QPS不超过阈值
    - 自动清理过期的令牌桶（超过1小时无请求）
    """

    # 过期桶的清理阈值（秒）
    _BUCKET_EXPIRY_SECONDS = 3600

    def __init__(
        self,
        default_rate: float = 10,
        default_capacity: float = 60,
        global_max_qps: float = 100,
    ) -> None:
        """初始化限流器

        Args:
            default_rate: 默认令牌补充速率（每秒）
            default_capacity: 默认桶容量
            global_max_qps: 全局最大QPS阈值
        """
        self._default_rate = default_rate
        self._default_capacity = default_capacity
        self._global_max_qps = global_max_qps

        self._buckets: dict[str, TokenBucket] = {}
        self._global_bucket = TokenBucket(rate=global_max_qps, capacity=global_max_qps, tokens=global_max_qps)
        self._lock = asyncio.Lock()

        # 统计
        self._rejected_count = 0

    def _get_bucket_config(self, role: Optional[UserRole] = None) -> dict[str, float]:
        """获取角色对应的桶配置

        Args:
            role: 用户角色，None则使用默认配置

        Returns:
            包含rate和capacity的配置字典
        """
        if role and role in ROLE_RATE_LIMITS:
            return ROLE_RATE_LIMITS[role]
        return {"rate": self._default_rate, "capacity": self._default_capacity}

    def _get_or_create_bucket(self, key: str, role: Optional[UserRole] = None) -> TokenBucket:
        """获取或创建令牌桶

        Args:
            key: 用户/API Key标识
            role: 用户角色

        Returns:
            对应的TokenBucket实例
        """
        if key not in self._buckets:
            config = self._get_bucket_config(role)
            self._buckets[key] = TokenBucket(
                rate=config["rate"],
                capacity=config["capacity"],
                tokens=config["capacity"],  # 初始满桶
            )
        return self._buckets[key]

    async def acquire(self, key: str, tokens: int = 1, role: Optional[UserRole] = None) -> bool:
        """尝试获取令牌（非阻塞）

        同时检查用户桶和全局限流桶。先检查两个桶的可用性，
        都可用时才实际消耗令牌，避免用户桶不足时全局桶已被消耗的问题。

        Args:
            key: 用户/API Key标识
            tokens: 请求消耗的令牌数
            role: 用户角色

        Returns:
            True表示获取成功，False表示被限流
        """
        async with self._lock:
            bucket = self._get_or_create_bucket(key, role)

            # 先检查两个桶的可用性
            if not self._global_bucket.can_consume(tokens):
                self._rejected_count += 1
                logger.warning("全局限流触发，当前QPS超限: key=%s", key)
                return False

            if not bucket.can_consume(tokens):
                self._rejected_count += 1
                logger.warning("用户限流触发: key=%s, 等待时间=%.2fs", key, bucket.wait_time)
                return False

            # 两个桶都可用，才实际消耗令牌
            self._global_bucket.consume(tokens)
            bucket.consume(tokens)

            return True

    async def wait_for_token(
        self,
        key: str,
        timeout: float = 30.0,
        tokens: int = 1,
        role: Optional[UserRole] = None,
    ) -> bool:
        """等待直到获取令牌或超时

        Args:
            key: 用户/API Key标识
            timeout: 最大等待时间（秒）
            tokens: 请求消耗的令牌数
            role: 用户角色

        Returns:
            True表示获取成功，False表示超时
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if await self.acquire(key, tokens=tokens, role=role):
                return True

            # 计算需要等待的时间
            async with self._lock:
                bucket = self._get_or_create_bucket(key, role)
                wait = min(bucket.wait_time, self._global_bucket.wait_time)
                wait = max(0.05, min(wait, 1.0))

            remaining = deadline - time.time()
            if remaining <= 0:
                break

            await asyncio.sleep(min(wait, remaining))

        return False

    async def cleanup_expired_buckets(self) -> int:
        """清理过期的令牌桶（超过1小时无请求的桶）

        Returns:
            被清理的桶数量
        """
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, bucket in self._buckets.items()
                if now - bucket.last_access > self._BUCKET_EXPIRY_SECONDS
            ]
            for key in expired_keys:
                del self._buckets[key]

            if expired_keys:
                logger.info("清理过期令牌桶: %d 个", len(expired_keys))

            return len(expired_keys)

    @property
    def stats(self) -> dict[str, Any]:
        """获取限流统计信息

        Returns:
            包含被限流次数、活跃桶数、全局当前令牌数的字典
        """
        return {
            "rejected_count": self._rejected_count,
            "active_buckets": len(self._buckets),
            "global_tokens": self._global_bucket.tokens,
            "global_capacity": self._global_bucket.capacity,
        }

    def get_bucket_stats(self, key: str) -> Optional[dict[str, Any]]:
        """获取指定用户的桶状态

        Args:
            key: 用户/API Key标识

        Returns:
            桶状态字典，不存在返回None
        """
        bucket = self._buckets.get(key)
        if bucket is None:
            return None
        bucket.refill()
        return {
            "tokens": bucket.tokens,
            "capacity": bucket.capacity,
            "rate": bucket.rate,
        }


# ---------------------------------------------------------------------------
# LLMCallOptimizer - LLM调用优化器
# ---------------------------------------------------------------------------

class LLMCallOptimizer:
    """LLM调用优化器

    集成以下优化能力：
    - 指数退避重试
    - 连接池复用（httpx.AsyncClient）
    - 请求超时分级
    - 请求预处理（截断超长prompt、压缩历史）
    - 响应后处理（去除空白、截断重复内容）
    - 并发控制（Semaphore）
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        max_concurrent: int = 10,
        max_connections: int = 50,
        max_keepalive: int = 20,
        cache: Optional[ResponseCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
        prompt_optimizer: Optional[PromptOptimizer] = None,
    ) -> None:
        """初始化LLM调用优化器

        Args:
            retry_config: 重试配置，None则使用默认值
            max_concurrent: 最大并发LLM请求数
            max_connections: 连接池最大连接数
            max_keepalive: 连接池最大keepalive连接数
            cache: 响应缓存实例，None则创建默认实例
            rate_limiter: 限流器实例，None则创建默认实例
            prompt_optimizer: Prompt优化器实例，None则创建默认实例
        """
        self._retry_config = retry_config or RetryConfig()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache = cache or ResponseCache()
        self._rate_limiter = rate_limiter or RateLimiter()
        self._prompt_optimizer = prompt_optimizer or PromptOptimizer()

        self._client: Optional[httpx.AsyncClient] = None
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive

        # 统计
        self._total_calls = 0
        self._failed_calls = 0

    async def _ensure_client(self) -> httpx.AsyncClient:
        """确保httpx.AsyncClient已创建"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive,
                ),
            )
        return self._client

    async def close(self) -> None:
        """关闭连接池，释放资源"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("LLM调用优化器连接池已关闭")

    def _get_timeout(self, timeout_level: TimeoutLevel) -> httpx.Timeout:
        """根据超时级别获取httpx.Timeout

        Args:
            timeout_level: 超时级别

        Returns:
            httpx.Timeout实例
        """
        return httpx.Timeout(timeout_level.value)

    @staticmethod
    def _compute_backoff_delay(
        attempt: int,
        config: RetryConfig,
    ) -> float:
        """计算指数退避延迟

        Args:
            attempt: 当前重试次数（从0开始）
            config: 重试配置

        Returns:
            延迟时间（秒）
        """
        delay = min(config.base_delay * (2 ** attempt), config.max_delay)
        if config.jitter:
            delay = delay * random.uniform(0.5, 1.5)
        return delay

    @staticmethod
    def _postprocess_response(text: str) -> str:
        """响应后处理：去除空白、截断重复内容

        Args:
            text: 原始响应文本

        Returns:
            处理后的文本
        """
        if not text:
            return text

        # 去除首尾空白
        text = text.strip()

        # 去除连续多个空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 检测并截断重复内容
        # 如果后半部分是前半部分的重复，则截断
        half_len = len(text) // 2
        if half_len > 50:
            first_half = text[:half_len]
            second_half = text[half_len:half_len * 2]
            if first_half == second_half:
                text = first_half.rstrip()

        return text

    async def call_with_retry(
        self,
        provider: str,
        prompt: str,
        *,
        model: str = "",
        lora_name: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout_level: TimeoutLevel = TimeoutLevel.NORMAL,
        messages: Optional[list[dict[str, str]]] = None,
        use_cache: bool = True,
        rate_limit_key: Optional[str] = None,
        rate_limit_role: Optional[UserRole] = None,
        **kwargs: Any,
    ) -> str:
        """带重试和优化的LLM调用

        完整流程：
        1. 检查缓存（可选）
        2. 限流检查
        3. Prompt优化
        4. 并发控制
        5. 指数退避重试
        6. 响应后处理
        7. 写入缓存

        Args:
            provider: LLM提供商标识（如 "openai", "local" 等）
            prompt: 用户prompt文本
            model: 模型名称
            lora_name: LoRA适配器名称
            temperature: 温度参数
            max_tokens: 最大token数
            timeout_level: 超时级别
            messages: 完整的消息列表（如提供则替代prompt参数）
            use_cache: 是否使用缓存
            rate_limit_key: 限流标识（用户/API Key）
            rate_limit_role: 限流角色
            **kwargs: 其他传递给LLM API的参数

        Returns:
            LLM生成的文本结果

        Raises:
            httpx.HTTPStatusError: API返回错误状态码（重试耗尽后）
            httpx.TimeoutException: 请求超时（重试耗尽后）
            RuntimeError: 被限流或所有重试失败
        """
        # 1. 构建消息列表
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # 2. Prompt优化
        optimized_messages = self._prompt_optimizer.optimize_prompt(
            messages, max_tokens=max_tokens
        )

        # 3. 缓存查询
        prompt_text = optimized_messages[-1].get("content", "") if optimized_messages else prompt
        prompt_hash = ResponseCache.compute_prompt_hash(prompt_text)
        cache_key = ResponseCache.build_cache_key(
            model_name=model, lora_name=lora_name, temperature=temperature
        )

        if use_cache:
            cached = await self._cache.get(prompt_hash, cache_key)
            if cached is not None:
                logger.debug("缓存命中: prompt_hash=%s", prompt_hash[:16])
                return cached

        # 4. 限流检查
        effective_key = rate_limit_key or "default"
        if not await self._rate_limiter.acquire(effective_key, role=rate_limit_role):
            raise RuntimeError(
                f"请求被限流，请稍后重试。key={effective_key}"
            )

        # 5. 并发控制 + 重试
        result_text = ""
        last_exception: Optional[Exception] = None

        async with self._semaphore:
            self._total_calls += 1

            for attempt in range(self._retry_config.max_retries + 1):
                try:
                    client = await self._ensure_client()
                    timeout = self._get_timeout(timeout_level)

                    # 构建请求体（通用格式，具体provider适配在外部处理）
                    request_body = {
                        "model": model,
                        "messages": optimized_messages,
                        "temperature": temperature,
                        **kwargs,
                    }

                    # 这里使用通用的API调用方式
                    # 实际项目中应根据provider选择不同的endpoint
                    response = await client.post(
                        provider,
                        json=request_body,
                        timeout=timeout,
                    )
                    response.raise_for_status()

                    data = response.json()
                    # 兼容不同API响应格式
                    if isinstance(data, dict):
                        result_text = (
                            data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            or data.get("response", "")
                            or data.get("text", "")
                            or str(data)
                        )
                    else:
                        result_text = str(data)

                    # 6. 响应后处理
                    result_text = self._postprocess_response(result_text)
                    last_exception = None
                    break

                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    last_exception = exc
                    if attempt < self._retry_config.max_retries:
                        delay = self._compute_backoff_delay(attempt, self._retry_config)
                        logger.warning(
                            "LLM调用失败(尝试 %d/%d)，%.1fs后重试: %s",
                            attempt + 1, self._retry_config.max_retries + 1,
                            delay, str(exc),
                        )
                        await asyncio.sleep(delay)
                    else:
                        self._failed_calls += 1
                        logger.error(
                            "LLM调用最终失败(共 %d 次尝试): %s",
                            self._retry_config.max_retries + 1, str(exc),
                        )

                except httpx.HTTPStatusError as exc:
                    # 429 Too Many Requests 可以重试
                    if exc.response.status_code == 429 and attempt < self._retry_config.max_retries:
                        delay = self._compute_backoff_delay(attempt, self._retry_config)
                        # 优先使用服务器建议的Retry-After
                        retry_after = exc.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                pass
                        logger.warning(
                            "收到429限流响应(尝试 %d/%d)，%.1fs后重试",
                            attempt + 1, self._retry_config.max_retries + 1, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # 5xx 服务器错误可以重试
                    if exc.response.status_code >= 500 and attempt < self._retry_config.max_retries:
                        delay = self._compute_backoff_delay(attempt, self._retry_config)
                        logger.warning(
                            "服务器错误 %d(尝试 %d/%d)，%.1fs后重试",
                            exc.response.status_code,
                            attempt + 1, self._retry_config.max_retries + 1, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # 4xx 客户端错误（非429）不重试
                    self._failed_calls += 1
                    raise

        if last_exception is not None:
            self._failed_calls += 1
            raise last_exception

        # 7. 写入缓存
        if use_cache and result_text:
            await self._cache.set(prompt_hash, cache_key, result_text)

        return result_text

    @property
    def stats(self) -> dict[str, Any]:
        """获取调用统计信息

        Returns:
            包含总调用次数、失败次数、缓存统计、限流统计的字典
        """
        return {
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "cache": self._cache.stats,
            "rate_limiter": self._rate_limiter.stats,
        }

    @property
    def cache(self) -> ResponseCache:
        """获取缓存实例"""
        return self._cache

    @property
    def rate_limiter(self) -> RateLimiter:
        """获取限流器实例"""
        return self._rate_limiter

    @property
    def prompt_optimizer(self) -> PromptOptimizer:
        """获取Prompt优化器实例"""
        return self._prompt_optimizer
