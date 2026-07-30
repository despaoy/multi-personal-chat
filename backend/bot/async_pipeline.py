#!/usr/bin/env python3
"""群级别令牌桶限流器。

历史上本模块曾包含 `AsyncMessagePipeline`（基于 Redis Streams 的异步消息处理
管道），但该类从未在生产路径被实例化——`bot/bot.py` 直接调用推理层，未走管道
抽象。AsyncMessagePipeline 连带 `MessageTask`/`_worker`/`_claim_loop` 已被
删除，仅保留 `GroupRateLimiter`（仍在测试与限流场景使用）。
"""
import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class GroupRateLimiter:
    """按群独立的令牌桶限流器"""

    def __init__(self, default_rate: float = 30.0, default_capacity: int = 60):
        self.default_rate = default_rate
        self.default_capacity = default_capacity
        self._buckets: Dict[str, tuple[float, float]] = {}

    def acquire(self, group_id: str) -> bool:
        now = time.monotonic()
        tokens, last_refill = self._buckets.get(group_id, (self.default_capacity, now))
        elapsed = now - last_refill
        refill = elapsed * self.default_rate
        tokens = min(self.default_capacity, tokens + refill)

        if tokens >= 1.0:
            self._buckets[group_id] = (tokens - 1.0, now)
            return True
        else:
            self._buckets[group_id] = (tokens, now)
            return False

    def cleanup(self, max_age: float = 3600.0):
        now = time.monotonic()
        stale = [gid for gid, (_, last) in self._buckets.items() if now - last > max_age]
        for gid in stale:
            del self._buckets[gid]
