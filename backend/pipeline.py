# -*- coding: utf-8 -*-
"""
QQ智能助手 - 高并发消息处理管道

整合架构（借鉴 vLLM/OpenAI 等主流大模型服务体系）:

  请求 → 缓存检查 → 按群限流 → 请求合并 → 有界队列 → 并发控制 → 模型推理 → 回复
                                     │                      │
                                     ▼                      ▼
                             相同消息合并         超限时优雅降级
                             (Coalescing)        (GracefulDegradation)

设计参考:
  - vLLM: Continuous Batching → 请求合并 Coalescing
  - OpenAI: Rate Limiting (RPM/TPM) → 按群限流 + 全局限流
  - Anthropic: Request Queuing → BoundedPriorityQueue
  - 主流做法: Graceful Degradation → 缓存降级 → 排队 → 拒绝
"""
import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 优雅降级策略
# ══════════════════════════════════════════════════════════════════

class DegradationLevel(Enum):
    """降级级别"""
    FULL = 0        # 完整推理
    CACHED = 1      # 缓存回复
    QUEUED = 2      # 排队等待
    REJECTED = 3    # 拒绝请求


@dataclass
class DegradationResult:
    level: DegradationLevel
    reply: str = ""
    model: str = "unknown"
    cost_time: float = 0.0
    queue_position: int = 0
    estimated_wait: float = 0.0


# ══════════════════════════════════════════════════════════════════
# 请求合并器 (Request Coalescer)
# ══════════════════════════════════════════════════════════════════

class RequestCoalescer:
    """
    请求合并器 - 借鉴 vLLM Continuous Batching 思想

    当多个用户在短时间内发送相同/相似的 prompt 时，
    合并为一次推理，将结果分发给所有等待者。

    与缓存不同：
    - 缓存: 第一次请求完成后，后续请求命中
    - 合并: 并发中的相同请求，只执行一次

    实现：
    - 维护一个 pending 字典: prompt_hash → list[asyncio.Future]
    - 第一个到达的请求执行推理
    - 后续相同 hash 的请求等待 first 的结果
    - 窗口时间内的请求自动合并
    """

    def __init__(self, window_ms: int = 500):
        """
        Args:
            window_ms: 合并窗口（毫秒），此时间内的相同请求会被合并
        """
        self._window_ms = window_ms
        self._pending: Dict[str, Dict] = {}  # prompt_hash → info dict
        self._lock = asyncio.Lock()
        self._stats = {"merged": 0, "total": 0}

    async def submit(
        self,
        prompt: str,
        session_id: str,
        do_inference: Callable[[], Any],
    ) -> Any:
        """
        提交请求，自动合并相同 prompt 的并发请求。

        Args:
            prompt: 用户消息文本
            session_id: 会话标识（用于区分缓存键）
            do_inference: 实际推理函数 async callable

        Returns:
            推理结果
        """
        prompt_hash = hashlib.sha256(
            f"{prompt}:{session_id[:8]}".encode()
        ).hexdigest()

        async with self._lock:
            self._stats["total"] += 1

            if prompt_hash in self._pending:
                # 已有相同请求在处理中，等待结果
                self._stats["merged"] += 1
                future: asyncio.Future = asyncio.get_event_loop().create_future()
                self._pending[prompt_hash]["waiters"].append(future)
                # 释放锁后等待
                pass
            else:
                # 第一个请求，创建 pending entry
                self._pending[prompt_hash] = {
                    "waiters": [],
                    "created_at": time.time(),
                }

        if prompt_hash in self._pending and "future" not in str(type(self._pending.get(prompt_hash, {}).get("waiters", []))):
            # 需要重新获取锁来检查
            pass

        # 检查是否应该等待或执行
        async with self._lock:
            entry = self._pending.get(prompt_hash)
            if entry is None:
                # 已经被处理完了，重新提交
                self._stats["total"] -= 1
                pass  # fall through to create new

        # 检查我们是否是 waiter 还是 first
        async with self._lock:
            entry = self._pending.get(prompt_hash)
            if entry is None:
                # 创建新 entry
                self._pending[prompt_hash] = {
                    "waiters": [],
                    "created_at": time.time(),
                    "result": None,
                    "done": False,
                }

        # 确定角色
        is_first = False
        future = None
        async with self._lock:
            entry = self._pending[prompt_hash]
            if entry.get("done"):
                # 已经有结果，直接返回
                return entry["result"]

            if entry.get("processing"):
                # 正在处理中，创建 waiter future
                future = asyncio.get_event_loop().create_future()
                entry["waiters"].append(future)
            else:
                # 我们是第一个
                entry["processing"] = True
                is_first = True

        if future is not None:
            # 等待第一个请求的结果
            try:
                return await asyncio.wait_for(future, timeout=60.0)
            except asyncio.TimeoutError:
                raise RuntimeError("等待合并结果超时")

        if is_first:
            try:
                result = await do_inference()
                # 通知所有等待者
                async with self._lock:
                    entry = self._pending.pop(prompt_hash, {})
                    entry["result"] = result
                    entry["done"] = True
                    for w in entry.get("waiters", []):
                        if not w.done():
                            w.set_result(result)
                return result
            except Exception as e:
                # 通知所有等待者失败
                async with self._lock:
                    entry = self._pending.pop(prompt_hash, {})
                    for w in entry.get("waiters", []):
                        if not w.done():
                            w.set_exception(e)
                raise

        # Fallback: 不应该到达这里
        async with self._lock:
            self._pending.pop(prompt_hash, None)
        from inference.model_manager import get_model_manager
        # M3 fix: 直接 await async_generate，避免 asyncio.to_thread → asyncio.run
        # 跨事件循环复用 AsyncClient 导致 "Event loop is closed"
        return await get_model_manager().async_generate(prompt=prompt)

    @property
    def stats(self):
        return dict(self._stats)


# ══════════════════════════════════════════════════════════════════
# 有界优先队列 (Bounded Priority Queue)
# ══════════════════════════════════════════════════════════════════

class BoundedQueue:
    """
    有界优先队列 - 借鉴业界 LLM API 排队机制

    特性:
    - 优先级排序（紧急消息优先）
    - 容量上限（防止内存溢出）
    - 排队位置和预估等待时间反馈
    - 超时自动丢弃
    """

    def __init__(self, max_size: int = 200, request_timeout: float = 120.0):
        """
        Args:
            max_size: 队列最大容量
            request_timeout: 请求在队列中的最大等待时间（秒）
        """
        self._max_size = max_size
        self._request_timeout = request_timeout
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._avg_process_time = 10.0  # 初始估计值，动态更新

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def available_slots(self) -> int:
        return self._max_size - self._queue.qsize()

    @property
    def estimated_wait(self) -> float:
        """预估等待时间（秒）"""
        return self._queue.qsize() * self._avg_process_time

    def update_avg_time(self, process_time: float):
        """更新平均处理时间（指数移动平均）"""
        self._avg_process_time = 0.8 * self._avg_process_time + 0.2 * process_time

    async def put(self, priority: int, item: Any) -> bool:
        """放入队列（非阻塞），失败返回 False"""
        try:
            self._queue.put_nowait((priority, time.time(), item))
            return True
        except asyncio.QueueFull:
            return False

    async def get(self) -> Any:
        """取出队列中的下一个项目"""
        priority, enqueue_time, item = await self._queue.get()
        wait_time = time.time() - enqueue_time
        if wait_time > self._request_timeout:
            logger.warning(f"队列中的请求超时 ({wait_time:.0f}s)，丢弃")
            self._queue.task_done()
            return None
        return item

    def task_done(self):
        self._queue.task_done()


# ══════════════════════════════════════════════════════════════════
# 统一消息处理管道
# ══════════════════════════════════════════════════════════════════

class MessagePipeline:
    """
    统一消息处理管道 - 整合所有高并发组件

    流程:
    1. 缓存检查 (ResponseCache)
    2. 按群限流 (Per-Group Rate Limiter)
    3. 全局限流 (Global Rate Limiter)
    4. 请求合并 (RequestCoalescer - 相同 prompt 并发合并)
    5. 排队 (BoundedQueue - 有界等待队列)
    6. 并发控制 (Semaphore)
    7. 模型推理 (ModelManager)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_queue_size: int = 200,
        group_rate: float = 3.0,  # 每群每秒最多 3 条
        global_rate: float = 30.0,  # 全局每秒最多 30 条
    ):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue = BoundedQueue(max_size=max_queue_size)
        self._coalescer = RequestCoalescer(window_ms=300)
        self._group_buckets: Dict[str, tuple[float, float]] = {}  # group → (tokens, last_refill)
        self._global_tokens = global_rate
        self._global_last_refill = time.monotonic()
        self._group_rate = group_rate
        self._global_rate = global_rate

        # 统计
        self._stats = {
            "total": 0, "cached": 0, "coalesced": 0,
            "queued": 0, "group_limited": 0, "global_limited": 0,
            "semaphore_timeout": 0, "failed": 0, "success": 0,
        }

    def _group_limiter_acquire(self, group_id: str) -> bool:
        """按群令牌桶限流"""
        now = time.monotonic()
        tokens, last_refill = self._group_buckets.get(
            group_id, (self._group_rate * 2, now)
        )
        elapsed = now - last_refill
        tokens = min(self._group_rate * 5, tokens + elapsed * self._group_rate)
        if tokens >= 1.0:
            self._group_buckets[group_id] = (tokens - 1.0, now)
            return True
        self._group_buckets[group_id] = (tokens, now)
        return False

    def _global_limiter_acquire(self) -> bool:
        """全局限流"""
        now = time.monotonic()
        elapsed = now - self._global_last_refill
        self._global_tokens = min(
            self._global_rate,
            self._global_tokens + elapsed * (self._global_rate / 2)
        )
        self._global_last_refill = now
        if self._global_tokens >= 1.0:
            self._global_tokens -= 1.0
            return True
        return False

    async def process(
        self,
        message: str,
        session_id: str,
        do_inference,
        session_type: str = "group",
        user_id: str = "",
        use_cache_check=None,
    ) -> DegradationResult:
        """处理一条消息，返回降级结果"""
        self._stats["total"] += 1

        # 1. 缓存检查
        if use_cache_check:
            cached = await use_cache_check()
            if cached:
                self._stats["cached"] += 1
                return DegradationResult(
                    level=DegradationLevel.CACHED,
                    reply=cached.get("reply", ""),
                    model=cached.get("model", "cached"),
                    cost_time=0.0,
                )

        # 2. 按群限流
        if session_type == "group":
            if not self._group_limiter_acquire(session_id or user_id):
                self._stats["group_limited"] += 1
                return DegradationResult(
                    level=DegradationLevel.REJECTED,
                    reply="本群消息过于频繁，请稍后再试",
                    model="rate_limited",
                )

        # 3. 全局限流
        if not self._global_limiter_acquire():
            self._stats["global_limited"] += 1
            # 全局超限进入排队
            queued = await self._queue.put(10, (message, session_id, user_id, do_inference))
            if queued:
                self._stats["queued"] += 1
                return DegradationResult(
                    level=DegradationLevel.QUEUED,
                    queue_position=self._queue.size,
                    estimated_wait=self._queue.estimated_wait,
                )
            else:
                return DegradationResult(
                    level=DegradationLevel.REJECTED,
                    reply="服务繁忙，请稍后重试",
                    model="queue_full",
                )

        # 4. 请求合并 + 并发控制
        sem_acquired = False
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=30.0)
            sem_acquired = True
        except asyncio.TimeoutError:
            # 并发满 → 尝试排队
            self._stats["semaphore_timeout"] += 1
            queued = await self._queue.put(5, (message, session_id, user_id, do_inference))
            if queued:
                self._stats["queued"] += 1
                return DegradationResult(
                    level=DegradationLevel.QUEUED,
                    queue_position=self._queue.size,
                    estimated_wait=self._queue.estimated_wait,
                )
            else:
                return DegradationResult(
                    level=DegradationLevel.REJECTED,
                    reply="服务繁忙，请稍后重试",
                    model="busy",
                )

        try:
            # 使用合并器（相同 prompt 的并发请求只执行一次推理）
            result = await self._coalescer.submit(
                message, session_id, do_inference
            )
            # result 是 (reply, cost_time) tuple 或 DegradationResult
            if isinstance(result, tuple):
                reply, cost = result
                self._stats["success"] += 1
                self._queue.update_avg_time(cost)
                return DegradationResult(
                    level=DegradationLevel.FULL,
                    reply=reply,
                    cost_time=cost,
                )
            return result
        except Exception:
            self._stats["failed"] += 1
            return DegradationResult(
                level=DegradationLevel.REJECTED,
                reply="AI 回复生成失败，请稍后重试",
                model="error",
            )
        finally:
            if sem_acquired:
                self._semaphore.release()

    @property
    def stats(self):
        return dict(self._stats)


# 全局单例
_pipeline: Optional[MessagePipeline] = None


def get_pipeline(
    max_concurrent: int = 10,
    max_queue_size: int = 200,
) -> MessagePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MessagePipeline(
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
        )
    return _pipeline
