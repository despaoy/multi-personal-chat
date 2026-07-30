"""
Redis Streams 消息队列 - 异步消息处理

替代内存 asyncio.PriorityQueue，提供：
- 消息持久化（Redis AOF）
- 优先级流（0-10，0=最高）
- 消费者组负载均衡
- 死信队列
- 背压控制
- Redis不可用时回退到内存队列
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Callable, Awaitable, Any, List

import redis.asyncio as aioredis

from interfaces import MessageQueueInterface

logger = logging.getLogger(__name__)

# Stream naming: mq:priority:{0-10}
# Consumer group: mq_workers
# Dead letter: mq:dead_letter
MAX_PRIORITY = 10
CONSUMER_GROUP = "mq_workers"
VISIBILITY_TIMEOUT = 30  # seconds
MAX_PENDING = 500
MAX_RETRY = 3


@dataclass
class QueueMessage:
    """Redis Streams 消息格式"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    group_id: str = ""
    user_id: str = ""
    message: str = ""
    priority: int = 10
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0

    def to_dict(self) -> dict:
        """序列化为 Redis Stream 兼容的 str-str 字典"""
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> "QueueMessage":
        """从 Redis Stream 数据反序列化"""
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            group_id=data.get("group_id", ""),
            user_id=data.get("user_id", ""),
            message=data.get("message", ""),
            priority=int(data.get("priority", 10)),
            created_at=float(data.get("created_at", time.time())),
            retry_count=int(data.get("retry_count", 0)),
        )


class RedisMessageQueue:
    """Redis Streams 消息队列"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        consumer_name: Optional[str] = None,
    ):
        self._redis_url = redis_url
        self._consumer_name = consumer_name or f"worker-{uuid.uuid4().hex[:8]}"
        self._client: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()
        self._running = False
        self._fallback_queue: Optional[asyncio.PriorityQueue] = None
        self._use_redis = True
        self._groups_created = False
        self._stats: Dict[str, int] = {
            "enqueued": 0,
            "dequeued": 0,
            "dead_lettered": 0,
            "claimed": 0,
            "rejected_backpressure": 0,
            "fallback_enqueued": 0,
        }

    # ------------------------------------------------------------------
    # Redis client lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> Optional[aioredis.Redis]:
        """Lazy init Redis client; returns None (and activates fallback) on failure."""
        if self._client is not None:
            return self._client

        async with self._lock:
            # Double-check after acquiring lock
            if self._client is not None:
                return self._client

            try:
                # 复用 cache.redis_client 的共享连接池，避免独立创建连接池
                from cache.redis_client import get_async_redis
                self._client = await get_async_redis()
                if self._client is None:
                    raise RuntimeError("Redis 不可用")
                await self._client.ping()
                await self._create_consumer_groups()
                self._use_redis = True
                logger.info("Redis消息队列已连接(共享连接池): %s", self._redis_url)
                return self._client
            except Exception as e:
                logger.warning("Redis不可用，回退到内存队列: %s", e)
                self._use_redis = False
                if self._fallback_queue is None:
                    self._fallback_queue = asyncio.PriorityQueue(maxsize=MAX_PENDING)
                # 注意：共享客户端由 cache.redis_client 管理生命周期，此处不关闭
                self._client = None
                return None

    async def _create_consumer_groups(self) -> None:
        """Create consumer groups for all priority streams and the dead-letter stream."""
        if self._groups_created or self._client is None:
            return
        for pri in range(MAX_PRIORITY + 1):
            stream = f"mq:priority:{pri}"
            try:
                await self._client.xgroup_create(
                    stream, CONSUMER_GROUP, id="0", mkstream=True
                )
            except aioredis.ResponseError as e:
                # BUSYGROUP: consumer group already exists — harmless
                if "BUSYGROUP" not in str(e):
                    logger.debug("xgroup_create %s: %s", stream, e)
        try:
            await self._client.xgroup_create(
                "mq:dead_letter", CONSUMER_GROUP, id="0", mkstream=True
            )
        except aioredis.ResponseError:
            pass
        self._groups_created = True

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    async def enqueue(
        self, group_id: str, user_id: str, message: str, priority: int = 10
    ) -> bool:
        """入队消息。返回 True 表示成功，False 表示队列满或入队失败。"""
        priority = max(0, min(priority, MAX_PRIORITY))
        msg = QueueMessage(
            group_id=group_id, user_id=user_id, message=message, priority=priority
        )
        client = await self._ensure_client()

        if client is not None and self._use_redis:
            try:
                # Backpressure check
                total_pending = 0
                for pri in range(MAX_PRIORITY + 1):
                    total_pending += await client.xlen(f"mq:priority:{pri}")
                if total_pending >= MAX_PENDING:
                    logger.warning(
                        "消息队列已满 (%d/%d)，拒绝入队", total_pending, MAX_PENDING
                    )
                    self._stats["rejected_backpressure"] += 1
                    return False

                stream = f"mq:priority:{priority}"
                await client.xadd(stream, msg.to_dict())
                self._stats["enqueued"] += 1
                return True
            except Exception as e:
                logger.warning("Redis入队失败，回退到内存: %s", e)
                self._use_redis = False
                if self._fallback_queue is None:
                    self._fallback_queue = asyncio.PriorityQueue(maxsize=MAX_PENDING)

        # Fallback: in-memory priority queue
        try:
            self._fallback_queue.put_nowait(
                (priority, time.monotonic(), msg)
            )
            self._stats["fallback_enqueued"] += 1
            return True
        except asyncio.QueueFull:
            self._stats["rejected_backpressure"] += 1
            return False

    # ------------------------------------------------------------------
    # Dequeue
    # ------------------------------------------------------------------

    async def dequeue(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        """从优先级流中取出消息（0=最高优先级最先消费）。

        当 Redis 可用时使用 XREADGROUP + consumer group；
        否则从内存 asyncio.PriorityQueue 取。
        """
        client = await self._ensure_client()

        if client is not None and self._use_redis:
            try:
                # Iterate priority streams from highest (0) to lowest (10)
                for pri in range(MAX_PRIORITY + 1):
                    stream = f"mq:priority:{pri}"
                    # Only block on the first (highest priority) stream read;
                    # subsequent streams use non-blocking reads.
                    block_ms = int(timeout * 1000) if pri == 0 else 0
                    messages = await client.xreadgroup(
                        CONSUMER_GROUP,
                        self._consumer_name,
                        {stream: ">"},
                        count=1,
                        block=block_ms,
                    )
                    if messages:
                        # messages: [(stream_name, [(msg_id, data), ...])]
                        for stream_name, msg_list in messages:
                            for msg_id, data in msg_list:
                                # Auto-ACK
                                await client.xack(stream_name, CONSUMER_GROUP, msg_id)
                                self._stats["dequeued"] += 1
                                return QueueMessage.from_dict(data)
                return None
            except Exception as e:
                logger.warning("Redis出队失败: %s", e)
                self._use_redis = False

        # Fallback: in-memory queue
        if self._fallback_queue is not None:
            try:
                _, _, msg = await asyncio.wait_for(
                    self._fallback_queue.get(), timeout=timeout
                )
                self._stats["dequeued"] += 1
                return msg
            except asyncio.TimeoutError:
                return None
        return None

    # ------------------------------------------------------------------
    # Dead letter queue
    # ------------------------------------------------------------------

    async def move_to_dead_letter(self, msg: QueueMessage, reason: str = "") -> None:
        """移动失败消息到死信队列。"""
        msg.retry_count += 1
        data = msg.to_dict()
        data["dead_reason"] = reason
        data["dead_at"] = str(time.time())

        client = await self._ensure_client()
        if client is not None and self._use_redis:
            try:
                await client.xadd("mq:dead_letter", data)
                self._stats["dead_lettered"] += 1
                logger.warning(
                    "消息移入死信队列: %s, 原因: %s, 重试次数: %d",
                    msg.id,
                    reason,
                    msg.retry_count,
                )
            except Exception as e:
                logger.error("移入死信队列失败: %s", e)
        else:
            logger.warning(
                "Redis不可用，消息 %s 丢失 (dead_letter): %s", msg.id, reason
            )

    # ------------------------------------------------------------------
    # Claim pending (visibility timeout)
    # ------------------------------------------------------------------

    async def claim_pending_messages(self) -> int:
        """认领超过 visibility timeout 的 pending 消息，重新入队或送入死信。

        返回本次重新入队的消息数量。
        """
        client = await self._ensure_client()
        if client is None or not self._use_redis:
            return 0

        reclaimed = 0
        try:
            for pri in range(MAX_PRIORITY + 1):
                stream = f"mq:priority:{pri}"
                try:
                    pending = await client.xpending_range(
                        stream,
                        CONSUMER_GROUP,
                        min="-",
                        max="+",
                        count=50,
                    )
                except (aioredis.ResponseError, AttributeError, TypeError):
                    # Stream has no pending entries or command not supported
                    continue

                if not pending:
                    continue

                now_ms = time.time() * 1000
                stale_ids: List[bytes] = []
                for p in pending:
                    # p.time_since_delivered is in milliseconds
                    delivered_ms = getattr(p, "time_since_delivered", None) or 0
                    if delivered_ms > VISIBILITY_TIMEOUT * 1000:
                        msg_id = p.message_id
                        if isinstance(msg_id, str):
                            msg_id = msg_id.encode()
                        stale_ids.append(msg_id)

                if not stale_ids:
                    continue

                # Claim the stale messages
                try:
                    claimed = await client.xclaim(
                        stream,
                        CONSUMER_GROUP,
                        self._consumer_name,
                        min_idle_time=VISIBILITY_TIMEOUT * 1000,
                        message_ids=stale_ids,
                    )
                except (aioredis.ResponseError, AttributeError, TypeError) as e:
                    logger.debug("xclaim error on %s: %s", stream, e)
                    continue

                # claimed: list of (msg_id, data_dict)
                for msg_id, data in claimed:
                    await client.xack(stream, CONSUMER_GROUP, msg_id)
                    msg = QueueMessage.from_dict(data)
                    if msg.retry_count >= MAX_RETRY:
                        await self.move_to_dead_letter(msg, "max_retry_exceeded")
                    else:
                        msg.retry_count += 1
                        await self.enqueue(
                            msg.group_id,
                            msg.user_id,
                            msg.message,
                            msg.priority,
                        )
                        reclaimed += 1

                self._stats["claimed"] += len(claimed)
        except Exception as e:
            logger.debug("claim_pending_messages error: %s", e)

        return reclaimed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息。"""
        stats: Dict[str, Any] = dict(self._stats)

        client = await self._ensure_client()
        if client is not None and self._use_redis:
            try:
                total = 0
                per_priority: Dict[str, int] = {}
                for pri in range(MAX_PRIORITY + 1):
                    length = await client.xlen(f"mq:priority:{pri}")
                    per_priority[f"priority_{pri}"] = length
                    total += length
                stats["redis_queue_size"] = total
                stats["per_priority"] = per_priority
                stats["dead_letter_count"] = await client.xlen("mq:dead_letter")
                stats["mode"] = "redis"
            except Exception:
                stats["mode"] = "memory"
        else:
            stats["memory_queue_size"] = (
                self._fallback_queue.qsize() if self._fallback_queue else 0
            )
            stats["mode"] = "memory"

        stats["consumer_name"] = self._consumer_name
        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_claim_loop(self, interval: float = 10.0) -> None:
        """后台定期认领超时 pending 消息。调用 close() 可停止。"""
        self._running = True
        while self._running:
            try:
                await self.claim_pending_messages()
            except Exception as e:
                logger.debug("claim loop error: %s", e)
            await asyncio.sleep(interval)

    async def close(self) -> None:
        """停止后台任务。

        注意：共享 Redis 客户端由 cache.redis_client.close_async_redis() 统一管理，
        此处仅清空本地引用，不调用 aclose()，否则会关闭其他模块（semantic_cache 等）
        正在使用的共享客户端。
        """
        self._running = False
        self._client = None
        logger.info("Redis消息队列已停止（共享客户端由 redis_client 统一关闭）")


# 接口契约验证：确保 RedisMessageQueue 实现 MessageQueueInterface 接口
assert isinstance(RedisMessageQueue(), MessageQueueInterface), f"RedisMessageQueue must implement MessageQueueInterface"
