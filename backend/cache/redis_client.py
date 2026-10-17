"""Shared asynchronous Redis resources for message deduplication."""
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ============================================
# 连接配置
# ============================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Shared asynchronous resources used by NoneBot message deduplication.
_async_pool: Any = None
_async_client: Any = None
_async_lock: Any = None


def _get_async_pool():
    """获取或创建异步 Redis 连接池（redis.asyncio.ConnectionPool）。

    异步客户端必须使用异步连接池：将同步 pool 传给 redis.asyncio.Redis
    不会在构造时报错，但所有命令都会在事件循环中阻塞，导致缓存、消息去重
    等异步功能持续退化。
    """
    global _async_pool
    if _async_pool is None:
        import redis.asyncio as aioredis
        from redis.asyncio.retry import Retry
        from redis.backoff import NoBackoff

        _async_pool = aioredis.ConnectionPool.from_url(
            REDIS_URL,
            db=REDIS_DB,
            max_connections=50,
            socket_timeout=5,
            socket_connect_timeout=3,
            # SET NX may already have succeeded when its acknowledgement is lost.
            retry=Retry(NoBackoff(), 0),
            decode_responses=True,
        )
        logger.info("Redis 异步连接池已创建")
    return _async_pool


async def get_async_redis():
    """获取 Redis 异步客户端（单例，全进程共享）。

    完成连接检查后才发布客户端；故障及取消直接传播。
    使用应用持有的异步连接池，不创建另一套连接资源。
    """
    global _async_client, _async_lock
    if _async_client is not None:
        return _async_client
    if _async_lock is None:
        import asyncio
        _async_lock = asyncio.Lock()
    async with _async_lock:
        if _async_client is not None:
            return _async_client
        import redis.asyncio as aioredis

        client = aioredis.Redis(connection_pool=_get_async_pool())
        if await client.ping() is not True:
            raise RuntimeError("Redis connection check did not succeed")
        _async_client = client
        logger.info("Redis 异步客户端已连接")
        return client


async def close_async_redis() -> None:
    """Release idle shared resources; attempt both closes and propagate failures."""
    global _async_client, _async_pool, _async_lock
    client, pool = _async_client, _async_pool
    _async_client = _async_pool = _async_lock = None
    try:
        if client is not None:
            await client.aclose()
    finally:
        if pool is not None:
            await pool.aclose()
