"""Shared Redis connections, lifecycle and health checks."""
import logging
import os
from typing import Any, Optional

import redis
from redis import ConnectionPool

logger = logging.getLogger(__name__)

# ============================================
# 连接配置
# ============================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# 同步/异步分别维护独立连接池：
# - 同步 `redis.ConnectionPool` 用于 redis.Redis（同步客户端）
# - 异步 `redis.asyncio.ConnectionPool` 用于 redis.asyncio.Redis（异步客户端）
# 此前曾把同步 pool 复用给异步 client，创建对象不会报错，但执行异步命令时
# 会在事件循环中调用阻塞 IO，真实 Redis 下缓存、消息去重等功能持续退化。
_pool: Optional[ConnectionPool] = None
_client: Optional[redis.Redis] = None
_async_pool: Optional[Any] = None  # redis.asyncio.ConnectionPool
_async_client: Optional[Any] = None
_async_lock: Any = None  # asyncio.Lock, 延迟初始化


def get_redis_pool() -> ConnectionPool:
    """获取或创建同步 Redis 连接池"""
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            REDIS_URL,
            db=REDIS_DB,
            max_connections=50,
            socket_timeout=5,
            socket_connect_timeout=3,
            retry_on_timeout=True,
            decode_responses=True,
        )
        logger.info(f"✅ Redis 同步连接池已创建: {REDIS_URL}")
    return _pool


def get_redis() -> redis.Redis:
    """获取 Redis 同步客户端"""
    global _client
    if _client is None:
        _client = redis.Redis(connection_pool=get_redis_pool())
    return _client


# ============================================
# 异步 Redis 客户端（统一共享，避免多模块各自创建连接池）
# 此前多个模块各自调用
# redis.asyncio.from_url() 创建独立连接池（分别 10/20 连接），
# 与同步客户端合计 80 个 max 连接。现统一为单一 async 客户端。
# ============================================
def _get_async_pool():
    """获取或创建异步 Redis 连接池（redis.asyncio.ConnectionPool）。

    异步客户端必须使用异步连接池：将同步 pool 传给 redis.asyncio.Redis
    不会在构造时报错，但所有命令都会在事件循环中阻塞，导致缓存、消息去重
    等异步功能持续退化。
    """
    global _async_pool
    if _async_pool is None:
        import redis.asyncio as aioredis
        _async_pool = aioredis.ConnectionPool.from_url(
            REDIS_URL,
            db=REDIS_DB,
            max_connections=50,
            socket_timeout=5,
            socket_connect_timeout=3,
            retry_on_timeout=True,
            decode_responses=True,
        )
        logger.info(f"✅ Redis 异步连接池已创建: {REDIS_URL}")
    return _async_pool


async def get_async_redis():
    """获取 Redis 异步客户端（单例，全进程共享）。

    返回 redis.asyncio.Redis 实例；若 Redis 不可用则返回 None。
    所有需要 async Redis 的生产模块（配置缓存、bot 去重等）都应调用此函数，
    不得各自通过 from_url 创建独立连接池。
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
        try:
            import redis.asyncio as aioredis
            # 使用异步连接池，避免与同步 pool 混用导致事件循环阻塞
            _async_client = aioredis.Redis(connection_pool=_get_async_pool())
            await _async_client.ping()
            logger.info(f"✅ Redis 异步客户端已连接(共享异步连接池): {REDIS_URL}")
            return _async_client
        except Exception as e:
            logger.warning(f"Redis 异步客户端不可用: {e}")
            _async_client = None
            return None


async def close_async_redis() -> None:
    """关闭异步 Redis 客户端与连接池（应用退出时调用）"""
    global _async_client, _async_pool
    if _async_client is not None:
        try:
            await _async_client.aclose()
        except Exception:
            pass
        _async_client = None
    if _async_pool is not None:
        try:
            await _async_pool.aclose()
        except Exception:
            pass
        _async_pool = None


def close_sync_redis() -> None:
    """关闭同步 Redis 客户端与连接池（应用退出时调用）。

    与 close_async_redis 对称，补全同步客户端的生命周期管理。
    进程退出时 OS 会回收资源，但显式关闭可在长驻进程中复用模块时
    避免连接泄漏。
    """
    global _client, _pool
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
    if _pool is not None:
        try:
            _pool.disconnect()
        except Exception:
            pass
        _pool = None


def health_check() -> bool:
    """Redis 健康检查"""
    try:
        return get_redis().ping()
    except Exception:
        return False
