"""资源池化管理模块。

提供SQLite连接池与共享HTTP客户端，以异步上下文管理资源借用。
"""

import asyncio
import logging
import sqlite3
import time
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """池化连接包装。"""

    conn: sqlite3.Connection
    created_at: float
    last_used_at: float
    in_use: bool = False


class ConnectionPool:
    """SQLite连接池。

    管理SQLite数据库连接的复用，支持最大连接数限制、
    空闲连接自动回收和连接健康检查。

    使用示例::

        pool = ConnectionPool(database="app.db", max_size=20)

        async with pool.acquire() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()

        await pool.close()
    """

    def __init__(
        self,
        database: str = ":memory:",
        max_size: int = 20,
        idle_timeout: float = 60.0,
        health_check_interval: float = 30.0,
    ) -> None:
        """初始化SQLite连接池。

        Args:
            database: 数据库文件路径。
            max_size: 最大连接数。
            idle_timeout: 空闲连接超时时间（秒），超时后自动回收。
            health_check_interval: 健康检查间隔（秒）。
        """
        self._database = database
        self._max_size = max_size
        self._idle_timeout = idle_timeout
        self._health_check_interval = health_check_interval
        self._pool: List[PooledConnection] = []
        self._semaphore = asyncio.Semaphore(max_size)
        self._lock = asyncio.Lock()
        self._closed = False
        self._total_acquired = 0
        self._total_released = 0
        self._total_recycled = 0
        self._total_health_failures = 0
        self._health_task: Optional[asyncio.Task] = None
        logger.info(
            "SQLite连接池初始化: database=%s, max_size=%d, idle_timeout=%.1fs",
            database,
            max_size,
            idle_timeout,
        )

    def _create_connection(self) -> PooledConnection:
        """创建新的SQLite连接。

        Returns:
            池化连接实例。
        """
        conn = sqlite3.connect(self._database, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        now = time.monotonic()
        return PooledConnection(conn=conn, created_at=now, last_used_at=now)

    def _is_connection_healthy(self, pooled: PooledConnection) -> bool:
        """检查连接是否健康。

        Args:
            pooled: 池化连接实例。

        Returns:
            连接是否健康。
        """
        try:
            pooled.conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[sqlite3.Connection]:
        """获取一个数据库连接，使用完毕自动归还。

        Yields:
            SQLite连接对象。

        Raises:
            RuntimeError: 连接池已关闭。
        """
        if self._closed:
            raise RuntimeError("连接池已关闭")

        await self._semaphore.acquire()
        pooled_conn: Optional[PooledConnection] = None

        try:
            async with self._lock:
                # 尝试从池中获取空闲连接
                for pc in self._pool:
                    if not pc.in_use:
                        if self._is_connection_healthy(pc):
                            pc.in_use = True
                            pc.last_used_at = time.monotonic()
                            pooled_conn = pc
                            break
                        else:
                            # 不健康的连接，移除并重建
                            self._pool.remove(pc)
                            self._total_health_failures += 1
                            try:
                                pc.conn.close()
                            except Exception:
                                pass
                            break

                # 没有可用连接则创建新的
                if pooled_conn is None:
                    pooled_conn = self._create_connection()
                    pooled_conn.in_use = True
                    self._pool.append(pooled_conn)

                self._total_acquired += 1

            try:
                yield pooled_conn.conn
            finally:
                pooled_conn.in_use = False
                pooled_conn.last_used_at = time.monotonic()
                self._total_released += 1
        finally:
            self._semaphore.release()

    async def _reclaim_idle(self) -> None:
        """回收空闲超时的连接。"""
        async with self._lock:
            now = time.monotonic()
            to_remove: List[PooledConnection] = []
            for pc in self._pool:
                if not pc.in_use and (now - pc.last_used_at) > self._idle_timeout:
                    to_remove.append(pc)

            for pc in to_remove:
                self._pool.remove(pc)
                try:
                    pc.conn.close()
                except Exception:
                    pass
                self._total_recycled += 1

            if to_remove:
                logger.debug("回收了 %d 个空闲连接", len(to_remove))

    async def _health_check_loop(self) -> None:
        """定期健康检查和空闲连接回收。"""
        while not self._closed:
            await asyncio.sleep(self._health_check_interval)
            if self._closed:
                break
            try:
                await self._reclaim_idle()
                # 检查并替换不健康的连接
                async with self._lock:
                    unhealthy: List[PooledConnection] = []
                    for pc in self._pool:
                        if not pc.in_use and not self._is_connection_healthy(pc):
                            unhealthy.append(pc)
                    for pc in unhealthy:
                        self._pool.remove(pc)
                        try:
                            pc.conn.close()
                        except Exception:
                            pass
                        self._total_health_failures += 1
                        # 创建新连接替代
                        new_pc = self._create_connection()
                        self._pool.append(new_pc)
                    if unhealthy:
                        logger.info("健康检查: 替换了 %d 个不健康连接", len(unhealthy))
            except Exception as exc:
                logger.error("健康检查异常: %s", exc)

    async def start(self) -> None:
        """启动连接池的健康检查循环。"""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())
            logger.info("SQLite连接池健康检查已启动")

    async def close(self) -> None:
        """关闭连接池，释放所有连接。"""
        self._closed = True
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            try:
                with ExitStack() as cleanup:
                    for pc in reversed(self._pool):
                        cleanup.callback(pc.conn.close)
            finally:
                self._pool.clear()
        logger.info("SQLite连接池已关闭，共回收 %d 个连接", self._total_recycled)

    def get_pool_stats(self) -> Dict[str, Any]:
        """返回连接池使用统计。

        Returns:
            连接池统计信息字典。
        """
        in_use = sum(1 for pc in self._pool if pc.in_use)
        idle = len(self._pool) - in_use
        return {
            "type": "sqlite",
            "database": self._database,
            "max_size": self._max_size,
            "total_connections": len(self._pool),
            "in_use": in_use,
            "idle": idle,
            "idle_timeout": self._idle_timeout,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "total_recycled": self._total_recycled,
            "total_health_failures": self._total_health_failures,
            "closed": self._closed,
        }


class HttpClientPool:
    """共享一个HTTP客户端；借用名额限制请求并发，连接回收由httpx负责。"""

    def __init__(self, max_connections: int = 100, *, request_timeout: float = 30.0) -> None:
        self._max_connections = max_connections
        self._request_timeout = request_timeout
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(max_connections)
        self._closed = False
        self._total_acquired = 0
        self._total_released = 0
        logger.info("HTTP客户端初始化: max_connections=%d, request_timeout=%.1fs", max_connections, request_timeout)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[httpx.AsyncClient]:
        """借用共享客户端；失败、取消和正常结束均归还请求名额。"""
        if self._closed:
            raise RuntimeError("HTTP客户端池已关闭")
        async with self._semaphore:
            # 关闭可能发生在等待名额期间。
            if self._closed:
                raise RuntimeError("HTTP客户端池已关闭")
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self._request_timeout),
                    limits=httpx.Limits(
                        max_connections=self._max_connections,
                        max_keepalive_connections=self._max_connections // 2,
                    ),
                )
            self._total_acquired += 1
            try:
                yield self._client
            finally:
                self._total_released += 1

    async def close(self) -> None:
        """关闭唯一持有的客户端；错误交给应用统一清理边界。"""
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()
        logger.info("HTTP客户端已关闭")

    def get_pool_stats(self) -> Dict[str, Any]:
        return {
            "type": "http_client",
            "max_connections": self._max_connections,
            "request_timeout": self._request_timeout,
            "active_requests": self._total_acquired - self._total_released,
            "client_initialized": self._client is not None,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "closed": self._closed,
        }
