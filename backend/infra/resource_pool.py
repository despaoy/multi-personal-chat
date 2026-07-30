"""资源池化管理模块。

提供SQLite连接池、HTTP客户端池和模型推理资源池，
支持acquire/release上下文管理、健康检查和自动回收。
"""

import asyncio
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

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
            for pc in self._pool:
                try:
                    pc.conn.close()
                except Exception:
                    pass
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
    """httpx异步客户端池。

    复用HTTP连接，支持最大连接数限制和连接健康检查。
    需要httpx库支持。

    使用示例::

        pool = HttpClientPool(max_connections=100)

        async with pool.acquire() as client:
            resp = await client.get("https://example.com")

        await pool.close()
    """

    def __init__(
        self,
        max_connections: int = 100,
        idle_timeout: float = 60.0,
        health_check_interval: float = 30.0,
        request_timeout: float = 30.0,
    ) -> None:
        """初始化HTTP客户端池。

        Args:
            max_connections: 最大连接数。
            idle_timeout: 空闲客户端超时时间（秒）。
            health_check_interval: 健康检查间隔（秒）。
            request_timeout: 单次HTTP请求超时时间（秒）。
                LLM 推理场景需设置为 120.0 或更高，默认 30.0 仅适用于
                短促的 API 调用（如 RAG 检索）。
        """
        self._max_connections = max_connections
        self._idle_timeout = idle_timeout
        self._health_check_interval = health_check_interval
        self._request_timeout = request_timeout
        self._clients: List[Any] = []  # httpx.AsyncClient instances
        self._client_info: Dict[int, Dict[str, Any]] = {}
        self._semaphore = asyncio.Semaphore(max_connections)
        self._lock = asyncio.Lock()
        self._closed = False
        self._total_acquired = 0
        self._total_released = 0
        self._total_recycled = 0
        self._health_task: Optional[asyncio.Task] = None
        self._httpx_available = False

        try:
            import httpx  # noqa: F401
            self._httpx_available = True
        except ImportError:
            logger.warning("httpx未安装，HttpClientPool将不可用")

        logger.info(
            "HTTP客户端池初始化: max_connections=%d, idle_timeout=%.1fs, request_timeout=%.1fs",
            max_connections,
            idle_timeout,
            request_timeout,
        )

    def _create_client(self) -> Any:
        """创建新的httpx异步客户端。

        Returns:
            httpx.AsyncClient 实例。
        """
        import httpx

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._request_timeout),
            limits=httpx.Limits(
                max_connections=self._max_connections,
                max_keepalive_connections=self._max_connections // 2,
            ),
        )
        now = time.monotonic()
        self._client_info[id(client)] = {
            "created_at": now,
            "last_used_at": now,
            "in_use": False,
        }
        return client

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """获取一个HTTP客户端，使用完毕自动归还。

        Yields:
            httpx.AsyncClient 实例。

        Raises:
            RuntimeError: 客户端池已关闭或httpx不可用。
        """
        if not self._httpx_available:
            raise RuntimeError("httpx未安装，无法使用HttpClientPool")
        if self._closed:
            raise RuntimeError("HTTP客户端池已关闭")

        await self._semaphore.acquire()
        client: Optional[Any] = None

        try:
            async with self._lock:
                # 尝试获取空闲客户端
                for c in self._clients:
                    info = self._client_info.get(id(c))
                    if info and not info["in_use"]:
                        info["in_use"] = True
                        info["last_used_at"] = time.monotonic()
                        client = c
                        break

                # 没有可用客户端则创建新的
                if client is None:
                    client = self._create_client()
                    self._clients.append(client)

                self._total_acquired += 1

            try:
                yield client
            finally:
                info = self._client_info.get(id(client))
                if info:
                    info["in_use"] = False
                    info["last_used_at"] = time.monotonic()
                self._total_released += 1
        finally:
            self._semaphore.release()

    async def _reclaim_idle(self) -> None:
        """回收空闲超时的HTTP客户端。"""
        async with self._lock:
            now = time.monotonic()
            to_remove: List[Any] = []
            for c in self._clients:
                info = self._client_info.get(id(c))
                if info and not info["in_use"] and (now - info["last_used_at"]) > self._idle_timeout:
                    to_remove.append(c)

            for c in to_remove:
                self._clients.remove(c)
                self._client_info.pop(id(c), None)
                try:
                    await c.aclose()
                except Exception:
                    pass
                self._total_recycled += 1

            if to_remove:
                logger.debug("回收了 %d 个空闲HTTP客户端", len(to_remove))

    async def _health_check_loop(self) -> None:
        """定期健康检查和空闲客户端回收。"""
        while not self._closed:
            await asyncio.sleep(self._health_check_interval)
            if self._closed:
                break
            try:
                await self._reclaim_idle()
            except Exception as exc:
                logger.error("HTTP客户端池健康检查异常: %s", exc)

    async def start(self) -> None:
        """启动客户端池的健康检查循环。"""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())
            logger.info("HTTP客户端池健康检查已启动")

    async def close(self) -> None:
        """关闭客户端池，释放所有客户端。"""
        self._closed = True
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for c in self._clients:
                try:
                    await c.aclose()
                except Exception:
                    pass
            self._clients.clear()
            self._client_info.clear()
        logger.info("HTTP客户端池已关闭，共回收 %d 个客户端", self._total_recycled)

    def get_pool_stats(self) -> Dict[str, Any]:
        """返回客户端池使用统计。

        Returns:
            客户端池统计信息字典。
        """
        in_use = sum(
            1 for info in self._client_info.values() if info.get("in_use", False)
        )
        idle = len(self._clients) - in_use
        return {
            "type": "http_client",
            "max_connections": self._max_connections,
            "total_clients": len(self._clients),
            "in_use": in_use,
            "idle": idle,
            "idle_timeout": self._idle_timeout,
            "request_timeout": self._request_timeout,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "total_recycled": self._total_recycled,
            "httpx_available": self._httpx_available,
            "closed": self._closed,
        }


@dataclass
class InferenceSlot:
    """模型推理槽位。"""

    slot_id: int
    in_use: bool = False
    current_task_id: Optional[str] = None
    acquired_at: Optional[float] = None


class ModelInferencePool:
    """模型推理资源池。

    管理GPU显存分配和并发推理请求排队，
    限制同时执行的推理数量以避免显存溢出。

    使用示例::

        pool = ModelInferencePool(max_concurrent=2, gpu_memory_limit_gb=8.0)

        async with pool.acquire(task_id="infer-001") as slot:
            result = await model.inference(input_data)

        await pool.close()
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        gpu_memory_limit_gb: Optional[float] = None,
        queue_timeout: float = 300.0,
    ) -> None:
        """初始化模型推理资源池。

        Args:
            max_concurrent: 最大并发推理数。
            gpu_memory_limit_gb: GPU显存限制（GB），None表示不限制。
            queue_timeout: 排队等待超时时间（秒）。
        """
        self._max_concurrent = max_concurrent
        self._gpu_memory_limit_gb = gpu_memory_limit_gb
        self._queue_timeout = queue_timeout
        self._slots: List[InferenceSlot] = [
            InferenceSlot(slot_id=i) for i in range(max_concurrent)
        ]
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._closed = False
        self._total_acquired = 0
        self._total_released = 0
        self._total_timeouts = 0
        self._current_memory_usage_gb: float = 0.0
        self._peak_memory_usage_gb: float = 0.0
        logger.info(
            "模型推理资源池初始化: max_concurrent=%d, gpu_memory_limit=%sGB",
            max_concurrent,
            gpu_memory_limit_gb or "无限制",
        )

    @asynccontextmanager
    async def acquire(
        self,
        task_id: Optional[str] = None,
        estimated_memory_gb: float = 0.0,
    ) -> AsyncIterator[InferenceSlot]:
        """获取一个推理槽位，使用完毕自动释放。

        Args:
            task_id: 任务标识，用于追踪。
            estimated_memory_gb: 预估显存占用（GB），用于显存管理。

        Yields:
            推理槽位实例。

        Raises:
            RuntimeError: 资源池已关闭。
            asyncio.TimeoutError: 排队超时。
        """
        if self._closed:
            raise RuntimeError("模型推理资源池已关闭")

        # 显存检查
        if (
            self._gpu_memory_limit_gb is not None
            and estimated_memory_gb > 0
            and (self._current_memory_usage_gb + estimated_memory_gb) > self._gpu_memory_limit_gb
        ):
            logger.warning(
                "预估显存 %.2fGB + 已用 %.2fGB 超过限制 %.2fGB，任务 %s 将排队等待",
                estimated_memory_gb,
                self._current_memory_usage_gb,
                self._gpu_memory_limit_gb,
                task_id,
            )

        acquired = await asyncio.wait_for(
            self._semaphore.acquire(),
            timeout=self._queue_timeout,
        )
        # asyncio.Semaphore.acquire 返回 bool
        if not acquired:
            self._total_timeouts += 1
            raise asyncio.TimeoutError(
                f"推理槽位排队超时 ({self._queue_timeout}s)，任务: {task_id}"
            )

        slot: Optional[InferenceSlot] = None

        try:
            async with self._lock:
                for s in self._slots:
                    if not s.in_use:
                        s.in_use = True
                        s.current_task_id = task_id
                        s.acquired_at = time.monotonic()
                        slot = s
                        break

                if slot is None:
                    # C-F2 fix: 不应发生，semaphore已控制并发数。
                    # 原先在此处 release() 后，外层 finally 会再次 release()，
                    # 导致信号量计数器永久偏移，最终允许超过 max_concurrent 的并发数。
                    # 现在直接 raise，由 finally 块统一 release 一次。
                    raise RuntimeError("无可用推理槽位（内部错误）")

                self._total_acquired += 1
                self._current_memory_usage_gb += estimated_memory_gb
                if self._current_memory_usage_gb > self._peak_memory_usage_gb:
                    self._peak_memory_usage_gb = self._current_memory_usage_gb

                logger.debug(
                    "分配推理槽位 #%d 给任务 %s (预估显存: %.2fGB)",
                    slot.slot_id,
                    task_id,
                    estimated_memory_gb,
                )

            try:
                yield slot
            finally:
                async with self._lock:
                    slot.in_use = False
                    slot.current_task_id = None
                    slot.acquired_at = None
                    self._total_released += 1
                    self._current_memory_usage_gb = max(
                        0.0, self._current_memory_usage_gb - estimated_memory_gb
                    )
        finally:
            self._semaphore.release()

    async def close(self) -> None:
        """关闭推理资源池。"""
        self._closed = True
        async with self._lock:
            for slot in self._slots:
                if slot.in_use:
                    logger.warning(
                        "关闭推理池时槽位 #%d 仍在使用 (任务: %s)",
                        slot.slot_id,
                        slot.current_task_id,
                    )
                slot.in_use = False
                slot.current_task_id = None
                slot.acquired_at = None
        logger.info("模型推理资源池已关闭")

    def get_pool_stats(self) -> Dict[str, Any]:
        """返回推理资源池使用统计。

        Returns:
            资源池统计信息字典。
        """
        in_use_slots = [s for s in self._slots if s.in_use]
        active_tasks = [
            {
                "slot_id": s.slot_id,
                "task_id": s.current_task_id,
                "acquired_at": s.acquired_at,
                "duration": time.monotonic() - s.acquired_at if s.acquired_at else 0.0,
            }
            for s in in_use_slots
        ]
        return {
            "type": "model_inference",
            "max_concurrent": self._max_concurrent,
            "gpu_memory_limit_gb": self._gpu_memory_limit_gb,
            "current_memory_usage_gb": round(self._current_memory_usage_gb, 2),
            "peak_memory_usage_gb": round(self._peak_memory_usage_gb, 2),
            "total_slots": len(self._slots),
            "in_use": len(in_use_slots),
            "idle": len(self._slots) - len(in_use_slots),
            "queue_timeout": self._queue_timeout,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "total_timeouts": self._total_timeouts,
            "active_tasks": active_tasks,
            "closed": self._closed,
        }
