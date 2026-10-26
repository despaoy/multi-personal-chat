"""共享HTTP客户端与请求并发管理。"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

import httpx

logger = logging.getLogger(__name__)


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
