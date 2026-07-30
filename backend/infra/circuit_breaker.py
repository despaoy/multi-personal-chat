"""服务熔断与降级策略模块。

实现基于三状态机的CircuitBreaker，支持装饰器模式和上下文管理器模式，
提供多种降级策略和熔断事件回调机制。
"""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, Union

from interfaces import CircuitBreakerInterface

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态枚举。"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class DegradationMode(Enum):
    """降级模式枚举。"""
    CACHE = "cache"          # 返回缓存结果
    DEFAULT = "default"      # 返回默认值
    QUEUE = "queue"          # 排队等待恢复
    RAISE = "raise"          # Raise CircuitOpenError; do not fabricate a success


class CircuitOpenError(Exception):
    """熔断器打开时抛出的异常。"""
    pass


class CircuitBreakerStats:
    """熔断器状态统计。"""

    def __init__(self) -> None:
        self.total_calls: int = 0
        self.success_calls: int = 0
        self.failure_calls: int = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.last_state_change_time: float = time.monotonic()

    def record_success(self) -> None:
        """记录一次成功调用。"""
        self.total_calls += 1
        self.success_calls += 1
        self.last_success_time = time.monotonic()

    def record_failure(self) -> None:
        """记录一次失败调用。"""
        self.total_calls += 1
        self.failure_calls += 1
        self.last_failure_time = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "total_calls": self.total_calls,
            "success_calls": self.success_calls,
            "failure_calls": self.failure_calls,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "last_state_change_time": self.last_state_change_time,
        }


class CircuitBreaker:
    """熔断器：三状态机实现（CLOSED/OPEN/HALF_OPEN）。

    当连续失败次数达到阈值时，熔断器从CLOSED切换到OPEN状态，
    阻止请求通过；经过恢复超时后进入HALF_OPEN状态，允许少量
    请求通过以探测服务是否恢复。

    Args:
        name: 熔断器名称，用于标识和日志。
        failure_threshold: 连续失败阈值，达到后触发熔断。
        recovery_timeout: 恢复超时秒数，OPEN状态持续时间。
        half_open_max_calls: HALF_OPEN状态下允许的最大探测调用数。
        fallback: 降级函数，熔断器打开时执行。
        degradation_mode: 降级模式。
        on_open: 熔断器打开时的回调。
        on_close: 熔断器关闭时的回调。
        on_half_open: 熔断器进入半开状态时的回调。
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        fallback: Optional[Callable[..., Any]] = None,
        degradation_mode: DegradationMode = DegradationMode.DEFAULT,
        on_open: Optional[Callable[["CircuitBreaker"], None]] = None,
        on_close: Optional[Callable[["CircuitBreaker"], None]] = None,
        on_half_open: Optional[Callable[["CircuitBreaker"], None]] = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.fallback = fallback
        self.degradation_mode = degradation_mode
        self.on_open = on_open
        self.on_close = on_close
        self.on_half_open = on_half_open

        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._last_failure_time: Optional[float] = None
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._cache: Optional[Any] = None
        self._default_value: Any = None
        self._queue: list[asyncio.Future[Any]] = []

    @property
    def state(self) -> CircuitState:
        """Current circuit state, refreshing OPEN -> HALF_OPEN after timeout."""
        self._maybe_transition_to_half_open()
        return self._state

    def _maybe_transition_to_half_open(self) -> None:
        """Move OPEN circuits to HALF_OPEN once the recovery timeout has elapsed."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            self._transition_to_half_open()

    def _transition_to_open(self) -> None:
        """转换到OPEN状态。"""
        if self._state != CircuitState.OPEN:
            old_state = self._state
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()
            self._stats.last_state_change_time = time.monotonic()
            logger.warning(
                "熔断器 [%s] 从 %s 切换到 OPEN 状态",
                self.name, old_state.value,
            )
            if self.on_open:
                try:
                    self.on_open(self)
                except Exception as e:
                    logger.error("熔断器 [%s] on_open 回调执行失败: %s", self.name, e)

    def _transition_to_closed(self) -> None:
        """转换到CLOSED状态。"""
        if self._state != CircuitState.CLOSED:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._stats.last_state_change_time = time.monotonic()
            logger.info(
                "熔断器 [%s] 从 %s 切换到 CLOSED 状态",
                self.name, old_state.value,
            )
            if self.on_close:
                try:
                    self.on_close(self)
                except Exception as e:
                    logger.error("熔断器 [%s] on_close 回调执行失败: %s", self.name, e)
            # 释放排队等待的请求
            self._release_queue()

    def _transition_to_half_open(self) -> None:
        """转换到HALF_OPEN状态。"""
        if self._state != CircuitState.HALF_OPEN:
            old_state = self._state
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._stats.last_state_change_time = time.monotonic()
            logger.info(
                "熔断器 [%s] 从 %s 切换到 HALF_OPEN 状态",
                self.name, old_state.value,
            )
            if self.on_half_open:
                try:
                    self.on_half_open(self)
                except Exception as e:
                    logger.error("熔断器 [%s] on_half_open 回调执行失败: %s", self.name, e)

    def _release_queue(self) -> None:
        """释放排队等待的所有请求，使其可以重新发起调用。"""
        for future in self._queue:
            if not future.done():
                future.set_result(None)
        self._queue.clear()

    async def _handle_degradation(self, *args: Any, **kwargs: Any) -> Any:
        """处理降级逻辑。

        根据降级模式返回缓存结果、默认值或将请求排队等待恢复。
        """
        if self.degradation_mode == DegradationMode.RAISE:
            raise CircuitOpenError(f"Circuit breaker [{self.name}] is open")

        if self.fallback is not None:
            logger.debug("熔断器 [%s] 执行降级函数", self.name)
            if asyncio.iscoroutinefunction(self.fallback):
                return await self.fallback(*args, **kwargs)
            return self.fallback(*args, **kwargs)

        if self.degradation_mode == DegradationMode.CACHE:
            if self._cache is not None:
                logger.debug("熔断器 [%s] 返回缓存结果", self.name)
                return self._cache
            logger.warning("熔断器 [%s] 无缓存结果可用，返回默认值", self.name)
            return self._default_value

        if self.degradation_mode == DegradationMode.QUEUE:
            # C5 fix: QUEUE 模式的 future await 必须在锁外进行，否则
            # _release_queue() 无法获取锁来 resolve future → 死锁。
            # 因此 call() 现在直接处理 QUEUE 模式（锁内创建 future、锁外 await），
            # 不会走到这里。此分支仅作为防御性兜底：若 _handle_degradation 被直接
            # 调用处理 QUEUE，返回默认值而非死锁。
            logger.warning(
                "熔断器 [%s] QUEUE 降级不应经由 _handle_degradation 处理，返回默认值",
                self.name,
            )
            return self._default_value

        # DegradationMode.DEFAULT
        logger.debug("熔断器 [%s] 返回默认值", self.name)
        return self._default_value

    def set_cache(self, value: Any) -> None:
        """设置缓存值，用于CACHE降级模式。"""
        self._cache = value

    def set_default(self, value: Any) -> None:
        """设置默认值，用于DEFAULT降级模式。"""
        self._default_value = value

    async def record_success(self) -> None:
        """记录一次成功调用（手动模式，用于流式等无法整体包裹的场景）。

        与 call() 内部逻辑一致：成功在 HALF_OPEN → CLOSED，CLOSED 下重置失败计数。
        必须由调用方在确认成功后显式调用。
        """
        async with self._lock:
            self._stats.record_success()
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def record_failure(self, error: Optional[Any] = None) -> None:
        """记录一次失败调用（手动模式，用于流式等无法整体包裹的场景）。

        与 call() 内部逻辑一致：失败在 HALF_OPEN → OPEN，CLOSED 下计数达阈值 → OPEN。
        必须由调用方在捕获异常后显式调用。
        """
        async with self._lock:
            self._stats.record_failure()
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self._failure_count >= self.failure_threshold:
                self._transition_to_open()
        if error is not None:
            logger.error(
                "熔断器 [%s] 调用失败: %s (连续失败: %d/%d)",
                self.name, error, self._failure_count, self.failure_threshold,
            )

    @property
    def default_value(self) -> Any:
        """DEFAULT/CACHE 降级时返回的默认值（读访问）。"""
        return self._default_value

    async def call(self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
        """通过熔断器调用异步函数。

        Args:
            func: 要调用的异步函数。
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            函数调用结果或降级响应。

        Raises:
            CircuitOpenError: 熔断器打开且无降级策略时抛出。
        """
        # C5 fix: QUEUE 降级模式的 future await 必须在锁外进行。
        # 此前 call() 在 async with self._lock 内调用 _handle_degradation，
        # QUEUE 模式在锁内 await future；而唯一能 resolve future 的
        # _release_queue() 从 _transition_to_closed() 调用，后者需要获取
        # 同一把 self._lock → 确定性死锁。
        # 修复：锁内仅完成 future 创建与入队，锁外再 await。
        queue_future: Optional[asyncio.Future[Any]] = None
        async with self._lock:
            self._maybe_transition_to_half_open()
            current_state = self._state
            if current_state == CircuitState.OPEN:
                if self._last_failure_time is not None and time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._transition_to_half_open()
                    current_state = CircuitState.HALF_OPEN
                else:
                    # OPEN 状态需要降级。
                    # QUEUE 模式特殊处理：锁内创建 future 入队，锁外 await。
                    if (
                        self.degradation_mode == DegradationMode.QUEUE
                        and self.fallback is None
                    ):
                        logger.debug("熔断器 [%s] 请求排队等待恢复", self.name)
                        queue_future = asyncio.get_running_loop().create_future()
                        self._queue.append(queue_future)
                    else:
                        logger.warning("熔断器 [%s] 处于 OPEN 状态，执行降级", self.name)
                        return await self._handle_degradation(*args, **kwargs)

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    # HALF_OPEN 达上限需要降级，同样特殊处理 QUEUE 模式。
                    if (
                        self.degradation_mode == DegradationMode.QUEUE
                        and self.fallback is None
                    ):
                        logger.debug("熔断器 [%s] 请求排队等待恢复", self.name)
                        queue_future = asyncio.get_running_loop().create_future()
                        self._queue.append(queue_future)
                    else:
                        logger.warning(
                            "熔断器 [%s] HALF_OPEN 状态已达最大探测调用数，执行降级",
                            self.name,
                        )
                        return await self._handle_degradation(*args, **kwargs)
                else:
                    self._half_open_calls += 1

        # C5 fix: QUEUE 模式在锁外 await future。
        # _release_queue() 在 _transition_to_closed() 中被调用，后者需要
        # 获取 self._lock；此处已释放锁，_release_queue 可以正常获取并 resolve future。
        if queue_future is not None:
            try:
                await asyncio.wait_for(queue_future, timeout=self.recovery_timeout * 2)
                logger.info("熔断器 [%s] 排队请求已恢复", self.name)
                return self._default_value
            except asyncio.TimeoutError:
                logger.warning("熔断器 [%s] 排队等待超时，返回默认值", self.name)
                return self._default_value

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._stats.record_success()
                self._cache = result  # 更新缓存
                if current_state == CircuitState.HALF_OPEN:
                    self._transition_to_closed()
                elif current_state == CircuitState.CLOSED:
                    self._failure_count = 0
            return result
        except Exception as e:
            async with self._lock:
                self._stats.record_failure()
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if current_state == CircuitState.HALF_OPEN:
                    self._transition_to_open()
                elif self._failure_count >= self.failure_threshold:
                    self._transition_to_open()
            logger.error(
                "熔断器 [%s] 调用失败: %s (连续失败: %d/%d)",
                self.name, e, self._failure_count, self.failure_threshold,
            )
            raise

    def protect(self, func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
        """装饰器模式：用熔断器保护异步函数。

        Usage:
            cb = CircuitBreaker(name="my_service")
            @cb.protect
            async def my_service_call():
                ...
        """
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.call(func, *args, **kwargs)
        return wrapper

    @asynccontextmanager
    async def __call__(self) -> Any:
        """上下文管理器模式：用熔断器保护代码块。

        Usage:
            async with circuit_breaker:
                result = await some_service()
        """
        async with self._lock:
            self._maybe_transition_to_half_open()
            current_state = self._state
            if current_state == CircuitState.OPEN:
                if self._last_failure_time is not None and time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._transition_to_half_open()
                    current_state = CircuitState.HALF_OPEN
                else:
                    logger.warning("熔断器 [%s] 处于 OPEN 状态", self.name)
                    raise CircuitOpenError(
                        f"熔断器 [{self.name}] 处于 OPEN 状态，请求被拒绝"
                    )

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(
                        f"熔断器 [{self.name}] HALF_OPEN 状态已达最大探测调用数"
                    )
                self._half_open_calls += 1

        try:
            yield self
            async with self._lock:
                self._stats.record_success()
                if current_state == CircuitState.HALF_OPEN:
                    self._transition_to_closed()
                elif current_state == CircuitState.CLOSED:
                    self._failure_count = 0
        except Exception as e:
            async with self._lock:
                self._stats.record_failure()
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if current_state == CircuitState.HALF_OPEN:
                    self._transition_to_open()
                elif self._failure_count >= self.failure_threshold:
                    self._transition_to_open()
            logger.error(
                "熔断器 [%s] 上下文内调用失败: %s", self.name, e,
            )
            raise

    def get_stats(self) -> dict[str, Any]:
        """获取熔断器状态统计。

        Returns:
            包含总调用数、成功数、失败数、当前状态等信息的字典。
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "current_failure_count": self._failure_count,
            "half_open_calls": self._half_open_calls,
            "degradation_mode": self.degradation_mode.value,
            **self._stats.to_dict(),
        }

    async def reset(self) -> None:
        """手动重置熔断器到CLOSED状态。"""
        async with self._lock:
            self._transition_to_closed()
            self._stats = CircuitBreakerStats()
            logger.info("熔断器 [%s] 已手动重置", self.name)

    async def trip(self) -> None:
        """手动触发熔断器到OPEN状态。"""
        async with self._lock:
            self._transition_to_open()
            logger.info("熔断器 [%s] 已手动触发熔断", self.name)


class CircuitBreakerRegistry:
    """熔断器全局注册表，按服务名管理多个熔断器。

    Usage:
        registry = CircuitBreakerRegistry()
        cb = registry.get_or_create("my_service", failure_threshold=3)
        stats = registry.get_all_stats()
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        fallback: Optional[Callable[..., Any]] = None,
        degradation_mode: DegradationMode = DegradationMode.DEFAULT,
        on_open: Optional[Callable[[CircuitBreaker], None]] = None,
        on_close: Optional[Callable[[CircuitBreaker], None]] = None,
        on_half_open: Optional[Callable[[CircuitBreaker], None]] = None,
    ) -> CircuitBreaker:
        """获取或创建指定名称的熔断器。

        如果同名熔断器已存在，则返回现有实例（忽略其他参数）。
        """
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    half_open_max_calls=half_open_max_calls,
                    fallback=fallback,
                    degradation_mode=degradation_mode,
                    on_open=on_open,
                    on_close=on_close,
                    on_half_open=on_half_open,
                )
                logger.info("注册熔断器: %s", name)
            return self._breakers[name]

    async def get(self, name: str) -> Optional[CircuitBreaker]:
        """获取指定名称的熔断器，不存在则返回None。"""
        async with self._lock:
            return self._breakers.get(name)

    async def remove(self, name: str) -> bool:
        """移除指定名称的熔断器。

        Returns:
            是否成功移除。
        """
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info("移除熔断器: %s", name)
                return True
            return False

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """获取所有熔断器的状态统计。"""
        return {name: cb.get_stats() for name, cb in self._breakers.items()}

    async def reset_all(self) -> None:
        """重置所有熔断器。"""
        for cb in self._breakers.values():
            await cb.reset()
        logger.info("已重置所有熔断器")

    @property
    def names(self) -> list[str]:
        """获取所有已注册的熔断器名称。"""
        return list(self._breakers.keys())


# 全局注册表实例
global_registry = CircuitBreakerRegistry()

# 接口契约验证：确保 CircuitBreaker 实现 CircuitBreakerInterface 接口
assert isinstance(CircuitBreaker(), CircuitBreakerInterface), f"CircuitBreaker must implement CircuitBreakerInterface"
