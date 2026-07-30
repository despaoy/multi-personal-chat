"""请求负载均衡器模块。

提供多种负载均衡策略，在多个模型提供商间分配请求，
支持健康检查、自动剔除和负载统计。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProviderStatus(str, Enum):
    """提供商健康状态。"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class Provider:
    """模型提供商信息。"""

    name: str
    url: str
    weight: float = 1.0
    status: ProviderStatus = ProviderStatus.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    total_response_time: float = 0.0
    last_failure_time: float = 0.0
    last_used_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """成功率。"""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def avg_response_time(self) -> float:
        """平均响应时间（秒）。"""
        if self.success_count == 0:
            return float("inf")
        return self.total_response_time / self.success_count

    def record_success(self, response_time: float) -> None:
        """记录一次成功请求。

        Args:
            response_time: 请求响应时间（秒）。
        """
        self.total_requests += 1
        self.success_count += 1
        self.consecutive_failures = 0
        self.total_response_time += response_time
        self.last_used_time = time.monotonic()
        if self.status == ProviderStatus.UNHEALTHY:
            self.status = ProviderStatus.HEALTHY
            logger.info("提供商 %s 已恢复健康", self.name)

    def record_failure(self) -> None:
        """记录一次失败请求。"""
        self.total_requests += 1
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= 3:
            self.status = ProviderStatus.UNHEALTHY
            logger.warning(
                "提供商 %s 连续失败 %d 次，标记为不健康",
                self.name,
                self.consecutive_failures,
            )

    def try_recover(self) -> bool:
        """尝试恢复不健康的提供商（30秒冷却后）。

        Returns:
            是否成功恢复。
        """
        if self.status != ProviderStatus.UNHEALTHY:
            return False
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= 30.0:
            self.status = ProviderStatus.HEALTHY
            self.consecutive_failures = 0
            logger.info("提供商 %s 冷却结束，重新标记为健康", self.name)
            return True
        return False

    def to_stats(self) -> Dict[str, Any]:
        """返回提供商统计信息。"""
        return {
            "name": self.name,
            "url": self.url,
            "status": self.status.value,
            "weight": self.weight,
            "current_connections": self.current_connections,
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": round(self.success_rate, 4),
            "avg_response_time": round(self.avg_response_time, 4),
        }


class BaseBalancer(ABC):
    """负载均衡器基类。"""

    def __init__(self) -> None:
        self._providers: List[Provider] = []
        self._lock = asyncio.Lock()

    def set_providers(self, providers: List[Provider]) -> None:
        """设置提供商列表。

        Args:
            providers: 提供商列表。
        """
        self._providers = list(providers)
        logger.info("负载均衡器已配置 %d 个提供商", len(self._providers))

    def _get_healthy_providers(self) -> List[Provider]:
        """获取健康的提供商列表，同时尝试恢复冷却结束的提供商。

        Returns:
            健康的提供商列表。
        """
        healthy: List[Provider] = []
        for p in self._providers:
            p.try_recover()
            if p.status == ProviderStatus.HEALTHY:
                healthy.append(p)
        return healthy

    @abstractmethod
    def select_provider(self, providers: Optional[List[Provider]] = None) -> Optional[Provider]:
        """从提供商列表中选择一个提供商。

        Args:
            providers: 可选的提供商列表，不传则使用内置列表。

        Returns:
            选中的提供商，无可用提供商时返回 None。
        """

    def get_stats(self) -> Dict[str, Any]:
        """返回各提供商的负载统计。

        Returns:
            包含所有提供商统计信息的字典。
        """
        healthy = self._get_healthy_providers()
        return {
            "total_providers": len(self._providers),
            "healthy_providers": len(healthy),
            "unhealthy_providers": len(self._providers) - len(healthy),
            "providers": [p.to_stats() for p in self._providers],
        }


class RoundRobinBalancer(BaseBalancer):
    """轮询负载均衡器。

    在多个模型提供商间按顺序轮流分配请求，
    自动跳过不健康的提供商。
    """

    def __init__(self) -> None:
        super().__init__()
        self._index: int = 0

    def select_provider(self, providers: Optional[List[Provider]] = None) -> Optional[Provider]:
        """轮询选择提供商。

        Args:
            providers: 可选的提供商列表。

        Returns:
            选中的提供商，无可用时返回 None。
        """
        candidates = providers if providers is not None else self._get_healthy_providers()
        if not candidates:
            logger.warning("无可用提供商")
            return None

        selected = candidates[self._index % len(candidates)]
        self._index = (self._index + 1) % len(candidates)
        logger.debug("轮询选择提供商: %s", selected.name)
        return selected


class WeightedBalancer(BaseBalancer):
    """加权轮询负载均衡器。

    根据提供商的权重分配请求，权重由响应时间和成功率动态计算。
    响应时间越短、成功率越高的提供商获得更多请求。
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_weights: Dict[str, float] = {}

    def _calculate_dynamic_weight(self, provider: Provider) -> float:
        """根据响应时间和成功率计算动态权重。

        权重 = 基础权重 * 成功率 / (1 + 平均响应时间)

        Args:
            provider: 提供商实例。

        Returns:
            动态权重值。
        """
        base_weight = provider.weight
        success_factor = provider.success_rate
        # 响应时间因子：响应时间越短权重越高
        avg_rt = provider.avg_response_time
        if avg_rt == float("inf"):
            time_factor = 0.1
        else:
            time_factor = 1.0 / (1.0 + avg_rt)
        return base_weight * success_factor * time_factor

    def select_provider(self, providers: Optional[List[Provider]] = None) -> Optional[Provider]:
        """加权轮询选择提供商。

        使用平滑加权轮询算法（Nginx风格），避免短时间内集中分配。

        Args:
            providers: 可选的提供商列表。

        Returns:
            选中的提供商，无可用时返回 None。
        """
        candidates = providers if providers is not None else self._get_healthy_providers()
        if not candidates:
            logger.warning("无可用提供商")
            return None

        # 计算每个候选的动态权重
        for p in candidates:
            if p.name not in self._current_weights:
                self._current_weights[p.name] = 0.0

        # 平滑加权轮询：每个候选加上自身权重，选最大的，选中后减去总权重
        total_weight = sum(self._calculate_dynamic_weight(p) for p in candidates)

        if total_weight <= 0:
            # 所有权重为0时回退到简单轮询
            return candidates[0]

        best: Optional[Provider] = None
        best_weight = -1.0

        for p in candidates:
            dynamic_w = self._calculate_dynamic_weight(p)
            self._current_weights[p.name] = self._current_weights.get(p.name, 0.0) + dynamic_w
            if self._current_weights[p.name] > best_weight:
                best_weight = self._current_weights[p.name]
                best = p

        if best is not None:
            self._current_weights[best.name] -= total_weight
            logger.debug("加权选择提供商: %s (当前权重: %.4f)", best.name, best_weight)

        return best


class LeastConnectionBalancer(BaseBalancer):
    """最少连接数负载均衡器。

    将请求分配给当前连接数最少的提供商，
    适合长连接或处理时间差异较大的场景。
    """

    def select_provider(self, providers: Optional[List[Provider]] = None) -> Optional[Provider]:
        """选择当前连接数最少的提供商。

        连接数相同时优先选择平均响应时间更短的。

        Args:
            providers: 可选的提供商列表。

        Returns:
            选中的提供商，无可用时返回 None。
        """
        candidates = providers if providers is not None else self._get_healthy_providers()
        if not candidates:
            logger.warning("无可用提供商")
            return None

        selected = min(
            candidates,
            key=lambda p: (p.current_connections, p.avg_response_time),
        )
        logger.debug(
            "最少连接选择提供商: %s (当前连接: %d)",
            selected.name,
            selected.current_connections,
        )
        return selected


class LoadBalancerManager:
    """负载均衡管理器。

    统一管理多种负载均衡策略，提供线程安全的提供商选择和状态更新。
    """

    STRATEGY_MAP = {
        "round_robin": RoundRobinBalancer,
        "weighted": WeightedBalancer,
        "least_connection": LeastConnectionBalancer,
    }

    def __init__(self, strategy: str = "round_robin") -> None:
        """初始化负载均衡管理器。

        Args:
            strategy: 均衡策略名称，支持 round_robin/weighted/least_connection。

        Raises:
            ValueError: 不支持的策略名称。
        """
        if strategy not in self.STRATEGY_MAP:
            raise ValueError(
                f"不支持的负载均衡策略: {strategy}，"
                f"可选: {list(self.STRATEGY_MAP.keys())}"
            )
        self._strategy_name = strategy
        self._balancer: BaseBalancer = self.STRATEGY_MAP[strategy]()
        self._lock = asyncio.Lock()
        logger.info("负载均衡管理器初始化，策略: %s", strategy)

    @property
    def strategy(self) -> str:
        """当前均衡策略名称。"""
        return self._strategy_name

    def set_providers(self, providers: List[Provider]) -> None:
        """设置提供商列表。

        Args:
            providers: 提供商列表。
        """
        self._balancer.set_providers(providers)

    async def select_provider(self) -> Optional[Provider]:
        """线程安全地选择一个提供商。

        Returns:
            选中的提供商，无可用时返回 None。
        """
        async with self._lock:
            return self._balancer.select_provider()

    async def record_success(self, provider_name: str, response_time: float) -> None:
        """记录提供商请求成功。

        Args:
            provider_name: 提供商名称。
            response_time: 响应时间（秒）。
        """
        async with self._lock:
            for p in self._balancer._providers:
                if p.name == provider_name:
                    p.record_success(response_time)
                    break

    async def record_failure(self, provider_name: str) -> None:
        """记录提供商请求失败。

        Args:
            provider_name: 提供商名称。
        """
        async with self._lock:
            for p in self._balancer._providers:
                if p.name == provider_name:
                    p.record_failure()
                    break

    async def acquire_connection(self, provider_name: str) -> None:
        """增加提供商的当前连接数。

        Args:
            provider_name: 提供商名称。
        """
        async with self._lock:
            for p in self._balancer._providers:
                if p.name == provider_name:
                    p.current_connections += 1
                    break

    async def release_connection(self, provider_name: str) -> None:
        """减少提供商的当前连接数。

        Args:
            provider_name: 提供商名称。
        """
        async with self._lock:
            for p in self._balancer._providers:
                if p.name == provider_name:
                    p.current_connections = max(0, p.current_connections - 1)
                    break

    def get_stats(self) -> Dict[str, Any]:
        """返回负载统计信息。

        Returns:
            包含策略和各提供商统计的字典。
        """
        stats = self._balancer.get_stats()
        stats["strategy"] = self._strategy_name
        return stats

    def sync_from_vllm_client(self, vllm_client: Any) -> None:
        """从 VLLMClient 同步实例统计（I-2 fix：桥接模式）。

        VLLMClient 内部已有完整的负载均衡、健康检查和熔断器，
        此方法将其实例统计同步到 LoadBalancerManager，使
        /api/enhanced/load-balancer/stats 端点能返回真实数据，
        而非之前因从未 set_providers 导致的空统计。

        Args:
            vllm_client: VLLMClient 实例。
        """
        try:
            instances = getattr(vllm_client, "_instances", [])
            if not instances:
                return

            # 将 VLLMInstance 映射为 load_balancer 的 Provider
            providers: List[Provider] = []
            for inst in instances:
                status = ProviderStatus.HEALTHY
                if inst.status.value != "healthy":
                    status = ProviderStatus.UNHEALTHY
                providers.append(Provider(
                    name=inst.name,
                    url=inst.base_url,
                    weight=inst.weight,
                    status=status,
                    current_connections=inst.current_connections,
                    total_requests=inst.total_requests,
                    success_count=inst.success_count,
                    failure_count=inst.failure_count,
                    consecutive_failures=inst.consecutive_failures,
                    total_response_time=inst.total_response_time,
                    last_failure_time=inst.last_failure_time,
                    last_used_time=inst.last_used_time,
                ))
            self._balancer.set_providers(providers)
            logger.debug("已从 VLLMClient 同步 %d 个实例统计", len(providers))
        except Exception as exc:
            logger.warning("从 VLLMClient 同步统计失败: %s", exc)
