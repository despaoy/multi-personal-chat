"""
vLLM 推理服务客户端适配层

通过 OpenAI 兼容 API 与 vLLM 服务通信，支持多实例负载均衡、
LoRA 动态切换、流式/非流式响应、健康检查与自动故障转移。

使用 httpx 异步客户端，不依赖 openai SDK。
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from infra.circuit_breaker import (
    CircuitBreaker, CircuitOpenError, CircuitState, DegradationMode,
)
from interfaces import InferenceInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类与枚举
# ---------------------------------------------------------------------------

class BalancerStrategy(str, Enum):
    """负载均衡策略"""
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTION = "least_connection"


class InstanceStatus(str, Enum):
    """vLLM 实例健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class VLLMInstance:
    """vLLM 实例信息"""
    name: str
    base_url: str
    weight: float = 1.0
    status: InstanceStatus = InstanceStatus.HEALTHY
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
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def avg_response_time(self) -> float:
        """平均响应时间（秒）"""
        if self.success_count == 0:
            return float("inf")
        return self.total_response_time / self.success_count

    def record_success(self, response_time: float) -> None:
        """记录一次成功请求"""
        self.total_requests += 1
        self.success_count += 1
        self.consecutive_failures = 0
        self.total_response_time += response_time
        self.last_used_time = time.monotonic()
        if self.status == InstanceStatus.UNHEALTHY:
            self.status = InstanceStatus.HEALTHY
            logger.info("实例 %s 已恢复健康", self.name)

    def record_failure(self) -> None:
        """记录一次失败请求"""
        self.total_requests += 1
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= 3:
            self.status = InstanceStatus.UNHEALTHY
            logger.warning(
                "实例 %s 连续失败 %d 次，标记为不可用",
                self.name, self.consecutive_failures,
            )

    def try_recover(self) -> bool:
        """尝试恢复不健康的实例（30秒冷却后）

        Returns:
            是否成功恢复
        """
        if self.status != InstanceStatus.UNHEALTHY:
            return False
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= 30.0:
            self.status = InstanceStatus.HEALTHY
            self.consecutive_failures = 0
            logger.info("实例 %s 冷却结束，重新标记为健康", self.name)
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """返回实例统计信息"""
        return {
            "name": self.name,
            "base_url": self.base_url,
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


# ---------------------------------------------------------------------------
# 负载均衡器
# ---------------------------------------------------------------------------

class _WeightedRoundRobinBalancer:
    """加权轮询负载均衡器（Nginx 平滑加权轮询算法）"""

    def __init__(self) -> None:
        self._current_weights: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _calculate_dynamic_weight(self, instance: VLLMInstance) -> float:
        """根据响应时间和成功率计算动态权重

        权重 = 基础权重 * 成功率 / (1 + 平均响应时间)
        """
        base_weight = instance.weight
        success_factor = instance.success_rate
        avg_rt = instance.avg_response_time
        time_factor = 0.1 if avg_rt == float("inf") else 1.0 / (1.0 + avg_rt)
        return base_weight * success_factor * time_factor

    async def select(self, instances: List[VLLMInstance]) -> Optional[VLLMInstance]:
        """选择一个实例"""
        async with self._lock:
            for inst in instances:
                if inst.name not in self._current_weights:
                    self._current_weights[inst.name] = 0.0

            total_weight = sum(self._calculate_dynamic_weight(i) for i in instances)
            if total_weight <= 0:
                return instances[0] if instances else None

            best: Optional[VLLMInstance] = None
            best_weight = -1.0

            for inst in instances:
                dynamic_w = self._calculate_dynamic_weight(inst)
                self._current_weights[inst.name] += dynamic_w
                if self._current_weights[inst.name] > best_weight:
                    best_weight = self._current_weights[inst.name]
                    best = inst

            if best is not None:
                self._current_weights[best.name] -= total_weight

            return best


class _LeastConnectionBalancer:
    """最少连接数负载均衡器"""

    async def select(self, instances: List[VLLMInstance]) -> Optional[VLLMInstance]:
        """选择当前连接数最少的实例，连接数相同时优先响应时间更短的"""
        if not instances:
            return None
        return min(
            instances,
            key=lambda i: (i.current_connections, i.avg_response_time),
        )


# ---------------------------------------------------------------------------
# VLLMClient 主类
# ---------------------------------------------------------------------------

class VLLMClient:
    """vLLM 推理服务客户端

    通过 OpenAI 兼容 API 与 vLLM 服务通信，支持：
    - 多实例负载均衡（加权轮询 / 最少连接数）
    - LoRA 动态切换（vLLM 原生 --enable-lora）
    - 流式和非流式响应
    - 健康检查和自动故障转移
    - 请求重试（指数退避）

    配置通过环境变量：
    - VLLM_BASE_URLS: 逗号分隔的vLLM实例URL
    - VLLM_API_KEY: API密钥（可选）
    - VLLM_MODEL: 模型名称
    - VLLM_TIMEOUT: 请求超时秒数
    - VLLM_MAX_RETRIES: 最大重试次数
    """

    # 连续失败多少次后标记为不可用
    _UNHEALTHY_THRESHOLD = 3
    # 不可用实例的冷却时间（秒）
    _RECOVERY_COOLDOWN = 30.0
    # 重试的基础延迟（秒）
    _RETRY_BASE_DELAY = 1.0
    # 重试的最大延迟（秒）
    _RETRY_MAX_DELAY = 30.0

    def __init__(
        self,
        base_urls: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        strategy: BalancerStrategy = BalancerStrategy.WEIGHTED_ROUND_ROBIN,
    ) -> None:
        """初始化 vLLM 客户端

        Args:
            base_urls: vLLM 实例 URL 列表，None 则从环境变量读取
            api_key: API 密钥，None 则从环境变量读取
            model: 模型名称，None 则从环境变量读取
            timeout: 请求超时秒数，None 则从环境变量读取
            max_retries: 最大重试次数，None 则从环境变量读取
            strategy: 负载均衡策略
        """
        # 从环境变量读取配置
        raw_urls = (
            base_urls
            or os.getenv("VLLM_BASE_URLS", "").strip()
            or os.getenv("VLLM_BASE_URL", "").strip()
            or "http://localhost:8001"
        )
        self._base_urls: List[str] = [
            u.strip().rstrip("/") for u in raw_urls.split(",") if u.strip()
        ]
        self._api_key: str = api_key or os.getenv("VLLM_API_KEY", "")
        # D-4 fix: 统一使用 get_vllm_served_model_name() 解析模型名
        from app.config import get_vllm_served_model_name
        self._model: str = model or get_vllm_served_model_name()
        self._timeout: float = timeout or float(os.getenv("VLLM_TIMEOUT", "120"))
        self._max_retries: int = max_retries or int(os.getenv("VLLM_MAX_RETRIES", "3"))
        self._max_concurrency: int = int(os.getenv("VLLM_MAX_CONCURRENCY", "8"))
        self._request_semaphore = asyncio.Semaphore(max(1, self._max_concurrency))

        # 初始化实例列表
        self._instances: List[VLLMInstance] = []
        for idx, url in enumerate(self._base_urls):
            self._instances.append(VLLMInstance(
                name=f"vllm-{idx}",
                base_url=url,
            ))

        # 初始化负载均衡器
        self._strategy = strategy
        if strategy == BalancerStrategy.LEAST_CONNECTION:
            self._balancer = _LeastConnectionBalancer()
        else:
            self._balancer = _WeightedRoundRobinBalancer()

        # httpx 异步客户端（懒初始化）
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

        # 实例操作锁（异步安全）
        self._instance_lock = asyncio.Lock()

        # 熔断器：保护 vLLM 调用，20次连续失败后熔断，60秒后半开
        # 优先注册到全局 circuit_breaker_registry，使 enhanced/stats 端点可见
        self._circuit_breaker = CircuitBreaker(
            name="vllm",
            failure_threshold=20,
            recovery_timeout=60.0,
            half_open_max_calls=5,
            degradation_mode=DegradationMode.RAISE,
        )
        self._register_to_global_registry()

        logger.info(
            "VLLMClient 初始化: %d 个实例, 模型=%s, 策略=%s",
            len(self._instances), self._model, strategy.value,
        )

    # ------------------------------------------------------------------
    # httpx 客户端管理
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        """确保 httpx.AsyncClient 已创建"""
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self._timeout, connect=10.0),
                        limits=httpx.Limits(
                            max_connections=50,
                            max_keepalive_connections=20,
                        ),
                    )
        return self._client

    def _register_to_global_registry(self) -> None:
        """将熔断器注册到全局 circuit_breaker_registry，使 enhanced/stats 端点可见。

        此前 VLLMClient 内部自建 CircuitBreaker 但未注册到 registry，
        导致 /api/enhanced/circuit-breaker/stats 看不到 vllm 的熔断状态。
        """
        try:
            from app.config import circuit_breaker_registry
            if circuit_breaker_registry is not None:
                # 直接注入已有实例，避免 registry 创建新实例导致状态隔离
                registry = circuit_breaker_registry
                if not hasattr(registry, '_breakers'):
                    return
                if "vllm" not in registry._breakers:
                    registry._breakers["vllm"] = self._circuit_breaker
                    logger.info("VLLMClient 熔断器已注册到全局 registry")
        except Exception as e:
            logger.debug(f"注册熔断器到全局 registry 失败（非致命）: {e}")

    async def close(self) -> None:
        """关闭连接池，释放资源"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("VLLMClient 连接池已关闭")

    # ------------------------------------------------------------------
    # 实例选择与健康检查
    # ------------------------------------------------------------------

    async def _get_healthy_instances(self) -> List[VLLMInstance]:
        """获取健康的实例列表，同时尝试恢复冷却结束的实例"""
        async with self._instance_lock:
            healthy: List[VLLMInstance] = []
            for inst in self._instances:
                inst.try_recover()
                if inst.status == InstanceStatus.HEALTHY:
                    healthy.append(inst)
            return healthy

    async def _select_instance(self) -> Optional[VLLMInstance]:
        """通过负载均衡器选择一个健康实例"""
        healthy = await self._get_healthy_instances()
        if not healthy:
            logger.warning("无可用 vLLM 实例")
            return None
        return await self._balancer.select(healthy)

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ------------------------------------------------------------------
    # 核心推理方法
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: List[Dict[str, str]],
        lora_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        top_p: float = 0.9,
    ) -> Any:
        """生成回复

        Args:
            messages: OpenAI 格式的消息列表
            lora_name: LoRA 适配器名称（vLLM 启动时 --enable-lora 加载的名称）
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            stream: 是否使用流式响应
            top_p: Top-p 采样参数

        Returns:
            非流式: 返回生成的文本字符串
            流式: 返回 AsyncGenerator，逐 token 产出

        Raises:
            RuntimeError: 所有实例不可用或所有重试失败
        """
        if stream:
            return self._generate_stream_with_circuit(
                messages, lora_name, temperature, max_tokens, top_p
            )
        async with self._request_semaphore:
            return await self._circuit_breaker.call(
                self._generate_non_stream,
                messages, lora_name, temperature, max_tokens, top_p
            )

    async def _generate_non_stream(
        self,
        messages: List[Dict[str, str]],
        lora_name: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> str:
        """非流式生成"""
        model_name = self._resolve_model_name(lora_name)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            instance = await self._select_instance()
            if instance is None:
                raise RuntimeError("所有 vLLM 实例不可用")

            # 增加连接计数
            async with self._instance_lock:
                instance.current_connections += 1

            start_time = time.monotonic()
            try:
                client = await self._ensure_client()
                resp = await client.post(
                    f"{instance.base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._build_headers(),
                )

                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    elapsed = time.monotonic() - start_time
                    async with self._instance_lock:
                        instance.record_success(elapsed)
                    return text

                # 5xx 服务器错误可重试
                if resp.status_code >= 500:
                    last_error = RuntimeError(
                        f"vLLM 返回 {resp.status_code}: {resp.text[:200]}"
                    )
                    async with self._instance_lock:
                        instance.record_failure()
                    if attempt < self._max_retries - 1:
                        delay = self._compute_backoff(attempt)
                        logger.warning(
                            "实例 %s 返回 %d (尝试 %d/%d)，%.1fs 后重试",
                            instance.name, resp.status_code,
                            attempt + 1, self._max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                    continue

                # 4xx 客户端错误不重试
                raise RuntimeError(
                    f"vLLM 返回 {resp.status_code}: {resp.text[:200]}"
                )

            except httpx.TimeoutException as exc:
                last_error = exc
                async with self._instance_lock:
                    instance.record_failure()
                if attempt < self._max_retries - 1:
                    delay = self._compute_backoff(attempt)
                    logger.warning(
                        "实例 %s 请求超时 (尝试 %d/%d)，%.1fs 后重试",
                        instance.name, attempt + 1, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                continue

            except httpx.ConnectError as exc:
                last_error = exc
                async with self._instance_lock:
                    instance.record_failure()
                if attempt < self._max_retries - 1:
                    delay = self._compute_backoff(attempt)
                    logger.warning(
                        "实例 %s 连接失败 (尝试 %d/%d)，%.1fs 后重试",
                        instance.name, attempt + 1, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                continue

            finally:
                async with self._instance_lock:
                    instance.current_connections = max(
                        0, instance.current_connections - 1
                    )

        raise last_error or RuntimeError("vLLM 请求失败: 未知错误")

    async def _generate_stream_with_circuit(
        self,
        messages: List[Dict[str, str]],
        lora_name: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> AsyncGenerator[str, None]:
        """流式生成，带熔断器保护"""
        async with self._request_semaphore:
            if self._circuit_breaker.state == CircuitState.OPEN:
                raise CircuitOpenError("vLLM circuit is open")
            try:
                async for chunk in self._generate_stream(
                    messages, lora_name, temperature, max_tokens, top_p
                ):
                    yield chunk
                await self._circuit_breaker.record_success()
            except Exception as e:
                await self._circuit_breaker.record_failure(str(e))
                raise

    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        lora_name: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> AsyncGenerator[str, None]:
        """流式生成，逐 token 产出"""
        model_name = self._resolve_model_name(lora_name)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }

        # 流式请求不重试（连接建立后无法回退）
        instance = await self._select_instance()
        if instance is None:
            raise RuntimeError("所有 vLLM 实例不可用")

        async with self._instance_lock:
            instance.current_connections += 1

        start_time = time.monotonic()
        try:
            client = await self._ensure_client()
            async with client.stream(
                "POST",
                f"{instance.base_url}/v1/chat/completions",
                json=payload,
                headers=self._build_headers(),
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    async with self._instance_lock:
                        instance.record_failure()
                    raise RuntimeError(
                        f"vLLM 流式返回 {resp.status_code}: "
                        f"{error_body.decode('utf-8', errors='replace')[:200]}"
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

            elapsed = time.monotonic() - start_time
            async with self._instance_lock:
                instance.record_success(elapsed)

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            async with self._instance_lock:
                instance.record_failure()
            raise RuntimeError(f"vLLM 流式请求失败: {exc}") from exc

        finally:
            async with self._instance_lock:
                instance.current_connections = max(
                    0, instance.current_connections - 1
                )

    async def generate_with_rag(
        self,
        messages: List[Dict[str, str]],
        rag_context: str,
        lora_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """带 RAG 上下文的生成

        将 RAG 检索结果注入 system 消息，然后调用 generate。

        Args:
            messages: 消息列表
            rag_context: RAG 检索到的上下文文本
            lora_name: LoRA 适配器名称
            temperature: 采样温度
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本
        """
        rag_system = {
            "role": "system",
            "content": f"参考资料：\n{rag_context[:2000]}",
        }
        # 如果已有 system 消息，将 RAG 上下文追加到其中
        augmented = list(messages)
        has_system = False
        for i, msg in enumerate(augmented):
            if msg.get("role") == "system":
                augmented[i] = {
                    "role": "system",
                    "content": f"{msg['content']}\n\n参考资料：\n{rag_context[:2000]}",
                }
                has_system = True
                break
        if not has_system:
            augmented.insert(0, rag_system)

        return await self.generate(
            messages=augmented,
            lora_name=lora_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

    # ------------------------------------------------------------------
    # 健康检查与状态查询
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """检查所有实例的健康状态

        Returns:
            包含每个实例健康状态和总体状态的字典
        """
        results: Dict[str, Any] = {}
        healthy_count = 0

        for inst in self._instances:
            try:
                client = await self._ensure_client()
                resp = await client.get(
                    f"{inst.base_url}/health",
                    timeout=httpx.Timeout(5.0, connect=3.0),
                )
                is_healthy = resp.status_code == 200
                results[inst.name] = {
                    "url": inst.base_url,
                    "healthy": is_healthy,
                    "status_code": resp.status_code,
                }
                if is_healthy:
                    healthy_count += 1
            except Exception as exc:
                results[inst.name] = {
                    "url": inst.base_url,
                    "healthy": False,
                    "error": str(exc),
                }

        total = len(self._instances)
        results["summary"] = {
            "total": total,
            "healthy": healthy_count,
            "unhealthy": total - healthy_count,
            "all_healthy": healthy_count == total,
        }
        return results

    async def get_active_loras(self) -> List[str]:
        """获取当前所有实例上加载的 LoRA 列表

        通过 vLLM 的 /v1/models 端点查询，筛选出 LoRA 类型的模型。

        Returns:
            LoRA 名称列表（去重）
        """
        loras: set[str] = set()

        for inst in self._instances:
            try:
                client = await self._ensure_client()
                resp = await client.get(
                    f"{inst.base_url}/v1/models",
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(10.0, connect=5.0),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for model_info in data.get("data", []):
                        model_id = model_info.get("id", "")
                        # vLLM LoRA 模型 ID 格式: base_model:lora_name
                        if ":" in model_id:
                            lora_name = model_id.split(":", 1)[1]
                            loras.add(lora_name)
                        # 也可能是独立的 LoRA 模型 ID
                        elif model_id != self._model:
                            loras.add(model_id)
            except Exception as exc:
                logger.warning("获取实例 %s 的 LoRA 列表失败: %s", inst.name, exc)

        return sorted(loras)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    async def _model_ids_for_instance(self, instance: VLLMInstance) -> Optional[List[str]]:
        try:
            client = await self._ensure_client()
            response = await client.get(
                f"{instance.base_url}/v1/models",
                headers=self._build_headers(),
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
            if response.status_code != 200:
                return None
            return [
                str(item.get("id", ""))
                for item in response.json().get("data", [])
                if item.get("id")
            ]
        except Exception as exc:
            logger.warning("Failed to list models from %s: %s", instance.name, exc)
            return None

    async def load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        """Load a trusted local LoRA on every configured vLLM instance."""
        if not lora_name or not lora_path:
            raise ValueError("LoRA name and path are required")

        loaded_instances: list[str] = []
        errors: list[str] = []
        client = await self._ensure_client()
        for instance in self._instances:
            model_ids = await self._model_ids_for_instance(instance)
            if model_ids is not None and (
                lora_name in model_ids
                or any(model_id.endswith(f":{lora_name}") for model_id in model_ids)
            ):
                loaded_instances.append(instance.name)
                continue

            try:
                response = await client.post(
                    f"{instance.base_url}/v1/load_lora_adapter",
                    json={"lora_name": lora_name, "lora_path": lora_path},
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(60.0, connect=5.0),
                )
                if response.status_code == 200:
                    loaded_instances.append(instance.name)
                else:
                    errors.append(f"{instance.name}: HTTP {response.status_code}")
            except Exception as exc:
                errors.append(f"{instance.name}: {type(exc).__name__}")

        if errors:
            logger.error(
                "LoRA runtime load failed name=%s loaded=%s errors=%s",
                lora_name,
                loaded_instances,
                errors,
            )
            raise RuntimeError("LoRA could not be loaded on every vLLM instance")

    async def unload_lora_adapter(self, lora_name: str) -> None:
        """Best-effort unload of an adapter from every vLLM instance."""
        if not lora_name:
            raise ValueError("LoRA name is required")
        client = await self._ensure_client()
        errors: list[str] = []
        for instance in self._instances:
            try:
                response = await client.post(
                    f"{instance.base_url}/v1/unload_lora_adapter",
                    json={"lora_name": lora_name},
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(30.0, connect=5.0),
                )
                if response.status_code not in {200, 404}:
                    errors.append(f"{instance.name}: HTTP {response.status_code}")
            except Exception as exc:
                errors.append(f"{instance.name}: {type(exc).__name__}")
        if errors:
            raise RuntimeError(f"LoRA unload failed: {', '.join(errors)}")

    async def list_loras(self) -> Optional[List[str]]:
        """查询vLLM中可用的LoRA适配器列表

        Returns:
            LoRA名称列表，查询失败返回None
        """
        try:
            client = await self._ensure_client()
            url = f"{self._instances[0].base_url}/v1/models" if self._instances else None
            if not url:
                return None
            resp = await client.get(url, headers=self._build_headers())
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                # 返回非基础模型的LoRA名称
                base_model = self._model
                return [m["id"] for m in models if m["id"] != base_model]
            return None
        except Exception as e:
            logger.debug(f"查询vLLM LoRA列表失败: {e}")
            return None

    def _resolve_model_name(self, lora_name: Optional[str]) -> str:
        """解析模型名称，返回 LoRA 或基础模型 ID

        vLLM 启动时通过 --lora-modules 注册的 LoRA 拥有独立模型 ID，
        直接使用该 ID 即可，无需 base:lora 格式。

        Args:
            lora_name: LoRA 适配器名称

        Returns:
            模型名称（LoRA 时直接返回 lora_name，否则返回基础模型）
        """
        if lora_name:
            return lora_name
        return self._model

    def _compute_backoff(self, attempt: int) -> float:
        """计算指数退避延迟

        Args:
            attempt: 当前重试次数（从 0 开始）

        Returns:
            延迟时间（秒）
        """
        import random
        delay = min(
            self._RETRY_BASE_DELAY * (2 ** attempt),
            self._RETRY_MAX_DELAY,
        )
        # 加入随机抖动，避免惊群
        delay *= random.uniform(0.5, 1.5)
        return delay

    async def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息

        Returns:
            包含配置、策略和各实例统计的字典
        """
        async with self._instance_lock:
            instances_info = [inst.to_dict() for inst in self._instances]
            healthy = sum(
                1 for i in self._instances
                if i.status == InstanceStatus.HEALTHY
            )

        return {
            "model": self._model,
            "strategy": self._strategy.value,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
            "max_concurrency": self._max_concurrency,
            "total_instances": len(self._instances),
            "healthy_instances": healthy,
            "instances": instances_info,
            "circuit_breaker": self._circuit_breaker.get_stats(),
        }


# 接口契约验证：确保 VLLMClient 的方法签名与 InferenceInterface 一致。
# 原实现 _check_interface() 在导入时实例化 VLLMClient("", "")（空实例 + 日志副作用），
# 且对非 @runtime_checkable 的 Protocol 做 isinstance 永真/永假，校验无意义。
# 改为静态方法签名比对，避免导入副作用。
def _check_interface() -> None:
    expected = {
        "generate", "generate_with_rag", "health_check", "get_active_loras",
        "close",
    }
    missing = expected - set(dir(VLLMClient))
    if missing:
        raise TypeError(
            f"VLLMClient 未实现 InferenceInterface 要求的方法: {missing}"
        )

_check_interface()


# ---------------------------------------------------------------------------
# 模块级单例：全进程共享同一个 VLLMClient 实例
# 此前 api/generate.py 和 bot/bot.py 各自创建独立 VLLMClient 实例，
# 导致连接池重复创建且熔断器状态隔离。现统一为单例。
# ---------------------------------------------------------------------------

_shared_client: Optional["VLLMClient"] = None
_shared_client_lock = asyncio.Lock()


async def get_vllm_client() -> "VLLMClient":
    """获取或创建全局共享的 VLLMClient 单例。

    从环境变量读取 VLLM_BASE_URLS / VLLM_MODEL / VLLM_TIMEOUT 进行初始化。
    所有调用方（api/generate.py、bot/bot.py）应使用此函数而非各自构造实例。
    """
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    async with _shared_client_lock:
        if _shared_client is not None:
            return _shared_client
        base_urls = os.getenv("VLLM_BASE_URLS", os.getenv("VLLM_BASE_URL", ""))
        # D-4 fix: 模型名统一通过 get_vllm_served_model_name() 解析
        from app.config import get_vllm_served_model_name
        model = get_vllm_served_model_name()
        timeout = float(os.getenv("VLLM_TIMEOUT", "120"))
        _shared_client = VLLMClient(
            base_urls=base_urls,
            model=model,
            timeout=timeout,
        )
        logger.info("VLLMClient 全局单例已创建")
        return _shared_client


async def close_shared_vllm_client() -> None:
    """关闭全局共享的 VLLMClient 单例（应用退出时调用）"""
    global _shared_client
    if _shared_client is not None:
        await _shared_client.close()
        _shared_client = None
