#!/usr/bin/env python3
"""
QQ智能助手 - 异步推理服务 (供消息管道使用)

将不同推理后端 (Mock/Ollama/vLLM) 统一为异步接口，
供 AsyncMessagePipeline 使用。

阶段3优化：
  - 语义缓存：L1进程内LRU + L2 Redis，减少重复推理
  - 熔断器保护：vLLM 连续失败自动降级
  - 连接池复用：共享 httpx.AsyncClient，避免每请求创建
  - 对话历史管理：支持多轮对话 KV Cache 复用，后续追问更快
  - 缓存回复多样化：同一问题缓存多个回复，随机选择避免机器人感
"""
import asyncio
import logging
import os
import random
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class ConversationManager:
    """对话历史管理器 — 支持多轮对话 KV Cache 复用

    QQ群聊特性适配：
    - 同一用户在同一群的对话视为一个会话
    - 保留最近 max_turns 轮对话历史
    - 会话超时自动清理（默认30分钟无消息则过期）
    - vLLM 支持 prefix caching，传完整对话历史可自动复用 KV Cache
    """

    def __init__(self, max_turns: int = 10, session_ttl: float = 1800.0, max_sessions: int = 500):
        self._max_turns = max_turns
        self._session_ttl = session_ttl
        self._max_sessions = max_sessions
        # key: f"{group_id}:{user_id}" → value: {"messages": [...], "last_active": float}
        self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get_messages(self, group_id: str, user_id: str) -> List[Dict[str, str]]:
        """获取指定会话的对话历史（用于 vLLM 推理，复用 KV Cache）

        Returns:
            OpenAI 格式的消息列表，如:
            [{"role": "user", "content": "原神好玩吗"},
             {"role": "assistant", "content": "原神非常好玩！"},
             {"role": "user", "content": "那胡桃呢"}]
        """
        key = f"{group_id}:{user_id}"
        session = self._sessions.get(key)
        if session is None:
            return []
        # 检查是否过期
        if time.monotonic() - session["last_active"] > self._session_ttl:
            del self._sessions[key]
            return []
        # 移到末尾（LRU）
        self._sessions.move_to_end(key)
        return list(session["messages"])

    def add_exchange(self, group_id: str, user_id: str, user_msg: str, assistant_msg: str) -> None:
        """添加一轮对话到历史"""
        key = f"{group_id}:{user_id}"
        if key in self._sessions:
            session = self._sessions[key]
            session["messages"].append({"role": "user", "content": user_msg})
            session["messages"].append({"role": "assistant", "content": assistant_msg})
            session["last_active"] = time.monotonic()
            self._sessions.move_to_end(key)
            # 保留最近 max_turns 轮（一轮 = user + assistant = 2条）
            max_msgs = self._max_turns * 2
            if len(session["messages"]) > max_msgs:
                session["messages"] = session["messages"][-max_msgs:]
        else:
            # 新会话，先清理最旧的会话（LRU淘汰）
            self._evict_if_needed()
            self._sessions[key] = {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "last_active": time.monotonic(),
            }

    def _evict_if_needed(self) -> None:
        """LRU淘汰最旧的会话"""
        while len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)

    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量"""
        now = time.monotonic()
        expired_keys = [
            k for k, v in self._sessions.items()
            if now - v["last_active"] > self._session_ttl
        ]
        for k in expired_keys:
            del self._sessions[k]
        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self._max_sessions,
            "max_turns": self._max_turns,
            "session_ttl": self._session_ttl,
        }


class AsyncInferenceService:
    """异步推理服务 — 统一接口，支持多后端"""

    def __init__(
        self,
        backend: str = "mock",
        ollama_url: str = None,
        vllm_url: str = None,
        openai_url: str = None,
        openai_key: str = None,
        model_name: str = None,
        timeout: float = None,
    ):
        # m-4 fix: 默认值从环境变量读取，与 bot/bot.py Config 类保持一致
        self.backend = backend
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        raw_vllm_url = vllm_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001")
        self.vllm_url = raw_vllm_url.rstrip("/")
        if not self.vllm_url.endswith("/v1"):
            self.vllm_url += "/v1"
        raw_openai_url = openai_url or os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
        self.openai_url = raw_openai_url.rstrip("/")
        if not self.openai_url.endswith("/v1"):
            self.openai_url += "/v1"
        self.openai_key = openai_key or os.getenv("OPENAI_COMPAT_API_KEY", "")
        if model_name:
            self.model_name = model_name
        elif backend == "vllm":
            self.model_name = os.getenv("VLLM_SERVED_MODEL_NAME", os.getenv("VLLM_MODEL", "qwen3-8b"))
        elif backend == "openai":
            self.model_name = os.getenv("OPENAI_COMPAT_MODEL", "deepseek-chat")
        else:
            self.model_name = os.getenv("OLLAMA_MODEL", "qwen3-8b")
        self.timeout = timeout if timeout is not None else float(os.getenv("MODEL_INFERENCE_TIMEOUT", "180"))
        self._stats = {"total": 0, "success": 0, "failed": 0, "total_latency": 0.0, "cache_hits": 0}

        # 共享 httpx 客户端（懒初始化）
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

        # 语义缓存（懒初始化）
        self._semantic_cache = None

        # 熔断器（懒初始化，仅 vLLM 后端使用）
        self._circuit_breaker = None

        # 对话历史管理器（KV Cache 复用）
        self._conversation_mgr = ConversationManager(
            max_turns=int(os.getenv("CONVERSATION_MAX_TURNS", "10")),
            session_ttl=float(os.getenv("CONVERSATION_SESSION_TTL", "1800")),
            max_sessions=int(os.getenv("CONVERSATION_MAX_SESSIONS", "500")),
        )

        # 缓存回复多样化：同一问题最多缓存几个不同回复
        self._cache_variants = int(os.getenv("CACHE_VARIANTS", "3"))

    async def _ensure_client(self) -> httpx.AsyncClient:
        """确保 httpx.AsyncClient 已创建（连接池复用）"""
        if self._client is not None and not self._client.is_closed:
            return self._client
        async with self._client_lock:
            if self._client is not None and not self._client.is_closed:
                return self._client
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
            )
            return self._client

    async def _ensure_cache(self):
        """懒初始化语义缓存"""
        if self._semantic_cache is None:
            try:
                from cache.semantic_cache import get_semantic_cache
                self._semantic_cache = await get_semantic_cache()
            except Exception as e:
                logger.warning(f"语义缓存初始化失败: {e}")

    async def _ensure_circuit_breaker(self):
        """懒初始化熔断器"""
        if self._circuit_breaker is None:
            try:
                from infra.circuit_breaker import CircuitBreaker, DegradationMode
                self._circuit_breaker = CircuitBreaker(
                    name="inference_service",
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    half_open_max_calls=2,
                    degradation_mode=DegradationMode.DEFAULT,
                )
                self._circuit_breaker.set_default("[系统提示] 暂时无法处理您的消息，请稍后再试")
            except Exception as e:
                logger.warning(f"熔断器初始化失败: {e}")

    async def infer(self, group_id: str, user_id: str, message: str) -> str:
        """异步推理 — 支持语义缓存、对话历史(KV Cache复用)、缓存多样化、熔断保护"""
        start = time.monotonic()
        self._stats["total"] += 1

        # 检查语义缓存（支持多样化：缓存可能是列表，随机选一个）
        await self._ensure_cache()
        if self._semantic_cache:
            try:
                cached = await self._semantic_cache.get(message, context=f"g:{group_id}")
                if cached is not None:
                    self._stats["cache_hits"] += 1
                    self._stats["success"] += 1
                    # 缓存多样化：如果是列表则随机选一个回复
                    if isinstance(cached, list) and len(cached) > 0:
                        reply = random.choice(cached)
                        logger.debug(f"语义缓存命中(多样化): {message[:30]}... 从{len(cached)}个候选中随机选择")
                        return reply
                    logger.debug(f"语义缓存命中: {message[:30]}...")
                    return cached
            except Exception as e:
                logger.debug(f"缓存查询失败: {e}")

        # 熔断器保护
        await self._ensure_circuit_breaker()
        if self._circuit_breaker and self._circuit_breaker.state.value == "open":
            logger.warning("熔断器开启，执行降级")
            self._stats["failed"] += 1
            return "[系统提示] 推理服务暂时不可用，请稍后再试"

        try:
            # 获取对话历史（用于 KV Cache 复用）
            history = self._conversation_mgr.get_messages(group_id, user_id)
            # 构建完整消息列表：历史 + 当前消息
            messages = history + [{"role": "user", "content": message}] if history else None

            if self.backend == "mock":
                reply = await self._infer_mock(message)
            elif self.backend == "ollama":
                reply = await self._infer_ollama(message, messages=messages)
            elif self.backend == "vllm":
                reply = await self._infer_vllm(message, messages=messages)
            elif self.backend == "openai_compat":
                reply = await self._infer_openai(message, messages=messages)
            else:
                reply = await self._infer_mock(message)

            self._stats["success"] += 1
            self._stats["total_latency"] += time.monotonic() - start

            # 记录对话历史（用于后续 KV Cache 复用）
            self._conversation_mgr.add_exchange(group_id, user_id, message, reply)

            # 写入缓存（支持多样化：同一问题缓存多个不同回复）
            if self._semantic_cache:
                try:
                    cache_key = message
                    cache_ctx = f"g:{group_id}"
                    # 检查是否已有缓存
                    existing = await self._semantic_cache.get(cache_key, context=cache_ctx)
                    if isinstance(existing, list):
                        # 已有多个候选回复，追加新回复（去重）
                        if reply not in existing and len(existing) < self._cache_variants:
                            existing.append(reply)
                            await self._semantic_cache.set(cache_key, existing, context=cache_ctx)
                        # 如果已经够多了，不更新
                    elif isinstance(existing, str):
                        # 已有单个回复，升级为列表
                        if reply != existing:
                            await self._semantic_cache.set(cache_key, [existing, reply], context=cache_ctx)
                        else:
                            # 完全一样的回复，保持单条
                            await self._semantic_cache.set(cache_key, existing, context=cache_ctx)
                    else:
                        # 首次缓存
                        await self._semantic_cache.set(cache_key, reply, context=cache_ctx)
                except Exception as e:
                    logger.debug(f"缓存写入失败: {e}")

            # 熔断器记录成功
            if self._circuit_breaker:
                await self._circuit_breaker.record_success()

            return reply

        except Exception as e:
            self._stats["failed"] += 1

            # 熔断器记录失败
            if self._circuit_breaker:
                await self._circuit_breaker.record_failure(e)

            logger.error(f"推理失败 [{self.backend}]: {e}")
            return "[系统提示] 暂时无法处理您的消息，请稍后再试"

    async def _infer_mock(self, message: str) -> str:
        """Mock 推理 (开发测试用)"""
        import random
        await asyncio.sleep(random.uniform(0.1, 0.3))
        replies = [
            f"收到你的消息啦～关于「{message[:20]}」我来想想怎么回复你～",
            f"哈哈，这个问题有意思！{message[:20]}... 😄",
            f"嗯嗯，我理解你的意思～ {message[:20]}... 让我帮你看看！",
        ]
        return random.choice(replies)

    async def _infer_ollama(self, message: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Ollama 推理（支持对话历史）"""
        request_messages = messages if messages else [{"role": "user", "content": message}]
        client = await self._ensure_client()
        resp = await client.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": request_messages,
                "stream": False,
            },
        )
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
        raise RuntimeError(f"Ollama 错误: {resp.status_code}")

    async def _infer_vllm(self, message: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """vLLM 推理 (OpenAI 兼容协议，连接池复用，支持对话历史 KV Cache 复用)

        Args:
            message: 当前用户消息
            messages: 完整对话历史（含当前消息），传给 vLLM 可复用 KV Cache
        """
        max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        temperature = float(os.getenv("TEMPERATURE", "0.85"))

        # 如果有对话历史，使用完整历史（vLLM 自动复用相同前缀的 KV Cache）
        # 如果没有历史，使用单条消息（向后兼容）
        request_messages = messages if messages else [{"role": "user", "content": message}]

        client = await self._ensure_client()
        resp = await client.post(
            f"{self.vllm_url}/chat/completions",
            json={
                "model": self.model_name,
                "messages": request_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.92,
            },
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"vLLM 错误: {resp.status_code} - {resp.text[:200]}")

    async def _infer_openai(self, message: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """OpenAI 兼容 API 推理（支持对话历史）"""
        if not self.openai_key:
            raise RuntimeError("未配置 OpenAI API Key")

        request_messages = messages if messages else [{"role": "user", "content": message}]
        client = await self._ensure_client()
        resp = await client.post(
            f"{self.openai_url}/chat/completions",
            json={
                "model": self.model_name,
                "messages": request_messages,
                "max_tokens": 512,
                "temperature": 0.8,
            },
            headers={"Authorization": f"Bearer {self.openai_key}"},
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"OpenAI 错误: {resp.status_code}")

    async def close(self):
        """关闭连接池"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        if self._semantic_cache:
            await self._semantic_cache.close()

    def get_stats(self) -> dict:
        """获取推理统计"""
        s = dict(self._stats)
        processed = max(s.get("success", 1), 1)
        stats = {
            **s,
            "avg_latency_ms": round(s["total_latency"] / processed * 1000, 1) if processed else 0,
            "backend": self.backend,
            "cache_hit_rate": round(
                s.get("cache_hits", 0) / max(s.get("total", 1), 1) * 100, 1
            ),
            "cache_variants": self._cache_variants,
            "conversation": self._conversation_mgr.get_stats(),
        }
        return stats
