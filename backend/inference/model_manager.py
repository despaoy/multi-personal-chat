#!/usr/bin/env python3
"""
模型管理模块
负责多提供商模型推理、本地模型文件管理、LoRA适配器切换。
支持 Ollama / llama.cpp / Transformers+PEFT / Mock 四种提供商。
"""

import os
import time
import json
import logging
import random
import threading
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# DB 配置缓存统一委托给 cache.config_cache（60s TTL + jitter + Redis 共享），
# 消除三套独立缓存导致的状态不同步问题。
# 此前 model_manager 维护独立 5s 缓存，与 config_cache 的 60s 缓存失效不联动，
# 导致配置更新后 inference 层与 API 层行为不一致。
_db_config_lock = threading.Lock()


def _coerce_config_value(value):
    """Convert persisted config strings to primitive Python values.

    M-3 fix: 委托到 db.config_utils.coerce_config_value，消除重复实现。
    保留此函数以兼容现有调用方（_get_db_config 内部使用）。
    """
    from db.config_utils import coerce_config_value
    return coerce_config_value(value)


def _get_db_config():
    """Read model config through the unified cache.config_cache layer.

    统一入口：先查 Redis/local 缓存（60s TTL + jitter），未命中则从 db 加载并回填缓存。
    这样 api/config.py 调用 invalidate_config_cache() 后，所有模块下次读取都会重新加载。
    """
    with _db_config_lock:
        try:
            from cache.config_cache import get_cached_config, set_cached_config
            cached = get_cached_config()
            if cached is not None:
                return dict(cached)
        except Exception:
            pass

        try:
            from db.adapter import db
            raw_config = getattr(db, "config", {}) or {}
            result = {key: _coerce_config_value(value) for key, value in raw_config.items()}
            try:
                from cache.config_cache import set_cached_config
                set_cached_config(result)
            except Exception:
                pass
            return result.copy()
        except Exception as exc:
            logger.warning("Failed to read model config from database adapter: %s", exc)
            return {}

_db_cfg = _get_db_config()
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS_PEFT = "transformers_peft"
    OPENAI_COMPAT = "openai_compat"
    VLLM = "vllm"
    MOCK = "mock"


@dataclass
class ModelConfig:
    name: str
    repo_id: str
    model_type: str
    size: str
    description: str
    required_files: List[str]
    checksum_file: Optional[str] = None


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen3-8b": ModelConfig(
        name="Qwen3-8B-Instruct",
        repo_id="Qwen/Qwen3-8B",
        model_type="qwen",
        size="7b",
        description="Qwen3 8B参数指令微调模型",
        required_files=[
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "generation_config.json"
        ]
    ),
    "qwen2.5-3b": ModelConfig(
        name="Qwen2.5-3B-Instruct",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        model_type="qwen",
        size="3b",
        description="Qwen2.5 3B参数指令微调模型",
        required_files=[
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "generation_config.json"
        ]
    ),
    "qwen2.5-1.5b": ModelConfig(
        name="Qwen2.5-1.5B-Instruct",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_type="qwen",
        size="1.5b",
        description="Qwen2.5 1.5B参数指令微调模型",
        required_files=[
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "generation_config.json"
        ]
    ),
}


class BaseProvider(ABC):
    """模型提供商基类"""

    def __init__(self, name: str):
        self.name = name
        self._loaded = False
        self._model_name = ""
        # R-1 fix: fallback httpx.AsyncClient，仅在 http_client_pool 不可用时使用
        # （PG 模式 / 测试环境）。生产环境通过 _acquire_http_client 走共享池。
        self._fallback_async_client: Any = None

    @abstractmethod
    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """子类必须实现同步生成方法"""
        raise NotImplementedError

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成，默认通过线程池运行同步方法，子类可覆写以提供原生异步实现"""
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, session_history, rag_docs, max_tokens_override)

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "modelName": self._model_name,
        }

    def set_lora_adapter(self, lora_path: Optional[str]):
        pass

    @asynccontextmanager
    async def _acquire_http_client(self, timeout: float = 120.0):
        """获取 httpx.AsyncClient（R-1 fix：统一复用 HttpClientPool）。

        优先从 app.config.http_client_pool 获取共享客户端；
        pool 不可用时创建调用级 AsyncClient，并在创建它的事件循环中关闭。
        生产请求仍通过共享池复用连接。

        Args:
            timeout: 回退模式下 AsyncClient 的请求超时（秒）。

        Yields:
            httpx.AsyncClient 实例。
        """
        pool = None
        try:
            from app.config import http_client_pool
            pool = http_client_pool
        except Exception as exc:
            logger.debug("http_client_pool 不可用，使用 fallback: %s", exc)

        if pool is not None:
            # P1-M4 fix: pool 的 client 以固定 request_timeout 创建，此处无法
            # 逐请求覆盖。调用方应在 client.post(...) 时显式传入 timeout 参数
            # 以使 vllmTimeout 等配置生效（httpx 支持请求级 timeout 覆盖）。
            async with pool.acquire() as client:
                yield client
            return

        # Fallback is for tests and standalone scripts; close it on the creating loop.
        import httpx
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=20, max_keepalive_connections=10
            ),
        ) as client:
            yield client

    def close(self):
        """释放底层资源（httpx 客户端、模型句柄等）。

        C6 fix: 子类若持有 httpx.Client/AsyncClient 等资源，应覆写此方法。
        ModelManager.shutdown() 会在应用关闭时遍历所有 provider 调用 close()。
        R-1 fix: 共享 pool 的客户端由 pool 自行管理生命周期，此处仅清理 fallback client。
        """
        client = self._fallback_async_client
        if client is not None:
            try:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(client.aclose(), loop=loop)
                    else:
                        asyncio.run(client.aclose())
                except RuntimeError:
                    asyncio.run(client.aclose())
            except Exception:
                pass
            self._fallback_async_client = None


class OpenAICompatProvider(BaseProvider):
    """OpenAI兼容API提供商（支持DeepSeek、通义千问等）"""

    def __init__(self):
        super().__init__("openai_compat")
        self.base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
        self.api_key = os.getenv("OPENAI_COMPAT_API_KEY", "")
        self.model = os.getenv("OPENAI_COMPAT_MODEL", "deepseek-chat")
        self._model_name = self.model
        self._loaded = True
        # R-1 fix: 不再自建 httpx.Client，统一通过 _acquire_http_client() 获取共享 AsyncClient
        # 从数据库配置覆盖
        self._refresh_db_config()

    def _refresh_db_config(self):
        """从数据库刷新配置"""
        global _db_cfg
        _db_cfg = _get_db_config()
        if _db_cfg.get("openaiCompatBaseUrl"):
            self.base_url = _db_cfg["openaiCompatBaseUrl"]
        if _db_cfg.get("openaiCompatApiKey"):
            self.api_key = _db_cfg["openaiCompatApiKey"]
        if _db_cfg.get("openaiCompatModel"):
            self.model = _db_cfg["openaiCompatModel"]
            self._model_name = self.model

    async def _generate_async(self, prompt: str, session_history: List[Dict] = None,
                              rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成实现（R-1 fix: 使用共享 HttpClientPool）"""
        # 每次生成前刷新配置，确保使用最新的API Key
        self._refresh_db_config()

        if not self.api_key:
            raise RuntimeError("未配置 API Key，请在设置页面配置 OpenAI 兼容 API Key")

        start = time.time()
        messages = []

        if rag_docs:
            rag_text = "\n".join(doc.get("content", "") for doc in rag_docs[:3])
            messages.append({
                "role": "system",
                "content": f"参考资料：\n{rag_text[:800]}"
            })

        if session_history:
            messages.extend(session_history)

        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens_override if max_tokens_override else int(_db_cfg.get('maxTokens', 512))

        try:
            async with self._acquire_http_client(timeout=120.0) as client:
                # P1-M4 fix: 请求级 timeout 覆盖 pool 默认值，确保生效
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": float(_db_cfg.get('temperature', 0.8)),
                        "max_tokens": max_tokens,
                    },
                    timeout=120.0,
                )
            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"]["content"].strip()
                cost = round(time.time() - start, 2)
                return reply, cost
            else:
                raise RuntimeError(f"API返回错误: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            logger.error(f"OpenAI兼容API调用失败: {e}")
            raise

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        # R-1 fix: 同步入口通过 asyncio.run 执行异步实现
        import asyncio
        return asyncio.run(self._generate_async(prompt, session_history, rag_docs, max_tokens_override))

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成 - 原生 async，无需线程池"""
        return await self._generate_async(prompt, session_history, rag_docs, max_tokens_override)

    def close(self):
        # R-1 fix: 共享 pool 的客户端由 pool 管理，此处仅清理 fallback client
        super().close()


class MockProvider(BaseProvider):
    """模拟提供商，用于测试"""

    def __init__(self):
        super().__init__("mock")
        self._model_name = "Mock Model"
        self._loaded = True

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        start = time.time()
        replies = [
            f"好的，我来帮您处理这个问题！您说的是：{prompt[:30]}...",
            f"这个问题很有趣，让我想想... 关于：{prompt[:30]}",
            f"哈哈，这个问题有意思！{prompt[:30]}... 让我陪你聊聊～",
        ]
        reply = random.choice(replies)
        cost = round(time.time() - start + random.uniform(1.0, 3.0), 2)
        return reply, cost


class OllamaProvider(BaseProvider):
    """Ollama 提供商"""

    def __init__(self):
        super().__init__("ollama")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self._model_name = self.model
        self._loaded = True
        # R-1 fix: 不再自建 httpx.Client，统一通过 _acquire_http_client() 获取共享 AsyncClient

    async def _generate_async(self, prompt: str, session_history: List[Dict] = None,
                              rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成实现（R-1 fix: 使用共享 HttpClientPool）"""
        start = time.time()
        messages = []

        if rag_docs:
            rag_text = "\n".join(doc.get("content", "") for doc in rag_docs[:3])
            messages.append({
                "role": "system",
                "content": f"参考资料：\n{rag_text[:800]}"
            })

        if session_history:
            messages.extend(session_history)

        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens_override if max_tokens_override else int(_db_cfg.get('maxTokens', 512))

        try:
            async with self._acquire_http_client(timeout=120.0) as client:
                # P1-M4 fix: 请求级 timeout 覆盖 pool 默认值
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": float(_db_cfg.get('temperature', 0.8)), "top_p": 0.9, "num_predict": max_tokens}
                    },
                    timeout=120.0,
                )
            if response.status_code == 200:
                data = response.json()
                reply = data["message"]["content"].strip()
                cost = round(time.time() - start, 2)
                return reply, cost
            else:
                raise RuntimeError(f"Ollama返回错误状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama调用失败: {e}")
            raise

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        # R-1 fix: 同步入口通过 asyncio.run 执行异步实现
        import asyncio
        return asyncio.run(self._generate_async(prompt, session_history, rag_docs, max_tokens_override))

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成 - 原生 async，无需线程池"""
        return await self._generate_async(prompt, session_history, rag_docs, max_tokens_override)

    def close(self):
        # R-1 fix: 共享 pool 的客户端由 pool 管理，此处仅清理 fallback client
        super().close()


class LlamaCppProvider(BaseProvider):
    """llama.cpp 提供商"""

    def __init__(self):
        super().__init__("llama_cpp")
        self.base_url = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")
        self._model_name = "llama.cpp"
        self._loaded = True
        # R-1 fix: 不再自建 httpx.Client，统一通过 _acquire_http_client() 获取共享 AsyncClient

    async def _generate_async(self, prompt: str, session_history: List[Dict] = None,
                              rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成实现（R-1 fix: 使用共享 HttpClientPool）"""
        start = time.time()
        full_prompt = prompt
        if session_history:
            parts = []
            for msg in session_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            parts.append(f"User: {prompt}")
            full_prompt = "\n".join(parts)

        max_tokens = max_tokens_override if max_tokens_override else int(_db_cfg.get('maxTokens', 512))

        try:
            async with self._acquire_http_client(timeout=120.0) as client:
                # P1-M4 fix: 请求级 timeout 覆盖 pool 默认值
                response = await client.post(
                    f"{self.base_url}/completion",
                    json={
                        "prompt": full_prompt,
                        "n_predict": max_tokens,
                        "temperature": float(_db_cfg.get('temperature', 0.8)),
                        "top_p": 0.9,
                    },
                    timeout=120.0,
                )
            if response.status_code == 200:
                data = response.json()
                reply = data.get("content", "").strip()
                cost = round(time.time() - start, 2)
                return reply, cost
            else:
                raise RuntimeError(f"llama.cpp返回错误状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"llama.cpp调用失败: {e}")
            raise

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        # R-1 fix: 同步入口通过 asyncio.run 执行异步实现
        import asyncio
        return asyncio.run(self._generate_async(prompt, session_history, rag_docs, max_tokens_override))

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成 - 原生 async，无需线程池"""
        return await self._generate_async(prompt, session_history, rag_docs, max_tokens_override)

    def close(self):
        # R-1 fix: 共享 pool 的客户端由 pool 管理，此处仅清理 fallback client
        super().close()


class TransformersPeftProvider(BaseProvider):
    """Transformers + PEFT 本地推理提供商"""

    def __init__(self):
        super().__init__("transformers_peft")
        self._model = None
        self._tokenizer = None
        self._lora_path: Optional[str] = None
        self._load_lock = threading.Lock()  # 线程安全锁
        base_model_path = os.getenv("BASE_MODEL_PATH", "models/Qwen3-8B-Instruct")
        if not os.path.isabs(base_model_path):
            # 相对路径基于 backend 根目录（项目根目录下的 backend/）
            base_model_path = str(Path(__file__).parent.parent / base_model_path)
        self._base_model_path = base_model_path
        self._model_name = "Qwen3-8B-Instruct"
        self._loaded = False

    def _ensure_loaded(self):
        """线程安全的模型加载（double-check locking）"""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            base_path = str(Path(self._base_model_path).resolve())
            if not Path(base_path).exists():
                raise FileNotFoundError(f"模型路径不存在: {base_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(base_path)

            # 尝试多种加载策略（Windows bitsandbytes 兼容性）
            load_strategies = [
                ("4-bit NF4 量化", self._load_4bit),
                ("8-bit 量化", self._load_8bit),
                ("FP16 半精度", self._load_fp16),
            ]

            base_model = None
            for strategy_name, load_fn in load_strategies:
                try:
                    logger.info(f"尝试 {strategy_name} 加载 Qwen3-8B...")
                    base_model = load_fn(base_path)
                    logger.info(f"✅ {strategy_name} 加载成功")
                    break
                except Exception as e:
                    logger.warning(f"❌ {strategy_name} 加载失败: {e}")
                    continue

            if base_model is None:
                raise RuntimeError("所有加载策略均失败，请检查模型文件和 GPU 显存")

            if self._lora_path and Path(self._lora_path).exists():
                logger.info(f"加载LoRA适配器: {self._lora_path}")
                self._model = PeftModel.from_pretrained(base_model, self._lora_path)
            else:
                self._model = base_model

            self._model.eval()
            self._loaded = True
            vram = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"7B 模型加载完成，显存: {vram:.1f}GB")

    def _load_4bit(self, base_path: str):
        """4-bit NF4 量化加载"""
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=nf4_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def _load_8bit(self, base_path: str):
        """8-bit 量化加载"""
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def _load_fp16(self, base_path: str):
        """FP16 半精度加载（需要足够显存）"""
        import torch
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def set_lora_adapter(self, lora_path: Optional[str]):
        with self._load_lock:
            if lora_path and Path(lora_path).exists():
                self._lora_path = lora_path
                if self._model is not None:
                    import torch
                    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
                    from peft import PeftModel

                    logger.info(f"热切换LoRA适配器: {lora_path}")
                    base_model = self._model.base_model.model if hasattr(self._model, 'base_model') else self._model
                    del self._model
                    torch.cuda.empty_cache()
                    self._model = PeftModel.from_pretrained(base_model, lora_path)
                    self._model.eval()
                logger.info(f"LoRA适配器已设置: {lora_path}")
            else:
                self._lora_path = None
                logger.info("LoRA适配器已清除")

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        import torch

        self._ensure_loaded()
        start = time.time()

        messages = []
        if rag_docs:
            rag_text = "\n".join(doc.get("content", "") for doc in rag_docs[:3])
            messages.append({
                "role": "system",
                "content": f"参考资料：\n{rag_text[:800]}"
            })

        if session_history:
            messages.extend(session_history)

        messages.append({"role": "user", "content": prompt})

        encoded = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )

        if hasattr(encoded, 'input_ids'):
            input_ids = encoded.input_ids.to(self._model.device)
        elif isinstance(encoded, dict) and 'input_ids' in encoded:
            input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=self._model.device)
        else:
            input_ids = torch.tensor(encoded, dtype=torch.long, device=self._model.device)

        max_tokens = max_tokens_override if max_tokens_override else int(_db_cfg.get('maxTokens', 512))

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=float(_db_cfg.get('temperature', 0.85)),
                top_p=0.92,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        reply = self._tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        cost = round(time.time() - start, 2)
        return reply, cost

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "modelName": self._model_name,
            "loraAdapter": self._lora_path,
        }

    def close(self):
        # C6 fix: 释放 GPU 显存与模型句柄
        try:
            if self._model is not None:
                import torch
                del self._model
                self._model = None
                self._tokenizer = None
                self._loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("TransformersPeftProvider 模型资源已释放")
        except Exception as e:
            logger.warning(f"释放 TransformersPeftProvider 资源失败: {e}")


class VLLMProvider(BaseProvider):
    """vLLM 提供商 - 高性能推理引擎，支持 Continuous Batching"""

    def __init__(self):
        super().__init__("vllm")
        self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        # D-4 fix: 统一使用 get_vllm_served_model_name() 解析模型名
        from app.config import get_vllm_served_model_name
        self.model = get_vllm_served_model_name()
        self.timeout = float(os.getenv("VLLM_TIMEOUT", "120.0"))
        self._model_name = self.model
        self._loaded = True
        # R-1 fix: 不再自建 httpx.AsyncClient，统一通过 _acquire_http_client() 获取共享 AsyncClient
        self._lora_adapter: Optional[str] = None
        self._refresh_db_config()

    def _refresh_db_config(self):
        """从数据库刷新配置"""
        global _db_cfg
        _db_cfg = _get_db_config()
        if _db_cfg.get("vllmBaseUrl"):
            self.base_url = _db_cfg["vllmBaseUrl"]
        if _db_cfg.get("vllmModel"):
            self.model = _db_cfg["vllmModel"]
            self._model_name = self.model
        if _db_cfg.get("vllmTimeout"):
            self.timeout = float(_db_cfg["vllmTimeout"])

    def _chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def set_lora_adapter(self, lora_path: Optional[str]):
        """通过 vLLM 的 --enable-lora 传递 LoRA 名称"""
        if lora_path:
            self._lora_adapter = lora_path
            logger.info(f"vLLM LoRA 适配器已设置: {lora_path}")
        else:
            self._lora_adapter = None
            logger.info("vLLM LoRA 适配器已清除")

    async def generate_async(self, prompt: str, session_history: List[Dict] = None,
                             rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成 - 复用 VLLMClient 全局单例（负载均衡 + 熔断器 + 健康检查）。

        Critical fix: 此前 VLLMProvider 通过 HttpClientPool 直接 POST vLLM，
        绕过了 VLLMClient 的负载均衡、熔断器、健康检查，导致：
        1. 同一 vLLM 后端有 3 条独立 HTTP 客户端路径，连接数不可控
        2. VLLMProvider 路径的失败不会触发熔断器
        3. VLLMClient 的多实例负载均衡对 VLLMProvider 无效
        现统一委托给 VLLMClient.generate，消除连接池冗余和状态隔离。
        """
        self._refresh_db_config()
        start = time.time()

        messages = []
        if rag_docs:
            rag_text = "\n".join(doc.get("content", "") for doc in rag_docs[:3])
            messages.append({"role": "system", "content": f"参考资料：\n{rag_text[:800]}"})

        if session_history:
            for msg in session_history[-10:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens_override if max_tokens_override else int(_db_cfg.get('maxTokens', 512))
        temperature = float(_db_cfg.get('temperature', 0.8))

        # 复用 VLLMClient 全局单例（内部已有 3 次重试 + 熔断器 + 负载均衡）
        from inference.vllm_client import get_vllm_client
        client = await get_vllm_client()
        reply = await client.generate(
            messages=messages,
            lora_name=self._lora_adapter,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        cost = round(time.time() - start, 2)
        return reply, cost

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """同步生成包装 - 通过 asyncio.run 执行异步方法"""
        import asyncio
        return asyncio.run(self.generate_async(prompt, session_history, rag_docs, max_tokens_override))

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成入口"""
        return await self.generate_async(prompt, session_history, rag_docs, max_tokens_override)

    def close(self):
        # R-1 fix: 共享 pool 的客户端由 pool 管理，此处仅清理 fallback client
        super().close()


class ModelManager:
    """模型管理器，负责多提供商切换、LoRA管理、模型文件操作。"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.cache_file = self.models_dir / "model_cache.json"

        self._providers: Dict[ModelProvider, BaseProvider] = {
            ModelProvider.MOCK: MockProvider(),
            ModelProvider.OLLAMA: OllamaProvider(),
            ModelProvider.LLAMA_CPP: LlamaCppProvider(),
            ModelProvider.OPENAI_COMPAT: OpenAICompatProvider(),
            ModelProvider.TRANSFORMERS_PEFT: TransformersPeftProvider(),
            ModelProvider.VLLM: VLLMProvider(),
        }

        self._current_provider = ModelProvider.MOCK
        env_provider = os.getenv("MODEL_PROVIDER", "").strip().lower()
        # C11 fix: DB 默认值从 "mock" 改为 "vllm"，但保留显式 mock 配置能力（开发环境）
        db_provider = _db_cfg.get("modelProvider", "vllm")
        provider_name = env_provider or db_provider
        if provider_name in [e.value for e in ModelProvider]:
            self._current_provider = ModelProvider(provider_name)
            source = "environment" if env_provider else "database"
            logger.info(f"Initialized model provider from {source}: {provider_name}")
        # C11 fix: 生产环境强制禁止 mock 作为默认 provider
        # 若生产环境显式配置 mock（如演示场景），允许使用但记录警告；
        # 若是默认值（未配置）且环境为生产，则拒绝启动
        is_production = os.getenv("ENVIRONMENT", "development").strip().lower() in {"production", "prod"}
        if is_production and self._current_provider == ModelProvider.MOCK:
            if not env_provider and db_provider == "mock":
                # 未显式配置，使用的是默认值 → 拒绝启动
                raise RuntimeError(
                    "生产环境禁止使用 mock 作为默认模型提供商。"
                    "请通过 MODEL_PROVIDER 环境变量或数据库 config.modelProvider "
                    "显式配置有效的提供商（如 vllm / openai_compat / ollama）。"
                )
            logger.warning(
                "生产环境显式配置为 mock provider，将返回罐头回复，仅适用于演示场景"
            )
        self._load_cache()

    def _load_cache(self):
        self.cache: Dict[str, Any] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"读取模型缓存失败: {e}")
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存模型缓存失败: {e}")

    def set_provider(self, provider: ModelProvider) -> bool:
        if provider not in self._providers:
            logger.error(f"未知提供商: {provider}")
            return False
        self._current_provider = provider
        logger.info(f"已切换到提供商: {provider.value}")
        return True

    def get_current_provider(self) -> BaseProvider:
        return self._providers[self._current_provider]

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        provider = self.get_current_provider()
        return provider.generate(prompt, session_history, rag_docs, max_tokens_override)

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             rag_docs: List[Dict] = None, max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成，使用提供商的异步方法避免阻塞事件循环"""
        provider = self.get_current_provider()
        return await provider.async_generate(prompt, session_history, rag_docs, max_tokens_override)

    def set_lora_adapter(self, lora_path: Optional[str]):
        provider = self.get_current_provider()
        if hasattr(provider, 'set_lora_adapter'):
            provider.set_lora_adapter(lora_path)
        else:
            peft_provider = self._providers.get(ModelProvider.TRANSFORMERS_PEFT)
            if peft_provider and hasattr(peft_provider, 'set_lora_adapter'):
                peft_provider.set_lora_adapter(lora_path)

    def get_status(self) -> Dict[str, Any]:
        providers_status = {}
        for key, provider in self._providers.items():
            providers_status[key.value] = provider.get_status()

        return {
            "currentProvider": self._current_provider.value,
            "providers": providers_status,
        }

    def list_available_models(self) -> List[Dict[str, Any]]:
        models = []
        for model_key, config in MODEL_CONFIGS.items():
            model_dir = self.models_dir / config.name
            downloaded = model_dir.exists()
            if downloaded:
                all_required = all(
                    (model_dir / f).exists() for f in config.required_files
                )
                has_weights = bool(
                    list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                )
                downloaded = all_required and has_weights

            models.append({
                "name": model_key,
                "display_name": config.name,
                "repo_id": config.repo_id,
                "size": config.size,
                "description": config.description,
                "downloaded": downloaded,
            })
        return models

    def check_model_exists(self, model_name: str) -> bool:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            return False

        model_dir = self.models_dir / config.name
        if not model_dir.exists():
            return False

        for required_file in config.required_files:
            if not (model_dir / required_file).exists():
                return False

        weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        return bool(weight_files)

    def shutdown(self):
        """关闭所有 provider 的底层资源（httpx 客户端、GPU 句柄等）。

        C6 fix: 应在应用 lifespan 的 shutdown 阶段调用，避免资源泄漏。
        """
        for provider in self._providers.values():
            try:
                provider.close()
            except Exception as e:
                logger.warning(f"关闭 provider {getattr(provider, 'name', '?')} 失败: {e}")

    def download_model_from_hf(self, model_name: str, force: bool = False) -> Dict[str, Any]:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            return {"success": False, "error": f"未知模型: {model_name}"}

        model_dir = self.models_dir / config.name

        if model_dir.exists() and not force:
            return {
                "success": True,
                "message": f"模型已存在: {config.name}",
                "model_name": model_name,
                "path": str(model_dir),
            }

        try:
            from huggingface_hub import snapshot_download
            logger.info(f"开始下载模型: {config.repo_id}...")
            download_path = snapshot_download(
                repo_id=config.repo_id,
                local_dir=str(model_dir),
                resume_download=True,
            )
            self.cache[model_name] = {
                "downloaded_at": datetime.now().isoformat(),
                "path": str(model_dir),
            }
            self._save_cache()

            return {
                "success": True,
                "message": f"模型下载完成: {config.name}",
                "model_name": model_name,
                "path": str(download_path),
            }
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            return {"success": False, "error": str(e)}

    def delete_model(self, model_name: str) -> bool:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            logger.warning(f"未知模型: {model_name}")
            return False

        model_dir = self.models_dir / config.name
        if not model_dir.exists():
            return True

        try:
            import shutil
            shutil.rmtree(model_dir)
            if model_name in self.cache:
                del self.cache[model_name]
                self._save_cache()
            logger.info(f"模型已删除: {config.name}")
            return True
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
