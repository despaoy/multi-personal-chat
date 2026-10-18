#!/usr/bin/env python3
"""
模型管理模块
负责多提供商模型推理、本地模型文件管理、LoRA适配器切换。
统一管理已配置的远程、本地模型提供商及资源生命周期。
"""

import logging
import os
import random
import threading
import time
from enum import Enum
from pathlib import Path

from inference.openai_protocol import (
    ProviderResponseError,
    chat_completions_endpoint,
    completed_chat_content,
    nonthinking_parameters,
    validate_completion_options,
)

from services.model_management import ModelDownloadError, UnknownModelError

logger = logging.getLogger(__name__)

def _get_db_config():
    from cache.config_cache import read_config
    from db.adapter import db

    return read_config(db)[0]

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS_PEFT = "transformers_peft"
    OPENAI_COMPAT = "openai_compat"
    VLLM = "vllm"
    MOCK = "mock"


def validate_provider(value) -> ModelProvider:
    try:
        provider = ModelProvider(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("模型提供商缺失或无效，请配置 MODEL_PROVIDER 或 config.modelProvider") from exc
    if provider == ModelProvider.MOCK:
        environment = os.getenv("ENVIRONMENT", "development").strip().lower()
        if environment not in {"development", "dev", "test", "testing"}:
            raise ValueError("mock 仅允许在 development/test 环境显式启用")
        logger.warning("Explicit mock provider: real model inference is unavailable")
    return provider


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

    @abstractmethod
    def generate(self, prompt: str, session_history: List[Dict] = None,
                 max_tokens_override: int = None) -> Tuple[str, float]:
        """子类必须实现同步生成方法"""
        raise NotImplementedError

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成，默认通过线程池运行同步方法，子类可覆写以提供原生异步实现"""
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, session_history, max_tokens_override)

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "modelName": self._model_name,
        }

    @asynccontextmanager
    async def _acquire_http_client(self, timeout: float = 120.0):
        """Borrow the shared pool client, or own a client for a standalone call."""
        from app.config import http_client_pool

        pool = http_client_pool
        if pool is not None:
            # P1-M4 fix: pool 的 client 以固定 request_timeout 创建，此处无法
            # 逐请求覆盖。调用方应在 client.post(...) 时显式传入 timeout 参数
            # 以使 vllmTimeout 等配置生效（httpx 支持请求级 timeout 覆盖）。
            async with pool.acquire() as client:
                yield client
            return

        # A standalone call owns its client and closes it on the creating loop.
        import httpx
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=20, max_keepalive_connections=10
            ),
        ) as client:
            yield client

    def close(self):
        """Providers with owned persistent resources override this hook."""
        return None


class OpenAICompatProvider(BaseProvider):
    """OpenAI兼容API提供商（支持DeepSeek、通义千问等）"""

    def __init__(self, config_snapshot: dict[str, Any] | None = None):
        super().__init__("openai_compat")
        self._snapshot_config = dict(config_snapshot) if config_snapshot is not None else None
        self._refresh_db_config()
        self._loaded = True

    def _refresh_db_config(self):
        """Resolve stored settings, then environment defaults, from one snapshot."""
        config = self._snapshot_config if self._snapshot_config is not None else _get_db_config()
        self.base_url = config.get("openaiCompatBaseUrl") or os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
        self.api_key = config.get("openaiCompatApiKey") or os.getenv("OPENAI_COMPAT_API_KEY", "")
        self.model = config.get("openaiCompatModel") or os.getenv("OPENAI_COMPAT_MODEL", "deepseek-chat")
        self._model_name = self.model
        return config

    def snapshot(self):
        """Own one resolved connection for a request, without refreshing mid-turn."""
        if self._snapshot_config is not None:
            return self
        return OpenAICompatProvider(_get_db_config())

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                              max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成实现（R-1 fix: 使用共享 HttpClientPool）"""
        config = self._refresh_db_config()

        messages = []

        if session_history:
            messages.extend(session_history)

        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens_override if max_tokens_override else int(config.get('maxTokens', 512))

        return await self._complete_messages(
            messages, temperature=float(config.get('temperature', 0.8)), max_tokens=max_tokens)

    async def async_complete(
        self, messages, *, temperature, max_tokens, top_p=None,
        frequency_penalty=0.0, repetition_penalty=1.0, enable_thinking=False, lora_name=None,
    ):
        """Send one compiled request with its explicit generation settings."""
        if self._snapshot_config is None:
            self._refresh_db_config()
        return await self._complete_messages(
            messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            frequency_penalty=frequency_penalty, repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking, lora_name=lora_name,
        )

    async def _complete_messages(
        self, messages, *, temperature, max_tokens, top_p=None,
        frequency_penalty=0.0, repetition_penalty=1.0, enable_thinking=False, lora_name=None,
    ):
        base_url, api_key, model = self.base_url, self.api_key, self.model
        validate_completion_options(
            base_url, top_p=top_p, frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty, enable_thinking=enable_thinking, lora_name=lora_name,
        )
        if not api_key:
            raise RuntimeError("未配置 API Key，请在设置页面配置 OpenAI 兼容 API Key")
        start = time.time()
        async with self._acquire_http_client(timeout=120.0) as client:
            # P1-M4 fix: 请求级 timeout 覆盖 pool 默认值，确保生效
            response = await client.post(
                chat_completions_endpoint(base_url),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **({"top_p": top_p} if top_p is not None else {}),
                    **({"frequency_penalty": frequency_penalty} if frequency_penalty else {}),
                    **nonthinking_parameters(base_url, local_template=False),
                },
                timeout=120.0,
            )
        if response.status_code == 200:
            data = response.json()
            reply = completed_chat_content(data)
            cost = round(time.time() - start, 2)
            return reply, cost
        else:
            raise ProviderResponseError(response.status_code)

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 max_tokens_override: int = None) -> Tuple[str, float]:
        # R-1 fix: 同步入口通过 asyncio.run 执行异步实现
        import asyncio
        return asyncio.run(self.async_generate(prompt, session_history, max_tokens_override))

class MockProvider(BaseProvider):
    """模拟提供商，用于测试"""

    def __init__(self):
        super().__init__("mock")
        self._model_name = "Mock Model"
        self._loaded = True

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 max_tokens_override: int = None) -> Tuple[str, float]:
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

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                              max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成实现（R-1 fix: 使用共享 HttpClientPool）"""
        config = _get_db_config()
        start = time.time()
        messages = []

        if session_history:
            messages.extend(session_history)

        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens_override if max_tokens_override else int(config.get('maxTokens', 512))

        async with self._acquire_http_client(timeout=120.0) as client:
            # P1-M4 fix: 请求级 timeout 覆盖 pool 默认值
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": float(config.get('temperature', 0.8)), "top_p": 0.9, "num_predict": max_tokens}
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

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 max_tokens_override: int = None) -> Tuple[str, float]:
        # R-1 fix: 同步入口通过 asyncio.run 执行异步实现
        import asyncio
        return asyncio.run(self.async_generate(prompt, session_history, max_tokens_override))

class LlamaCppProvider(BaseProvider):
    """llama.cpp 提供商"""

    def __init__(self):
        super().__init__("llama_cpp")
        self.base_url = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")
        self._model_name = "llama.cpp"
        self._loaded = True
        # R-1 fix: 不再自建 httpx.Client，统一通过 _acquire_http_client() 获取共享 AsyncClient

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                              max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成实现（R-1 fix: 使用共享 HttpClientPool）"""
        config = _get_db_config()
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

        max_tokens = max_tokens_override if max_tokens_override else int(config.get('maxTokens', 512))

        async with self._acquire_http_client(timeout=120.0) as client:
            # P1-M4 fix: 请求级 timeout 覆盖 pool 默认值
            response = await client.post(
                f"{self.base_url}/completion",
                json={
                    "prompt": full_prompt,
                    "n_predict": max_tokens,
                    "temperature": float(config.get('temperature', 0.8)),
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

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 max_tokens_override: int = None) -> Tuple[str, float]:
        # R-1 fix: 同步入口通过 asyncio.run 执行异步实现
        import asyncio
        return asyncio.run(self.async_generate(prompt, session_history, max_tokens_override))

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
        """Publish a complete NF4 model only after initialization succeeds."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            from peft import PeftModel
            from transformers import AutoTokenizer

            base_path = str(Path(self._base_model_path).resolve())
            if not Path(base_path).is_dir():
                raise FileNotFoundError(f"模型目录不存在: {base_path}")
            if self._lora_path is not None and not Path(self._lora_path).is_dir():
                raise FileNotFoundError(f"LoRA适配器目录不存在: {self._lora_path}")

            tokenizer = AutoTokenizer.from_pretrained(base_path)
            model = self._load_4bit(base_path)
            if self._lora_path is not None:
                model = PeftModel.from_pretrained(model, self._lora_path)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True
            logger.info("Transformers PEFT 模型加载完成")

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

    def set_lora_adapter(self, lora_path: Optional[str]):
        with self._load_lock:
            if lora_path is not None:
                if not lora_path or not Path(lora_path).is_dir():
                    raise FileNotFoundError(f"LoRA适配器目录不存在: {lora_path}")
                lora_path = str(Path(lora_path).resolve())
            if lora_path == self._lora_path:
                return
            # PEFT may change base weights; a new selection requires a clean load.
            self._model = None
            self._tokenizer = None
            self._loaded = False
            self._lora_path = lora_path

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 max_tokens_override: int = None) -> Tuple[str, float]:
        config = _get_db_config()
        import torch

        self._ensure_loaded()
        start = time.time()

        messages = []

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

        max_tokens = max_tokens_override if max_tokens_override else int(config.get('maxTokens', 512))

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=float(config.get('temperature', 0.85)),
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
        with self._load_lock:
            had_model = self._model is not None
            self._model = None
            self._tokenizer = None
            self._loaded = False
            if had_model:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("TransformersPeftProvider 模型资源已释放")


class VLLMProvider(BaseProvider):
    """vLLM 提供商 - 高性能推理引擎，支持 Continuous Batching"""

    def __init__(self):
        super().__init__("vllm")
        # base_url / model / timeout 由 VLLMClient 单例从环境变量统一管理，
        # VLLMProvider 不再持有独立副本，避免设置页修改后状态显示与实际生效不一致。
        # 这些字段保留为属性是为了向后兼容（部分代码可能读取 provider.base_url），
        # 但值反映 VLLMClient 的实际配置，不接受 db 覆盖。
        self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        # D-4 fix: 统一使用 get_vllm_served_model_name() 解析模型名
        from app.config import get_vllm_served_model_name
        self.model = get_vllm_served_model_name()
        self.timeout = float(os.getenv("VLLM_TIMEOUT", "120.0"))
        self._model_name = self.model
        self._loaded = True
        # R-1 fix: 不再自建 httpx.AsyncClient，统一通过 _acquire_http_client() 获取共享 AsyncClient
        self._lora_adapter: Optional[str] = None

    def set_lora_adapter(self, lora_path: Optional[str]):
        """通过 vLLM 的 --enable-lora 传递 LoRA 名称"""
        if lora_path:
            self._lora_adapter = lora_path
            logger.info(f"vLLM LoRA 适配器已设置: {lora_path}")
        else:
            self._lora_adapter = None
            logger.info("vLLM LoRA 适配器已清除")

    async def generate_async(self, prompt: str, session_history: List[Dict] = None,
                             max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成 - 复用 VLLMClient 全局单例（负载均衡 + 熔断器 + 健康检查）。

        Critical fix: 此前 VLLMProvider 通过 HttpClientPool 直接 POST vLLM，
        绕过了 VLLMClient 的负载均衡、熔断器、健康检查，导致：
        1. 同一 vLLM 后端有 3 条独立 HTTP 客户端路径，连接数不可控
        2. VLLMProvider 路径的失败不会触发熔断器
        3. VLLMClient 的多实例负载均衡对 VLLMProvider 无效
        现统一委托给 VLLMClient.generate，消除连接池冗余和状态隔离。
        """
        config = _get_db_config()
        start = time.time()

        messages = []

        if session_history:
            for msg in session_history[-10:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens_override if max_tokens_override else int(config.get('maxTokens', 512))
        temperature = float(config.get('temperature', 0.8))

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
                 max_tokens_override: int = None) -> Tuple[str, float]:
        """同步生成包装 - 通过 asyncio.run 执行异步方法"""
        import asyncio
        return asyncio.run(self.generate_async(prompt, session_history, max_tokens_override))

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成入口"""
        return await self.generate_async(prompt, session_history, max_tokens_override)

class ProviderShutdownError(RuntimeError):
    """All provider close failures, retained across asynchronous/thread boundaries."""

    def __init__(self, failures):
        self.failures = tuple(failures)
        details = ", ".join(f"{name}: {type(error).__name__}" for name, error in self.failures)
        super().__init__(f"Model provider cleanup failed ({details})")


class ModelManager:
    """模型管理器，负责多提供商切换、LoRA管理、模型文件操作。"""

    def __init__(self, base_dir: Optional[Path] = None):
        env_provider = os.getenv("MODEL_PROVIDER", "").strip().lower()
        provider_name = env_provider or _get_db_config().get("modelProvider")
        self._current_provider = validate_provider(provider_name)
        logger.info("Model provider selected from %s: %s",
                    "environment" if env_provider else "database", self._current_provider.value)
        backend_dir = Path(__file__).resolve().parents[1]
        configured_root = os.getenv("MODEL_STORAGE_PATH", "").strip()
        if configured_root and base_dir is None:
            model_root = Path(configured_root).expanduser()
            self.models_dir = model_root if model_root.is_absolute() else backend_dir / model_root
            self.base_dir = self.models_dir.parent
        else:
            self.base_dir = base_dir or backend_dir
            self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model_files_lock = threading.RLock()

        self._providers: Dict[ModelProvider, BaseProvider] = {
            ModelProvider.MOCK: MockProvider(),
            ModelProvider.OLLAMA: OllamaProvider(),
            ModelProvider.LLAMA_CPP: LlamaCppProvider(),
            ModelProvider.OPENAI_COMPAT: OpenAICompatProvider(),
            ModelProvider.TRANSFORMERS_PEFT: TransformersPeftProvider(),
            ModelProvider.VLLM: VLLMProvider(),
        }

    def set_provider(self, provider: ModelProvider) -> bool:
        provider = validate_provider(provider)
        self._current_provider = provider
        logger.info(f"已切换到提供商: {provider.value}")
        return True

    def get_current_provider(self) -> BaseProvider:
        return self._providers[self._current_provider]

    def generate(self, prompt: str, session_history: List[Dict] = None,
                 max_tokens_override: int = None) -> Tuple[str, float]:
        provider = self.get_current_provider()
        return provider.generate(prompt, session_history, max_tokens_override)

    async def async_generate(self, prompt: str, session_history: List[Dict] = None,
                             max_tokens_override: int = None) -> Tuple[str, float]:
        """异步生成，使用提供商的异步方法避免阻塞事件循环"""
        provider = self.get_current_provider()
        return await provider.async_generate(prompt, session_history, max_tokens_override)

    def set_lora_adapter(self, lora_path: Optional[str]):
        provider = self.get_current_provider()
        setter = getattr(provider, 'set_lora_adapter', None)
        if setter is None:
            if lora_path is not None:
                raise ValueError("Selected model provider does not support LoRA")
            return
        setter(lora_path)

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

            models.append({
                "name": model_key,
                "display_name": config.name,
                "repo_id": config.repo_id,
                "size": config.size,
                "description": config.description,
                "downloaded": self.check_model_exists(model_key),
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
        """Attempt all provider closes; retain every failure and cancellation."""
        failures = []
        for provider in self._providers.values():
            try:
                provider.close()
            except BaseException as error:
                failures.append((provider.name, error))
        if failures:
            error = ProviderShutdownError(failures)
            for _, failure in failures:
                if not isinstance(failure, Exception):
                    raise failure from error
            raise error

    def download_model_from_hf(self, model_name: str, force: bool = False) -> Dict[str, Any]:
        """Download one model while serializing destructive model-directory changes."""

        with self._model_files_lock:
            return self._download_model_from_hf_locked(model_name, force=force)

    def _download_model_from_hf_locked(self, model_name: str, force: bool = False) -> Dict[str, Any]:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            raise UnknownModelError("模型名称未登记")

        model_dir = self.models_dir / config.name

        if self.check_model_exists(model_name) and not force:
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
            if not self.check_model_exists(model_name):
                raise RuntimeError("模型下载后缺少必需文件或权重，未完成下载")

            return {
                "success": True,
                "message": f"模型下载完成: {config.name}",
                "model_name": model_name,
                "path": str(download_path),
            }
        except Exception as exc:
            raise ModelDownloadError(exc) from exc

    def delete_model(self, model_name: str) -> bool:
        """Delete one model without racing an in-process download."""

        with self._model_files_lock:
            return self._delete_model_locked(model_name)

    def _delete_model_locked(self, model_name: str) -> bool:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            logger.warning(f"未知模型: {model_name}")
            return False

        model_dir = self.models_dir / config.name
        try:
            model_dir.lstat()
        except FileNotFoundError:
            return True

        import shutil
        shutil.rmtree(model_dir)
        logger.info("模型已删除: %s", config.name)
        return True


_model_manager: Optional[ModelManager] = None
_model_manager_init_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        with _model_manager_init_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager
