"""Shared embedding providers, explicit model resolution and vector validation.

The default SentenceTransformer provider is reused by long-term memory.
Character retrieval uses its own pooling provider with the same path and
vector validation helpers. Missing required models propagate clear errors.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from infra.environment import read_bool

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIM = 384
ENCODING_CONTRACT = "complete-token-windows-weighted-v1"

_LOCAL_MODEL_SEARCH_PATHS: list[Path] = [
    # backend/RAG、backend/models（与现有 vector_db 约定一致）
    Path(__file__).resolve().parents[2] / "RAG" / DEFAULT_EMBEDDING_MODEL_ID,
    Path(__file__).resolve().parents[2] / "models" / DEFAULT_EMBEDDING_MODEL_ID,
]


class EmbeddingModelError(RuntimeError):
    """embedding 模型不可用/加载失败。"""


def resolve_local_model_path(model_id: str = DEFAULT_EMBEDDING_MODEL_ID) -> str:
    """解析本地 embedding 模型路径。

    顺序：EMBEDDING_MODEL_PATH 环境变量 → HF hub 缓存 snapshot →
    backend 本地目录。仅在 ALLOW_REMOTE_EMBEDDING_MODEL=true 且
    显式允许时返回模型名（联网加载），否则抛错。
    """
    allow_remote = read_bool(os.environ, "ALLOW_REMOTE_EMBEDDING_MODEL")
    env_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if env_path:
        if not Path(env_path).is_dir() or not (Path(env_path) / "config.json").is_file():
            raise EmbeddingModelError("EMBEDDING_MODEL_PATH 无效：请配置包含 config.json 的模型目录")
        return env_path

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    hub_id = model_id if "/" in model_id else f"sentence-transformers/{model_id}"
    candidate_hub = hub_root / ("models--" + hub_id.replace("/", "--"))
    if candidate_hub.exists():
        snapshots = candidate_hub / "snapshots"
        if snapshots.exists():
            for snapshot in sorted(snapshots.iterdir()):
                if (snapshot / "config.json").exists() and (
                    (snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists()
                ):
                    return str(snapshot)

    if model_id in {DEFAULT_EMBEDDING_MODEL_ID, f"sentence-transformers/{DEFAULT_EMBEDDING_MODEL_ID}"}:
        candidates = list(_LOCAL_MODEL_SEARCH_PATHS)
        lab_root = os.getenv("MULTIPERSONAL_LAB_ROOT") or os.getenv("QQCHAT_LAB_ROOT")
        if lab_root:
            candidates.append(Path(lab_root) / "models" / DEFAULT_EMBEDDING_MODEL_ID)
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return str(candidate)

    if allow_remote:
        return model_id

    raise EmbeddingModelError(
        f"本地未找到 embedding 模型 {model_id}：设置 EMBEDDING_MODEL_PATH 或预下载到 "
        f"HF 缓存；仅在外网可用时设置 ALLOW_REMOTE_EMBEDDING_MODEL=true"
    )


def compute_model_fingerprint(model_path: str) -> str:
    """Fingerprint the encoding contract and model configuration for cache invalidation."""
    digest = hashlib.sha256(ENCODING_CONTRACT.encode())
    for name in ("config.json", "modules.json", "sentence_bert_config.json"):
        p = Path(model_path) / name
        if p.exists():
            digest.update(name.encode())
            digest.update(p.read_bytes())
    return digest.hexdigest()[:16]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """embedding 提供方协议（可测试、可注入）。"""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class SentenceTransformerEmbeddingProvider:
    """基于本地 sentence-transformers 的 embedding 实现。

    - 惰性加载，首次调用时初始化
    - 批量编码 + 墙钟超时（超时抛 TimeoutError，不静默截断）
    - 归一化向量（内积 = 余弦相似度）
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        expected_dim: int = DEFAULT_EMBEDDING_DIM,
        batch_size: int = 32,
        timeout_seconds: float = 300.0,
        device: str | None = None,
    ):
        self._model_path = model_path
        self.model_id = model_id
        self._expected_dim = expected_dim
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds
        self._device = device
        self._model = None
        self._fingerprint: str | None = None
        # 模型加载与编码串行化（后台预热与业务检索并发安全）
        self._model_lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._expected_dim

    @property
    def model_path(self) -> str:
        if self._model_path is None:
            self._model_path = resolve_local_model_path(self.model_id)
        return self._model_path

    @property
    def model_fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = compute_model_fingerprint(self.model_path)
        return self._fingerprint

    def _load_model(self):
        with self._model_lock:
            return self._load_model_locked()

    def _load_model_locked(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingModelError(f"sentence-transformers 不可用: {e}") from e

        device = self._device
        if device is None:
            device = "cuda" if _cuda_available() else "cpu"
        self._model = SentenceTransformer(self.model_path, device=device)
        logger.info("embedding 模型已加载: %s (device=%s)", self.model_path, device)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        with self._model_lock:
            return self._embed_texts_locked(texts)

    def _embed_texts_locked(self, texts: list[str]) -> np.ndarray:
        model = self._load_model_locked()
        return sentence_transformer_embeddings(
            model,
            texts,
            batch_size=self._batch_size,
            dimension=self.dimension,
            timeout_seconds=self._timeout_seconds,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


_default_embedding_provider: SentenceTransformerEmbeddingProvider | None = None


def get_default_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """返回进程内共享的惰性 embedding provider，避免业务模块重复加载模型。"""
    global _default_embedding_provider
    if _default_embedding_provider is None:
        _default_embedding_provider = SentenceTransformerEmbeddingProvider()
    return _default_embedding_provider


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def vector_norm(vector: np.ndarray) -> float:
    """Validate an embedding without changing its inner-product magnitude."""
    if not np.isfinite(vector).all():
        raise ValueError("embedding vector contains nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm):
        raise ValueError("embedding vector norm is nonfinite")
    if norm <= 0.0:
        raise ValueError("embedding vector has zero norm")
    return norm


def normalized_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    return value / vector_norm(value)


def complete_embeddings(texts, tokenizer, *, max_tokens, batch_size, dimension, encode_batch, timeout_seconds=300):
    if any(type(value) is not int or value < 1 for value in (max_tokens, batch_size, dimension)):
        raise ValueError("Embedding window, batch and dimension must be positive integers")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Embedding deadline must be finite and positive")
    capacity = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 1:
        raise ValueError("Embedding window has no room for payload tokens")
    if not texts:
        return np.zeros((0, dimension), dtype=np.float32)
    deadline = time.monotonic() + timeout_seconds
    accumulated = np.zeros((len(texts), dimension), dtype=np.float64)
    for source_start in range(0, len(texts), batch_size):
        source_texts = texts[source_start : source_start + batch_size]
        tokens = tokenizer(source_texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        if len(tokens) != len(source_texts):
            raise ValueError("Tokenizer did not preserve the number of source texts")
        encoded = tokenizer(
            source_texts,
            truncation=True,
            max_length=max_tokens,
            stride=0,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
            padding=True,
            return_tensors="pt",
        )
        owners = encoded.pop("overflow_to_sample_mapping").tolist()
        specials = encoded.pop("special_tokens_mask").tolist()
        recovered = [[] for _ in source_texts]
        weights = []
        for owner, row, attention, special in zip(
            owners, encoded["input_ids"].tolist(), encoded["attention_mask"].tolist(), specials, strict=True
        ):
            payload = [
                token for token, active, marker in zip(row, attention, special, strict=True) if active and not marker
            ]
            recovered[owner].extend(payload)
            weights.append(max(1, len(payload)))
        if recovered != tokens:
            raise ValueError("Tokenizer did not preserve complete source tokens")
        for start in range(0, len(owners), batch_size):
            if time.monotonic() > deadline:
                raise TimeoutError("Complete embedding encoding exceeded its deadline")
            stop = start + batch_size
            features = {name: values[start:stop] for name, values in encoded.items()}
            rows = np.asarray(encode_batch(features), dtype=np.float32)
            if rows.shape != (len(owners[start:stop]), dimension):
                raise ValueError("embedding shape mismatch for complete token windows")
            for owner, weight, row in zip(owners[start:stop], weights[start:stop], rows, strict=True):
                accumulated[source_start + owner] += weight * row / vector_norm(row)
    if time.monotonic() > deadline:
        raise TimeoutError("Complete embedding encoding exceeded its deadline")
    return np.asarray([row / vector_norm(row) for row in accumulated], dtype=np.float32)


def sentence_transformer_embeddings(model, texts, *, batch_size, dimension, timeout_seconds=300):
    import torch

    def encode(features):
        features = {name: value.to(model.device) for name, value in features.items()}
        features["modality"] = "text"
        return model(features)["sentence_embedding"].detach().float().cpu().numpy()

    model.eval()
    with torch.inference_mode():
        return complete_embeddings(
            texts,
            model.tokenizer,
            max_tokens=model.max_seq_length,
            batch_size=batch_size,
            dimension=dimension,
            encode_batch=encode,
            timeout_seconds=timeout_seconds,
        )
