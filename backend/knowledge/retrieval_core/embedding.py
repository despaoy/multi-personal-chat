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

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIM = 384

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

    if os.getenv("ALLOW_REMOTE_EMBEDDING_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}:
        return model_id

    raise EmbeddingModelError(
        f"本地未找到 embedding 模型 {model_id}：设置 EMBEDDING_MODEL_PATH 或预下载到 "
        f"HF 缓存；仅在外网可用时设置 ALLOW_REMOTE_EMBEDDING_MODEL=true"
    )


def compute_model_fingerprint(model_path: str) -> str:
    """模型指纹：config.json + modules.json 内容哈希（缓存失效依据）。"""
    digest = hashlib.sha256()
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
        self._batch_size = max(1, int(batch_size))
        self._timeout_seconds = float(timeout_seconds)
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
        start = time.monotonic()
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=len(batch),
                show_progress_bar=False,
            )
            all_embeddings.append(np.asarray(embeddings, dtype=np.float32))
            if time.monotonic() - start > self._timeout_seconds:
                raise TimeoutError(
                    f"embedding 编码超时（>{self._timeout_seconds}s，已完成 {i + len(batch)}/{len(texts)}）"
                )
        matrix = np.vstack(all_embeddings)
        if matrix.shape[1] != self.dimension:
            raise EmbeddingModelError(f"embedding 维度不符: 期望 {self.dimension}, 实际 {matrix.shape[1]}")
        return matrix.astype(np.float32, copy=False)


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
