"""Embedding Provider 抽象与缓存（P6）。

- EmbeddingProvider 协议：单条/批量向量化、模型 ID、维度、超时、
  错误传播。领域代码不读环境密钥、不初始化云端客户端、不写死
  localhost、不偷偷下载模型——路径解析与开关收敛在 Provider 层。
- EmbeddingCache：缓存键绑定 domain_id + document_id +
  embedding_text 指纹 + 模型 ID + 模型指纹，不同模型向量不混用。

默认实现复用项目既有的本地 sentence-transformers 模型
（paraphrase-multilingual-MiniLM-L12-v2，384 维），优先从本地
缓存解析路径；未找到本地模型且未显式允许联网时抛错（不静默下载）。
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIM = 384

_HUB_CACHE_NAME = "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"

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
    if env_path and Path(env_path).exists():
        return env_path

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    candidate_hub = hub_root / f"models--sentence-transformers--{model_id.replace('/', '--')}"
    if not candidate_hub.exists():
        candidate_hub = hub_root / _HUB_CACHE_NAME
    if candidate_hub.exists():
        snapshots = candidate_hub / "snapshots"
        if snapshots.exists():
            for snapshot in sorted(snapshots.iterdir()):
                if (snapshot / "config.json").exists() and (
                    (snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists()
                ):
                    return str(snapshot)

    for candidate in _LOCAL_MODEL_SEARCH_PATHS:
        if candidate.exists() and (candidate / "config.json").exists():
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
        max_retries: int = 1,
        device: str | None = None,
    ):
        self._model_path = model_path
        self.model_id = model_id
        self._expected_dim = expected_dim
        self._batch_size = max(1, int(batch_size))
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
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
        try:
            self._model = SentenceTransformer(self.model_path, device=device)
        except Exception as e:
            if device != "cpu" and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
                logger.warning("GPU 加载失败，回退 CPU: %s", e)
                device = "cpu"
                self._model = SentenceTransformer(self.model_path, device="cpu")
            else:
                raise EmbeddingModelError(f"embedding 模型加载失败: {e}") from e
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
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
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
            except Exception as e:  # noqa: BLE001 - 重试后仍失败则传播
                last_error = e
                all_embeddings = []
                if attempt >= self._max_retries:
                    break
                logger.warning("embedding 批量编码失败，重试 %d/%d: %s", attempt + 1, self._max_retries, e)
                time.sleep(0.5 * (attempt + 1))
        raise last_error if last_error else EmbeddingModelError("embedding 编码失败")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch 缺失时按 CPU 处理
        return False


# ---------------------------------------------------------------------------
# embedding 缓存
# ---------------------------------------------------------------------------


def embedding_cache_key(
    domain_id: str,
    document_id: str,
    embedding_text_fingerprint: str,
    model_id: str,
    model_fingerprint: str,
) -> str:
    """缓存键绑定 domain + 文档 ID + 文本指纹 + 模型 ID + 模型指纹。"""
    raw = f"{domain_id}|{document_id}|{embedding_text_fingerprint}|{model_id}|{model_fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class EmbeddingCache:
    """npz 持久化的向量缓存（全量加载 + 原子写回）。

    适用于当前知识域规模（数百至数万文档）；超大规模域可替换为
    键值后端实现同接口。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._vectors: dict[str, np.ndarray] = {}
        self._dirty = False
        self._loaded = False

    def load(self) -> int:
        self._vectors = {}
        self._loaded = True
        if not self.path.exists():
            return 0
        try:
            with np.load(self.path, allow_pickle=False) as data:
                for key in data.files:
                    self._vectors[str(key)] = np.asarray(data[key], dtype=np.float32)
        except Exception as e:  # pragma: no cover - 损坏缓存视为空
            logger.warning("embedding 缓存加载失败（视为空缓存）: %s", e)
            self._vectors = {}
        return len(self._vectors)

    def get(self, key: str) -> np.ndarray | None:
        if not self._loaded:
            self.load()
        vector = self._vectors.get(key)
        if vector is None:
            return None
        return vector.astype(np.float32, copy=True)

    def put(self, key: str, vector: np.ndarray) -> None:
        if not self._loaded:
            self.load()
        self._vectors[key] = np.asarray(vector, dtype=np.float32)
        self._dirty = True

    def drop(self, keys: list[str]) -> int:
        if not self._loaded:
            self.load()
        dropped = sum(1 for k in keys if k in self._vectors)
        for key in keys:
            self._vectors.pop(key, None)
        if dropped:
            self._dirty = True
        return dropped

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # npz 临时文件必须以 .npz 结尾，否则 numpy 会再追加后缀
        tmp_path = self.path.with_name(self.path.stem + ".tmp.npz")
        if self._vectors:
            np.savez(tmp_path, **self._vectors)
        else:
            # 空缓存也落盘占位，避免残留旧文件被误加载
            np.savez(tmp_path, __empty__=np.zeros(0, dtype=np.float32))
        os.replace(str(tmp_path), str(self.path))
        self._dirty = False

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._vectors)

    def prune_to(self, valid_keys: set) -> int:
        """删除不在 valid_keys 内的缓存项，返回删除数。"""
        if not self._loaded:
            self.load()
        stale = [k for k in self._vectors if k not in valid_keys]
        return self.drop(stale)
