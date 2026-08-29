"""Local vector provider for character knowledge retrieval.

It reproduces the cached SentenceTransformer model's Transformer + mean-pooling
pipeline with Hugging Face Transformers. No model is downloaded at runtime.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from knowledge.retrieval_core.embedding import resolve_local_model_path

if TYPE_CHECKING:
    from collections.abc import Sequence


class LocalMeanPoolingEmbeddingProvider:
    model_id = "paraphrase-multilingual-MiniLM-L12-v2"
    dimension = 384

    def __init__(self, model_path: str | None = None, *, batch_size: int = 24, max_length: int = 128) -> None:
        self.model_path = model_path or resolve_local_model_path(self.model_id)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self._tokenizer = None
        self._model = None
        self._query_cache: dict[str, np.ndarray] = {}
        # Singleton warmup and the first request may arrive together. Protect
        # Transformers lazy imports and CPU model inference from that race.
        self._runtime_lock = threading.RLock()

    def _load(self):
        with self._runtime_lock:
            if self._model is not None:
                return self._tokenizer, self._model
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
            self._model.eval()
            self._model.to("cpu")
            torch.set_grad_enabled(False)
            return self._tokenizer, self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        with self._runtime_lock:
            import torch

            tokenizer, model = self._load()
            batches: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    encoded = tokenizer(
                        texts[start : start + self.batch_size],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    hidden = model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        matrix = np.vstack(batches)
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"embedding 维度异常: {matrix.shape}")
        return matrix

    def embed_query(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is None:
            cached = self.embed_texts([query])[0]
            self._query_cache[query] = cached
        return cached.copy()


class MatrixInnerProductIndex:
    """Small FAISS-compatible search surface over normalized NumPy vectors."""

    def __init__(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors 必须为二维矩阵")
        self.matrix = np.ascontiguousarray(matrix)
        self.ntotal = int(matrix.shape[0])
        self.d = int(matrix.shape[1]) if self.ntotal else 0

    def search(self, query: np.ndarray, top_k: int):
        query_matrix = np.asarray(query, dtype=np.float32).reshape(-1, self.d)
        scores = query_matrix @ self.matrix.T
        k = min(max(0, int(top_k)), self.ntotal)
        if k == 0:
            return np.zeros((len(query_matrix), 0), dtype=np.float32), np.zeros((len(query_matrix), 0), dtype=np.int64)
        rows = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        ranked_scores = np.take_along_axis(scores, rows, axis=1)
        return ranked_scores, rows


def attach_vectors(index, vectors: np.ndarray) -> None:
    """Attach real vectors to an in-memory DomainIndex without FAISS files."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if len(index.documents) != len(matrix):
        raise ValueError(f"文档与向量数量不一致: {len(index.documents)} != {len(matrix)}")
    index._faiss_index = MatrixInnerProductIndex(matrix)
    index.dimension = int(matrix.shape[1])
    index.manifest.update(
        {
            "document_count": len(index.documents),
            "vector_dimension": int(matrix.shape[1]),
            "distance_metric": "cosine (normalized inner product)",
            "embedding_model_id": LocalMeanPoolingEmbeddingProvider.model_id,
        }
    )


def write_vector_artifacts(root: Path, documents: Sequence, vectors: np.ndarray, manifest: dict) -> None:
    """Write an isolated, inspectable document/vector bundle."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(document.to_jsonl() + "\n")
    np.save(root / "vectors.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
