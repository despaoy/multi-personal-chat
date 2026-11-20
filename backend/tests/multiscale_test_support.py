"""Deterministic query vectors for tests that run without model assets."""

import hashlib

import numpy as np


class FakeQueryEmbeddingProvider:
    dimension = 384
    model_id = "paraphrase-multilingual-MiniLM-L12-v2"
    encoding_contract = "complete-token-windows-weighted-v1"

    def __init__(self) -> None:
        self._model = object()

    def embed_query(self, query: str) -> np.ndarray:
        digest = hashlib.sha256(query.encode("utf-8")).digest()
        raw = np.resize(np.frombuffer(digest, dtype=np.uint8).astype(np.float32), self.dimension)
        norm = np.linalg.norm(raw)
        return raw / norm if norm else raw
