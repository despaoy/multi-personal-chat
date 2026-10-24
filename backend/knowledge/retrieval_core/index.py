"""In-memory vector, sparse and entity indexes shared by retrieval.

Production artifact loading belongs to ``multiscale_rag.runtime``; this module
keeps the runtime query structures and deterministic BM25 construction.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from .embedding import vector_norm
from .errors import RetrievalIndexError
from .tokenization import tokenize

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .documents import KnowledgeIndexDocument


class BM25Index:
    """域内 BM25 索引（确定性重建，无持久化状态）。

    文档表示：title/entities/keywords 重复加权拼接 + summary + content，
    保证精确实体与关键词召回不被长 evidence 稀释。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, field_weight: int = 3):
        self.k1 = k1
        self.b = b
        self.field_weight = max(1, field_weight)
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._built = False

    @staticmethod
    def document_text(doc: KnowledgeIndexDocument, field_weight: int = 3) -> str:
        precise = " ".join(
            [
                doc.title,
                " ".join(doc.entities),
                " ".join(doc.keywords),
                " ".join(doc.relations),
            ]
        )
        return "\n".join([precise] * field_weight + [doc.summary, doc.content])

    def build(self, documents: Sequence[KnowledgeIndexDocument]) -> None:
        tokenized = [tokenize(self.document_text(doc, self.field_weight)) for doc in documents]
        self._doc_tokens = tokenized
        self._doc_lens = []
        self._doc_freqs = defaultdict(int)
        for tokens in tokenized:
            self._doc_lens.append(len(tokens))
            for token in set(tokens):
                self._doc_freqs[token] += 1
        n_docs = len(documents)
        self._avgdl = (sum(self._doc_lens) / n_docs) if n_docs else 0.0
        self._idf = {
            token: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0) for token, freq in self._doc_freqs.items()
        }
        self._built = True

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 (文档行号, BM25 分数) 列表，按分数降序。"""
        if top_k <= 0:
            return []
        if not self._built:
            raise RetrievalIndexError("sparse", "not_initialized")
        if not self._doc_tokens:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores: list[tuple[int, float]] = []
        for doc_idx, tokens in enumerate(self._doc_tokens):
            counts = Counter(tokens)
            doc_len = self._doc_lens[doc_idx] or 1
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(token, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1.0))
                score += idf * numerator / denominator
            if score > 0:
                scores.append((doc_idx, score))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]

    def stats(self) -> dict[str, Any]:
        return {
            "docs": len(self._doc_tokens),
            "vocab": len(self._doc_freqs),
            "avgdl": round(self._avgdl, 1),
            "built": self._built,
        }


class DomainIndex:
    """Domain-local vector, BM25 and entity lookup state."""

    def __init__(self, domain_id: str, dimension: int = 384):
        self.domain_id = domain_id
        self.dimension = dimension
        self.documents: list[KnowledgeIndexDocument] = []
        self._faiss_index = None
        self._entity_rows: dict[str, list[int]] = {}
        self.bm25 = BM25Index()
        self.manifest: dict[str, Any] = {}

    def _rebuild_entity_index(self) -> None:
        self._entity_rows = defaultdict(list)
        for row, doc in enumerate(self.documents):
            for entity in set(doc.entities):
                self._entity_rows[entity].append(row)

    # -- 检索原语 ---------------------------------------------------------
    def search_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """向量近邻检索：返回 (行号, 相似度)，按相似度降序。"""
        if top_k <= 0:
            return []
        self.require_ready("vector")
        if not self.documents:
            return []
        vector = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if vector.shape[1] != self._faiss_index.d:
            raise ValueError(f"Query embedding dimension mismatch: {vector.shape[1]} != {self._faiss_index.d}")
        vector_norm(vector)
        k = min(top_k, self._faiss_index.ntotal)
        scores, rows = self._faiss_index.search(vector, k)
        return [(int(row), float(score)) for row, score in zip(rows[0], scores[0]) if row >= 0 and score >= threshold]

    def search_sparse(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        self.require_ready("sparse")
        return self.bm25.search(query, top_k)

    def entity_lookup(self, entities: Iterable[str]) -> list[int]:
        """实体精确索引：返回包含任一实体的文档行号（去重升序）。"""
        rows: set = set()
        for entity in entities:
            rows.update(self._entity_rows.get(entity, []))
        return sorted(rows)

    def get_document(self, row: int) -> KnowledgeIndexDocument:
        if not 0 <= row < len(self.documents):
            raise RetrievalIndexError("documents", "invalid_row")
        return self.documents[row]

    # -- 状态 -------------------------------------------------------------
    def count(self) -> int:
        return len(self.documents)

    def require_ready(self, mode: str = "hybrid") -> None:
        """Only requested channels must be initialized and match document rows."""
        if mode not in {"sparse", "vector", "hybrid"}:
            raise ValueError("Unknown retrieval mode; choose sparse, vector or hybrid")
        if mode in {"sparse", "hybrid"}:
            if not self.bm25._built:
                raise RetrievalIndexError("sparse", "not_initialized")
            if len(self.bm25._doc_tokens) != len(self.documents):
                raise RetrievalIndexError("sparse", "row_count_mismatch")
        if mode in {"vector", "hybrid"}:
            if self._faiss_index is None:
                raise RetrievalIndexError("vector", "not_initialized")
            if self._faiss_index.ntotal != len(self.documents):
                raise RetrievalIndexError("vector", "row_count_mismatch")

    def is_healthy(self) -> bool:
        """一致性检查：manifest 与 documents/faiss 行数闭合。"""
        declared = self.manifest.get("document_count")
        if declared is None or declared != len(self.documents):
            return False
        try:
            self.require_ready()
        except RetrievalIndexError:
            return False
        return True

    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = defaultdict(int)
        for doc in self.documents:
            type_counts[doc.document_type] += 1
        return {
            "domain_id": self.domain_id,
            "document_count": len(self.documents),
            "document_type_counts": dict(type_counts),
            "vector_dimension": self.manifest.get("vector_dimension"),
            "index_version": self.manifest.get("index_version"),
            "bm25": self.bm25.stats(),
            "healthy": self.is_healthy(),
        }
