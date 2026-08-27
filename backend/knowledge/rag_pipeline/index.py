"""持久化知识索引（P6）。

单域索引产物（index_root 目录）：
- documents.jsonl      canonical 文档（行序 = 向量行序，权威数据源）
- faiss.bin            FAISS IndexFlatIP（精确内积，行对齐 documents）
- index_manifest.json  构建清单（commit 标记，最后原子替换）
- embedding_cache.npz  embedding 缓存（增量构建复用）

原子切换策略：documents/faiss 先落临时文件并逐个 os.replace，
manifest 最后替换——崩溃中断时 manifest 的 count 校验失败，索引
判定为不可用（health check），服务清晰降级，重建即恢复。

BM25 稀疏索引从 documents.jsonl 确定性重建（加载时构建，不持久
化词频状态），避免 pickle 状态与文档错位。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .documents import KnowledgeIndexDocument

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .embedding import EmbeddingProvider

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

DOCUMENTS_FILE = "documents.jsonl"
FAISS_FILE = "faiss.bin"
MANIFEST_FILE = "index_manifest.json"
CACHE_FILE = "embedding_cache.npz"


def _tokenize(text: str) -> list[str]:
    """中文 jieba 分词 + 英文/数字词切分（与现有 BM25 约定一致）。"""
    tokens = re.findall(r"\w+", text.lower())
    try:
        import jieba

        cn_tokens = list(jieba.cut(text))
        tokens.extend(t for t in cn_tokens if t.strip() and not t.isascii())
    except ImportError:
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))
    return [t for t in tokens if t.strip()]


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
        self._doc_tokens = []
        self._doc_lens = []
        self._doc_freqs = defaultdict(int)
        for doc in documents:
            tokens = _tokenize(self.document_text(doc, self.field_weight))
            self._doc_tokens.append(tokens)
            self._doc_lens.append(len(tokens))
            for token in set(tokens):
                self._doc_freqs[token] += 1
        n_docs = len(documents)
        self._avgdl = (sum(self._doc_lens) / n_docs) if n_docs else 0.0
        self._idf = {
            token: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0) for token, freq in self._doc_freqs.items()
        }
        self._built = n_docs > 0

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 (文档行号, BM25 分数) 列表，按分数降序。"""
        if not self._built or top_k <= 0:
            return []
        query_tokens = _tokenize(query)
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
    """单域持久化索引：向量（FAISS）+ 稀疏（BM25）+ 实体精确索引。

    - load()：从 index_root 读取 documents + faiss + manifest，
      校验一致性（count/维度/schema），失败抛 IndexNotAvailableError。
    - search_vector / search_sparse / entity_lookup：检索原语。
    - metadata 过滤在检索原语之上由 retrieval 层实现。
    """

    def __init__(self, domain_id: str, index_root: Path, dimension: int = 384):
        self.domain_id = domain_id
        self.index_root = Path(index_root)
        self.dimension = dimension
        self.documents: list[KnowledgeIndexDocument] = []
        self._faiss_index = None
        self._entity_rows: dict[str, list[int]] = {}
        self.bm25 = BM25Index()
        self.manifest: dict[str, Any] = {}

    # -- 文件路径 ---------------------------------------------------------
    @property
    def documents_path(self) -> Path:
        return self.index_root / DOCUMENTS_FILE

    @property
    def faiss_path(self) -> Path:
        return self.index_root / FAISS_FILE

    @property
    def manifest_path(self) -> Path:
        return self.index_root / MANIFEST_FILE

    @property
    def cache_path(self) -> Path:
        return self.index_root / CACHE_FILE

    # -- 加载 -------------------------------------------------------------
    def load(self) -> bool:
        """加载并校验索引；不可用返回 False（不抛错，调用方降级）。"""
        try:
            if not self.manifest_path.exists() or not self.documents_path.exists():
                logger.info("索引不存在: %s", self.index_root)
                return False
            with open(self.manifest_path, encoding="utf-8") as f:
                self.manifest = json.load(f)
            if self.manifest.get("schema_version") != SCHEMA_VERSION:
                logger.warning("索引 schema 版本不符: %s", self.manifest.get("schema_version"))
                return False
            if self.manifest.get("domain_id") != self.domain_id:
                logger.warning("索引 domain 不符: %s", self.manifest.get("domain_id"))
                return False

            documents: list[KnowledgeIndexDocument] = []
            with open(self.documents_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        documents.append(KnowledgeIndexDocument.from_dict(json.loads(line)))
            declared = int(self.manifest.get("document_count", -1))
            if declared != len(documents):
                logger.warning("索引 count 不一致: manifest=%s actual=%s", declared, len(documents))
                return False

            if documents:
                self._load_faiss(len(documents))
            else:
                self._faiss_index = None

            self.documents = documents
            self._rebuild_entity_index()
            self.bm25.build(documents)
            logger.info(
                "知识索引已加载: domain=%s docs=%d dim=%s",
                self.domain_id,
                len(documents),
                self.manifest.get("vector_dimension"),
            )
            return True
        except Exception as e:
            logger.warning("知识索引加载失败（domain=%s）: %s", self.domain_id, e)
            self.documents = []
            self._faiss_index = None
            self.manifest = {}
            return False

    def _load_faiss(self, expected_rows: int) -> None:
        if self.faiss_path.exists():
            import faiss

            self._faiss_index = faiss.read_index(str(self.faiss_path))
            if self._faiss_index.ntotal != expected_rows:
                raise ValueError(f"FAISS 行数不一致: index={self._faiss_index.ntotal} docs={expected_rows}")
            dim = int(self.manifest.get("vector_dimension", self.dimension))
            if self._faiss_index.d != dim:
                raise ValueError(f"FAISS 维度不一致: {self._faiss_index.d} != {dim}")
        else:
            if expected_rows:
                raise ValueError("documents 非空但 faiss.bin 缺失")
            self._faiss_index = None

    def _rebuild_entity_index(self) -> None:
        self._entity_rows = defaultdict(list)
        for row, doc in enumerate(self.documents):
            for entity in set(doc.entities):
                self._entity_rows[entity].append(row)

    # -- 写入（构建入口使用） ---------------------------------------------
    def write_atomic(
        self,
        documents: Sequence[KnowledgeIndexDocument],
        embeddings: np.ndarray,
        manifest: dict[str, Any],
    ) -> None:
        """原子写入索引（documents → faiss → manifest 最后替换）。"""
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"文档数与向量数不一致: docs={len(documents)} vectors={embeddings.shape[0]}")
        if documents and embeddings.shape[1] != self.dimension:
            raise ValueError(f"向量维度不符: 期望 {self.dimension} 实际 {embeddings.shape[1]}")
        self.index_root.mkdir(parents=True, exist_ok=True)

        tmp_docs = self.documents_path.with_name(DOCUMENTS_FILE + ".tmp")
        with open(tmp_docs, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.to_jsonl() + "\n")
        os.replace(str(tmp_docs), str(self.documents_path))

        tmp_faiss = self.faiss_path.with_name(FAISS_FILE + ".tmp")
        import faiss

        index = faiss.IndexFlatIP(int(embeddings.shape[1]) if documents else self.dimension)
        if documents:
            index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
        faiss.write_index(index, str(tmp_faiss))
        os.replace(str(tmp_faiss), str(self.faiss_path))

        manifest = {**manifest, "schema_version": SCHEMA_VERSION, "domain_id": self.domain_id}
        tmp_manifest = self.manifest_path.with_name(MANIFEST_FILE + ".tmp")
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(str(tmp_manifest), str(self.manifest_path))

        # 更新内存态
        self.documents = list(documents)
        self._faiss_index = index
        self.manifest = manifest
        self._rebuild_entity_index()
        self.bm25.build(self.documents)

    # -- 检索原语 ---------------------------------------------------------
    def search_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """向量近邻检索：返回 (行号, 相似度)，按相似度降序。"""
        if not self.documents or self._faiss_index is None or top_k <= 0:
            return []
        vector = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        k = min(top_k, self._faiss_index.ntotal)
        if k <= 0:
            return []
        scores, rows = self._faiss_index.search(vector, k)
        return [(int(row), float(score)) for row, score in zip(rows[0], scores[0]) if row >= 0 and score >= threshold]

    def search_sparse(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, top_k)

    def entity_lookup(self, entities: Iterable[str]) -> list[int]:
        """实体精确索引：返回包含任一实体的文档行号（去重升序）。"""
        rows: set = set()
        for entity in entities:
            rows.update(self._entity_rows.get(entity, []))
        return sorted(rows)

    def get_document(self, row: int) -> KnowledgeIndexDocument | None:
        if 0 <= row < len(self.documents):
            return self.documents[row]
        return None

    # -- 状态 -------------------------------------------------------------
    def count(self) -> int:
        return len(self.documents)

    def is_healthy(self) -> bool:
        """一致性检查：manifest 与 documents/faiss 行数闭合。"""
        declared = self.manifest.get("document_count")
        return declared is not None and declared == len(self.documents)

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


def build_domain_index(
    config,
    documents: Sequence[KnowledgeIndexDocument],
    embeddings: np.ndarray,
    embedding_provider: EmbeddingProvider,
    source_fingerprint: str,
    build_params: dict[str, Any] | None = None,
) -> DomainIndex:
    """构建并原子落盘一个域索引，返回加载后的 DomainIndex。"""
    index = DomainIndex(
        domain_id=config.domain_id,
        index_root=config.resolve_index_root(),
        dimension=int(embeddings.shape[1]) if documents else embedding_provider.dimension,
    )
    type_counts: dict[str, int] = defaultdict(int)
    for doc in documents:
        type_counts[doc.document_type] += 1
    manifest = {
        "index_version": config.index_version,
        "document_count": len(documents),
        "document_type_counts": dict(type_counts),
        "embedding_provider": type(embedding_provider).__name__,
        "embedding_model_id": getattr(embedding_provider, "model_id", "unknown"),
        "vector_dimension": int(embeddings.shape[1]) if documents else embedding_provider.dimension,
        "distance_metric": "cosine (normalized inner product)",
        "source_fingerprint": source_fingerprint,
        "build_params": dict(build_params or {}),
    }
    index.write_atomic(documents, embeddings, manifest)
    return index
