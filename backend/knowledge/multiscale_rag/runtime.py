"""Production runtime for multi-scale character knowledge retrieval."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.index import DomainIndex
from knowledge.retrieval_core.query import QueryAnalyzer
from knowledge.retrieval_core.registry import KnowledgeDomainConfig, get_default_registry

from .constants import (
    DEFAULT_INDEX_DIRECTORY,
    EMBEDDING_TEXT_VERSION,
    INDEX_FORMAT_VERSION,
    LEGACY_EMBEDDING_TEXT_VERSIONS,
    LEGACY_INDEX_FORMAT_VERSIONS,
)
from .service import CARD_TYPES, RoutedMultiScaleService
from .source_text import OriginalTextExtractor
from .vector_runtime import LocalMeanPoolingEmbeddingProvider, attach_vectors

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_ROOT.parent
_DEFAULT_INDEX_ROOT = _BACKEND_ROOT / "data" / "knowledge" / "tsukiyashiro_kisaki" / DEFAULT_INDEX_DIRECTORY


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _load_bundle(root: Path) -> tuple[list[KnowledgeIndexDocument], np.ndarray, dict[str, Any]]:
    documents_path = root / "documents.jsonl"
    vectors_path = root / "vectors.npy"
    manifest_path = root / "manifest.json"
    if not documents_path.exists() or not vectors_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"多粒度索引不完整: {root}")
    documents = [
        KnowledgeIndexDocument.from_dict(json.loads(line))
        for line in documents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.asarray(np.load(vectors_path, allow_pickle=False), dtype=np.float32)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if vectors.shape != (len(documents), 384):
        raise ValueError(f"多粒度索引数量或维度异常: {root} {vectors.shape} docs={len(documents)}")
    return documents, vectors, manifest


def _memory_index(domain_id: str, documents: list[KnowledgeIndexDocument], vectors: np.ndarray) -> DomainIndex:
    index = DomainIndex(domain_id, Path("__character_rag_runtime__"), int(vectors.shape[1]))
    index.documents = documents
    index._rebuild_entity_index()
    index.bm25.build(documents)
    index.manifest = {
        "document_count": len(documents),
        "index_version": INDEX_FORMAT_VERSION,
    }
    attach_vectors(index, vectors)
    return index


def _route_indexes(
    domain_id: str,
    documents: list[KnowledgeIndexDocument],
    vectors: np.ndarray,
) -> dict[frozenset[str], DomainIndex]:
    routes = {
        CARD_TYPES,
        frozenset({"fact"}),
        frozenset({"relation"}),
        frozenset({"event"}),
        frozenset({"fact", "relation"}),
        frozenset({"fact", "event"}),
        frozenset({"relation", "event"}),
        frozenset({"story", "scene"}),
        frozenset({"evidence"}),
    }
    indexes: dict[frozenset[str], DomainIndex] = {}
    for route in routes:
        rows = [row for row, document in enumerate(documents) if document.document_type in route]
        if rows:
            indexes[route] = _memory_index(
                domain_id,
                [documents[row] for row in rows],
                vectors[rows],
            )
    return indexes


class MultiScaleRagRuntime:
    """Lazy production facade with the same bundle contract used by generation."""

    def __init__(self, index_root: Path | None = None) -> None:
        configured = (
            os.getenv("CHARACTER_RAG_INDEX_ROOT", "").strip() or os.getenv("MULTISCALE_RAG_INDEX_ROOT", "").strip()
        )
        self.index_root = Path(configured) if configured else Path(index_root or _DEFAULT_INDEX_ROOT)
        self._load_lock = threading.Lock()
        self._loaded = False
        self._load_attempted = False
        self._warmup_started = False
        self._service: RoutedMultiScaleService | None = None
        self._base_config = get_default_registry().require("tsukiyashiro_kisaki")
        self._gate = QueryAnalyzer([self._base_config])
        self._provider = LocalMeanPoolingEmbeddingProvider(
            model_path=os.getenv("EMBEDDING_MODEL_PATH", "").strip() or None
        )
        self._stats: dict[str, Any] = {}

    @property
    def config(self) -> KnowledgeDomainConfig:
        return self._base_config

    def _ensure_loaded(self) -> bool:
        if self._loaded and self._service is not None:
            return True
        with self._load_lock:
            if self._loaded and self._service is not None:
                return True
            self._load_attempted = True
            try:
                bundles = [
                    _load_bundle(self.index_root / "card_index"),
                    _load_bundle(self.index_root / "scene_story_index"),
                    _load_bundle(self.index_root / "evidence_index"),
                ]
                documents = [document for docs, _, _ in bundles for document in docs]
                vectors = np.vstack([matrix for _, matrix, _ in bundles])
                domain_id = self._base_config.domain_id
                for document in documents:
                    # The promoted experiment index is format-compatible. Expose
                    # stable production identifiers without rewriting artifacts.
                    document.domain_id = domain_id
                    if document.index_version in LEGACY_INDEX_FORMAT_VERSIONS:
                        document.index_version = INDEX_FORMAT_VERSION
                    text_version = document.metadata.get("embedding_text_version")
                    if text_version in LEGACY_EMBEDDING_TEXT_VERSIONS:
                        document.metadata["embedding_text_version"] = EMBEDDING_TEXT_VERSION
                config = KnowledgeDomainConfig(
                    domain_id=domain_id,
                    source_root=self._base_config.source_root,
                    loader=lambda _root: [],
                    document_types=["story", "scene", "fact", "relation", "event", "evidence"],
                    aliases=dict(self._base_config.aliases),
                    story_titles=list(self._base_config.story_titles),
                    narrative_policy=self._base_config.narrative_policy,
                    retrieval_defaults=self._base_config.retrieval_defaults,
                    prompt_supplement=self._base_config.prompt_supplement,
                    index_version=INDEX_FORMAT_VERSION,
                    enabled=True,
                )
                self._service = RoutedMultiScaleService(
                    config,
                    _route_indexes(domain_id, documents, vectors),
                    self._provider,
                    all_documents=documents,
                    source_extractor=OriginalTextExtractor(_REPO_ROOT),
                )
                counts: dict[str, int] = {}
                for document in documents:
                    counts[document.document_type] = counts.get(document.document_type, 0) + 1
                self._stats = {
                    "available": True,
                    "domain_id": self._base_config.domain_id,
                    "index_root": str(self.index_root),
                    "documents": len(documents),
                    "document_type_counts": counts,
                    "embedding_model": self._provider.model_id,
                }
                self._loaded = True
                logger.info("多粒度角色知识索引已加载: %s", self._stats)
                return True
            except Exception as exc:  # noqa: BLE001 - caller owns controlled fallback
                self._loaded = False
                self._service = None
                self._stats = {"available": False, "index_root": str(self.index_root), "error": str(exc)}
                logger.warning("多粒度角色知识索引加载失败: %s", exc)
                return False

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def is_warm(self) -> bool:
        """Return without I/O; used only to choose the cold-start timeout."""
        return self._loaded and self._service is not None and self._provider._model is not None

    def stats(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._stats)

    def warmup_async(self) -> None:
        if self._warmup_started:
            return
        self._warmup_started = True

        def _warm() -> None:
            if self._ensure_loaded():
                try:
                    self._provider.embed_query("多粒度检索预热")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("多粒度角色知识检索 embedding 预热失败: %s", exc)

        threading.Thread(target=_warm, name="character-rag-warmup", daemon=True).start()

    def retrieve_with_citations(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        domain_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not query.strip() or not self._ensure_loaded() or self._service is None:
            return None
        # knowledge_base_id belongs to the generic user-managed KB. Character
        # knowledge retrieval owns a curated work domain and must not silently
        # ignore that filter.
        if filters:
            return None
        if domain_id and domain_id not in {self._base_config.domain_id, self._service.config.domain_id}:
            return None
        gate = self._gate.analyze(query)
        if not domain_id and not gate.matched_domains:
            return None
        result = self._service.retrieve(query, top_k=top_k)
        results = result.get("results", [])
        top = results[0] if results else {}
        confidence = float(top.get("rerank_score") or 0.0)
        if confidence <= 0:
            confidence = min(1.0, float(top.get("fused_score") or 0.0) * 8.0)
        threshold_name = (
            "CHARACTER_RAG_ABSTAIN_THRESHOLD"
            if os.getenv("CHARACTER_RAG_ABSTAIN_THRESHOLD") is not None
            else "MULTISCALE_RAG_ABSTAIN_THRESHOLD"
        )
        threshold = _env_float(threshold_name, 0.25)
        abstained = not results or confidence < threshold
        return {
            **result,
            "confidence": round(confidence, 4),
            "abstained": abstained,
            "domains": [self._base_config.domain_id],
            "query_analysis": {
                "entities": gate.entities,
                "matched_domains": gate.matched_domains,
                "route_types": result.get("route_types", []),
            },
            "warnings": [],
        }


_runtime: MultiScaleRagRuntime | None = None
_runtime_lock = threading.Lock()


def get_multiscale_rag_service() -> MultiScaleRagRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = MultiScaleRagRuntime()
                _runtime.warmup_async()
    return _runtime


def reset_multiscale_rag_service() -> None:
    global _runtime
    with _runtime_lock:
        _runtime = None
