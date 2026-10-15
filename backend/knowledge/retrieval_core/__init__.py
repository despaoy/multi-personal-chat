"""Domain-independent primitives shared by knowledge retrieval services.

分层：
- documents  canonical 索引文档契约
- registry   知识域注册与配置（domain/alias/叙事层策略）
- loaders    Source Loader（approved 卡等来源 → canonical 文档）
- embedding  embedding providers and vector validation
- index      in-memory vector/sparse/entity indexes
- query      查询理解与别名归一
- retrieval  混合召回与 RRF 融合
- rerank     重排（CrossEncoder + 确定性降级）
This package is infrastructure, not an independently deployable RAG chain.
Character-domain orchestration lives in ``knowledge.multiscale_rag.runtime``.
"""

from .documents import KnowledgeIndexDocument, SourceReference
from .embedding import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_default_embedding_provider,
)
from .index import DomainIndex
from .loaders import AliasEntityNormalizer, ApprovedCardsLoader, UnapprovedDataError
from .query import QueryAnalysis, QueryAnalyzer
from .registry import (
    DomainRegistry,
    KnowledgeDomainConfig,
    NarrativeLayerPolicy,
    RetrievalDefaults,
    build_default_registry,
    get_default_registry,
)
from .rerank import DeterministicReranker, PipelineReranker
from .retrieval import HybridRetriever, RetrievalCandidate

__all__ = [
    "ApprovedCardsLoader",
    "AliasEntityNormalizer",
    "DeterministicReranker",
    "DomainIndex",
    "DomainRegistry",
    "EmbeddingProvider",
    "HybridRetriever",
    "KnowledgeDomainConfig",
    "KnowledgeIndexDocument",
    "NarrativeLayerPolicy",
    "PipelineReranker",
    "QueryAnalysis",
    "QueryAnalyzer",
    "RetrievalCandidate",
    "RetrievalDefaults",
    "SentenceTransformerEmbeddingProvider",
    "get_default_embedding_provider",
    "SourceReference",
    "UnapprovedDataError",
    "build_default_registry",
    "get_default_registry",
]
