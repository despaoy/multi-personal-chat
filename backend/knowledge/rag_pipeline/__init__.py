"""P6 统一知识 RAG 管线。

分层：
- documents  canonical 索引文档契约
- registry   知识域注册与配置（domain/alias/叙事层策略）
- loaders    Source Loader（approved 卡等来源 → canonical 文档）
- embedding  Embedding Provider 抽象 + 缓存
- index      持久化向量/稀疏/实体索引
- query      查询理解与别名归一
- retrieval  混合召回与 RRF 融合
- rerank     重排（CrossEncoder + 确定性降级）
- context    上下文预算组装与引用
- pipeline   编排
- service    业务接入门面（兼容现有 bundle 契约）

业务入口：knowledge.rag_pipeline.service.get_rag_pipeline_service()
构建入口：backend/scripts/build_knowledge_index.py
评估入口：backend/scripts/evaluate_rag_retrieval.py
"""

from .context import Citation, ContextBuilder, RetrievedContext
from .documents import KnowledgeIndexDocument, SourceReference
from .embedding import (
    EmbeddingCache,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    embedding_cache_key,
)
from .index import DomainIndex, build_domain_index
from .loaders import AliasEntityNormalizer, ApprovedCardsLoader, UnapprovedDataError
from .pipeline import RagPipeline
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
from .service import RagPipelineService, get_rag_pipeline_service, reset_rag_pipeline_service

__all__ = [
    "ApprovedCardsLoader",
    "AliasEntityNormalizer",
    "Citation",
    "ContextBuilder",
    "DeterministicReranker",
    "DomainIndex",
    "DomainRegistry",
    "EmbeddingCache",
    "EmbeddingProvider",
    "HybridRetriever",
    "KnowledgeDomainConfig",
    "KnowledgeIndexDocument",
    "NarrativeLayerPolicy",
    "PipelineReranker",
    "QueryAnalysis",
    "QueryAnalyzer",
    "RagPipeline",
    "RagPipelineService",
    "RetrievalCandidate",
    "RetrievalDefaults",
    "RetrievedContext",
    "SentenceTransformerEmbeddingProvider",
    "SourceReference",
    "UnapprovedDataError",
    "build_default_registry",
    "build_domain_index",
    "embedding_cache_key",
    "get_default_registry",
    "get_rag_pipeline_service",
    "reset_rag_pipeline_service",
]
