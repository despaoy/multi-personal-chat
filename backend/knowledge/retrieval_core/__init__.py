"""Domain-independent primitives shared by knowledge retrieval services.

分层：
- documents  canonical 索引文档契约
- registry   知识域注册与配置（domain/alias/叙事层策略）
- loaders    Source Loader（approved 卡等来源 → canonical 文档）
- embedding  embedding providers and vector validation
- index      in-memory vector/sparse/entity indexes
- query      查询理解与别名归一
- retrieval  混合召回与 RRF 融合
- rerank     模型重排及显式配置的排序
This package is infrastructure, not an independently deployable RAG chain.
Character-domain orchestration lives in ``knowledge.multiscale_rag.runtime``.
Import concrete modules at use sites; lightweight contracts need no engines.
"""
