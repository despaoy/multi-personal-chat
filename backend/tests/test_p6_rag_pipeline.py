"""P6 统一知识 RAG 管线定向测试。

覆盖最小验证清单：
- approved 输入门禁（未批准数据拒绝）
- canonical document 数量闭合（真实数据 218/32/159）
- 统一文档契约字段与 embedding_text 覆盖面
- embedding 缓存键绑定与增量复用
- 索引持久化后可重新加载（原子写 + 一致性校验）
- sparse / vector / hybrid 查询可执行
- reranker 确定性降级与 CrossEncoder 注入
- context builder 预算与去重
- 域门控（普通聊天不命中游戏域）
- 业务入口 bundle 兼容（api/generate._retrieve_rag_bundle）

测试不依赖真实 embedding 模型安装（FakeEmbeddingProvider 注入），
符合项目测试隔离约束。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge.rag_pipeline.context import ContextBuilder  # noqa: E402
from knowledge.rag_pipeline.documents import KnowledgeIndexDocument  # noqa: E402
from knowledge.rag_pipeline.embedding import (  # noqa: E402
    EmbeddingCache,
    embedding_cache_key,
)
from knowledge.rag_pipeline.index import DomainIndex  # noqa: E402
from knowledge.rag_pipeline.loaders import (  # noqa: E402
    AliasEntityNormalizer,
    ApprovedCardsLoader,
    UnapprovedDataError,
)
from knowledge.rag_pipeline.pipeline import RagPipeline  # noqa: E402
from knowledge.rag_pipeline.query import QueryAnalyzer  # noqa: E402
from knowledge.rag_pipeline.registry import (  # noqa: E402
    DomainRegistry,
    KnowledgeDomainConfig,
    NarrativeLayerPolicy,
    RetrievalDefaults,
)
from knowledge.rag_pipeline.rerank import DeterministicReranker, PipelineReranker  # noqa: E402

if TYPE_CHECKING:
    from knowledge.rag_pipeline.retrieval import HybridRetriever

BACKEND_DIR = Path(__file__).resolve().parents[1]
REAL_SOURCE = BACKEND_DIR / "data" / "knowledge" / "tsukiyashiro_kisaki" / "knowledge_candidate_review"


# ---------------------------------------------------------------------------
# 测试用 Fake embedding（确定性，无模型依赖）
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """确定性哈希向量 provider（测试隔离用）。"""

    def __init__(self, dimension: int = 8, model_id: str = "fake-embed", salt: str = ""):
        self.dimension = dimension
        self.model_id = model_id
        self.salt = salt

    @property
    def model_fingerprint(self) -> str:
        return "fake-fp-" + self.salt

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256((self.salt + text).encode("utf-8")).digest()
        raw = np.frombuffer(digest[: self.dimension], dtype=np.uint8).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm == 0:
            return np.ones(self.dimension, dtype=np.float32) / np.sqrt(self.dimension)
        return (raw / norm).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)


# ---------------------------------------------------------------------------
# 测试数据：小型 approved 卡 fixture
# ---------------------------------------------------------------------------

FACT_CARD = {
    "id": "fact_test_0001",
    "document_type": "fact",
    "title": "测试者的身份",
    "subject": "甲",
    "predicate": "身份",
    "value": "虚构学园的学生",
    "summary": "甲是虚构学园的学生",
    "evidence_text": "[甲] 「我在虚构学园上学。」\n长证据文本第二行。",
    "story": {
        "volume_number": 1,
        "story_unit_id": "vol01_test",
        "story_title": "1测试卷",
        "continuity_id": None,
        "sequence_order": None,
        "viewpoint": "甲第一人称",
        "content_scope": "main_story",
        "temporal_scope": "current",
        "route": None,
    },
    "source": {"source_path": "gametext/test/1测试卷.txt", "line_start": 10, "line_end": 20},
    "reality_status": "objective",
    "review_status": "approved",
}

RELATION_CARD = {
    "id": "rel_test_0001",
    "document_type": "relation",
    "title": "甲与乙的关系",
    "subject": "甲",
    "relation": "朋友",
    "target": "乙",
    "summary": "甲与乙是旧时好友",
    "evidence_text": "[甲] 「乙是我的旧时好友。」",
    "story": {
        "volume_number": 2,
        "story_unit_id": "vol02_test",
        "story_title": "2测试卷",
        "viewpoint": "甲第一人称",
        "content_scope": "main_story",
        "temporal_scope": "flashback",
        "route": None,
    },
    "source": {"source_path": "gametext/test/2测试卷.txt", "line_start": 5, "line_end": 8},
    "reality_status": "objective",
    "review_status": "approved",
}

EVENT_CARD = {
    "id": "event_test_0001",
    "document_type": "event",
    "title": "甲与乙初次见面",
    "summary": "甲与乙在教室初次见面",
    "participants": ["甲", "乙"],
    "causes": ["乙转学来到虚构学园"],
    "outcomes": ["甲与乙成为朋友"],
    "evidence_text": "[乙] 「初次见面。」",
    "story": {
        "volume_number": 1,
        "story_unit_id": "vol01_test",
        "story_title": "1测试卷",
        "viewpoint": "第三人称",
        "content_scope": "bonus_story",
        "temporal_scope": "current",
        "route": None,
    },
    "source": {"source_path": "gametext/test/1测试卷.txt", "line_start": 100, "line_end": 120},
    "reality_status": "objective",
    "review_status": "approved",
}

ALIASES = {"甲": "甲", "乙": "乙", "大甲": "甲", "小乙": "乙", "虚构学园": "虚构学园"}
STORY_TITLES = ["1测试卷", "2测试卷"]


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    root = tmp_path / "approved"
    root.mkdir()
    for name, card in [
        ("facts_approved.jsonl", FACT_CARD),
        ("relations_approved.jsonl", RELATION_CARD),
        ("events_approved.jsonl", EVENT_CARD),
    ]:
        with open(root / name, "w", encoding="utf-8") as f:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    return root


def make_domain_config(
    source_root: Path,
    index_root: Path,
    domain_id: str = "test_domain",
) -> KnowledgeDomainConfig:
    normalizer = AliasEntityNormalizer(ALIASES)
    loader = ApprovedCardsLoader(
        domain_id=domain_id,
        index_version="v1",
        entity_normalizer=normalizer,
    )
    return KnowledgeDomainConfig(
        domain_id=domain_id,
        source_root=source_root,
        loader=loader,
        aliases=ALIASES,
        story_titles=STORY_TITLES,
        narrative_policy=NarrativeLayerPolicy(),
        retrieval_defaults=RetrievalDefaults(),
        index_root=index_root,
    )


@pytest.fixture()
def domain_config(source_dir: Path, tmp_path: Path) -> KnowledgeDomainConfig:
    return make_domain_config(source_dir, tmp_path / "rag_index")


def build_test_index(config: KnowledgeDomainConfig, provider: FakeEmbeddingProvider) -> DomainIndex:
    documents = config.loader(config.source_root)
    embeddings = provider.embed_texts([doc.embedding_text for doc in documents])
    from knowledge.rag_pipeline.index import build_domain_index

    return build_domain_index(config, documents, embeddings, provider, source_fingerprint="test-fp")


# ---------------------------------------------------------------------------
# Loader 与 approved 门禁
# ---------------------------------------------------------------------------


class TestLoader:
    def test_loads_all_document_types(self, domain_config: KnowledgeDomainConfig):
        docs = domain_config.loader(domain_config.source_root)
        types = sorted(doc.document_type for doc in docs)
        assert types == ["event", "fact", "relation"]

    def test_rejects_unapproved_data(self, source_dir: Path, tmp_path: Path):
        bad = dict(FACT_CARD)
        bad["review_status"] = "pending"
        bad["id"] = "fact_test_0002"
        with open(source_dir / "facts_approved.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(bad, ensure_ascii=False) + "\n")
        config = make_domain_config(source_dir, tmp_path / "rag_index")
        with pytest.raises(UnapprovedDataError):
            config.loader(source_dir)

    def test_fact_embedding_text_covers_required_fields(self, domain_config):
        doc = domain_config.loader(domain_config.source_root)[0]
        assert doc.document_type == "fact"
        for part in ("甲", "身份", "虚构学园的学生", "测试者的身份", "甲是虚构学园的学生", "证据", "1测试卷"):
            assert part in doc.embedding_text, f"embedding_text 缺少 {part}"
        # 不允许整份 JSON 序列化
        assert '"subject"' not in doc.embedding_text

    def test_relation_embedding_text_covers_direction(self, domain_config):
        docs = domain_config.loader(domain_config.source_root)
        rel = next(d for d in docs if d.document_type == "relation")
        for part in ("甲", "乙", "朋友", "甲与乙是旧时好友", "证据", "2测试卷"):
            assert part in rel.embedding_text
        assert rel.relations == ["甲-朋友-乙"]

    def test_event_embedding_text_covers_causality(self, domain_config):
        docs = domain_config.loader(domain_config.source_root)
        event = next(d for d in docs if d.document_type == "event")
        for part in ("甲与乙初次见面", "参与者", "甲", "乙", "起因", "转学", "结果", "成为朋友"):
            assert part in event.embedding_text

    def test_full_evidence_preserved_in_content_and_source(self, domain_config):
        docs = domain_config.loader(domain_config.source_root)
        fact = docs[0]
        assert "[甲] 「我在虚构学园上学。」" in fact.content
        assert fact.source.source_path == "gametext/test/1测试卷.txt"
        assert fact.source.line_start == 10
        assert fact.source.card_id == "fact_test_0001"

    def test_entities_precise_and_mentions_separated(self, domain_config):
        docs = domain_config.loader(domain_config.source_root)
        fact = docs[0]
        # 结构化字段的实体进入 entities
        assert "甲" in fact.entities
        # evidence 提及的实体不进入 entities（此处 evidence 只提甲）
        rel = next(d for d in docs if d.document_type == "relation")
        assert set(rel.entities) >= {"甲", "乙"}
        assert "mentioned_entities" in rel.metadata

    def test_narrative_layers_mapped(self, domain_config):
        docs = domain_config.loader(domain_config.source_root)
        rel = next(d for d in docs if d.document_type == "relation")
        assert rel.temporal_scope == "flashback"
        event = next(d for d in docs if d.document_type == "event")
        assert event.content_scope == "bonus_story"
        assert docs[0].reality_status == "objective"

    def test_duplicate_ids_rejected(self, source_dir: Path, tmp_path: Path):
        with open(source_dir / "facts_approved.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(FACT_CARD, ensure_ascii=False) + "\n")
        config = make_domain_config(source_dir, tmp_path / "rag_index")
        with pytest.raises(ValueError, match="ID 重复"):
            config.loader(source_dir)

    @pytest.mark.skipif(not REAL_SOURCE.exists(), reason="真实 approved 数据不存在")
    def test_real_data_count_closure(self):
        """正式知识域 canonical 数量闭合：409 = 218 fact + 32 relation + 159 event。"""
        from knowledge.rag_pipeline.registry import get_default_registry

        registry = get_default_registry()
        config = registry.require("tsukiyashiro_kisaki")
        docs = config.loader(config.source_root)
        counts: dict[str, int] = {}
        for doc in docs:
            counts[doc.document_type] = counts.get(doc.document_type, 0) + 1
        assert counts == {"fact": 218, "relation": 32, "event": 159}
        assert len(docs) == 409
        assert all(doc.review_status == "approved" for doc in docs)


# ---------------------------------------------------------------------------
# embedding 缓存
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_cache_key_binds_model_and_text(self):
        key1 = embedding_cache_key("d1", "doc1", "fp1", "model-a", "mfp1")
        key2 = embedding_cache_key("d1", "doc1", "fp1", "model-b", "mfp1")
        key3 = embedding_cache_key("d1", "doc1", "fp2", "model-a", "mfp1")
        key4 = embedding_cache_key("d2", "doc1", "fp1", "model-a", "mfp1")
        assert len({key1, key2, key3, key4}) == 4

    def test_cache_roundtrip_and_prune(self, tmp_path: Path):
        cache = EmbeddingCache(tmp_path / "cache.npz")
        cache.load()
        vector = np.ones(4, dtype=np.float32)
        cache.put("k1", vector)
        cache.put("k2", vector * 2)
        cache.save()

        cache2 = EmbeddingCache(tmp_path / "cache.npz")
        assert cache2.load() == 2
        assert np.allclose(cache2.get("k1"), vector)
        dropped = cache2.prune_to({"k1"})
        assert dropped == 1
        cache2.save()

        cache3 = EmbeddingCache(tmp_path / "cache.npz")
        cache3.load()
        assert cache3.get("k2") is None
        assert np.allclose(cache3.get("k1"), vector)

    def test_fake_provider_dimension_consistency(self):
        provider = FakeEmbeddingProvider(dimension=16)
        matrix = provider.embed_texts(["a", "b"])
        assert matrix.shape == (2, 16)
        # 归一化向量：内积 = 余弦
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 索引持久化与检索原语
# ---------------------------------------------------------------------------


class TestDomainIndex:
    def test_persist_and_reload(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        assert index.count() == 3
        assert index.is_healthy()

        reloaded = DomainIndex("test_domain", domain_config.resolve_index_root(), provider.dimension)
        assert reloaded.load()
        assert reloaded.count() == 3
        assert reloaded.is_healthy()
        assert [doc.id for doc in reloaded.documents] == [doc.id for doc in index.documents]

    def test_manifest_records_build_info(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        manifest = index.manifest
        assert manifest["document_count"] == 3
        assert manifest["domain_id"] == "test_domain"
        assert manifest["embedding_model_id"] == "fake-embed"
        assert manifest["vector_dimension"] == provider.dimension
        assert manifest["schema_version"] == 1
        assert manifest["source_fingerprint"] == "test-fp"

    def test_corrupted_manifest_fails_health(self, domain_config):
        provider = FakeEmbeddingProvider()
        build_test_index(domain_config, provider)
        manifest_path = domain_config.resolve_index_root() / "index_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["document_count"] = 999
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reloaded = DomainIndex("test_domain", domain_config.resolve_index_root(), provider.dimension)
        assert reloaded.load() is False

    def test_vector_search_executable(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        query_vector = provider.embed_query("甲的身份是什么")
        results = index.search_vector(query_vector, top_k=3)
        assert 1 <= len(results) <= 3
        assert all(isinstance(row, int) and isinstance(score, float) for row, score in results)

    def test_sparse_search_executable(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        results = index.search_sparse("甲 乙 朋友", top_k=3)
        assert results, "BM25 应能命中"

    def test_entity_lookup(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        rows = index.entity_lookup(["甲"])
        assert rows, "实体索引应命中含甲的文档"

    def test_embedding_dimension_mismatch_rejected(self, domain_config):
        documents = domain_config.loader(domain_config.source_root)
        embeddings16 = FakeEmbeddingProvider(dimension=16).embed_texts([doc.embedding_text for doc in documents])
        index = DomainIndex("test_domain", domain_config.resolve_index_root(), 8)
        with pytest.raises(ValueError, match="维度"):
            index.write_atomic(documents, embeddings16, {"document_count": len(documents)})


# ---------------------------------------------------------------------------
# 查询分析器
# ---------------------------------------------------------------------------


class TestQueryAnalyzer:
    @pytest.fixture()
    def analyzer(self) -> QueryAnalyzer:
        return QueryAnalyzer([make_domain_config(Path("unused"), Path("unused"))])

    def test_alias_normalization(self, analyzer: QueryAnalyzer):
        analysis = analyzer.analyze("大甲的目标是什么")
        assert analysis.entities == ["甲"]
        assert "甲" in analysis.normalized_query
        assert analysis.matched_domains == ["test_domain"]

    def test_plain_chat_does_not_match_domain(self, analyzer: QueryAnalyzer):
        analysis = analyzer.analyze("今天天气怎么样")
        assert analysis.matched_domains == []
        assert analysis.entities == []

    def test_single_char_entity_with_intent_matches(self, analyzer: QueryAnalyzer):
        analysis = analyzer.analyze("甲的哥哥是谁")
        assert analysis.matched_domains == ["test_domain"]

    def test_single_char_entity_final_death_query_prefers_objective_fact(self, analyzer: QueryAnalyzer):
        analysis = analyzer.analyze("甲最终是怎么死的")
        assert analysis.matched_domains == ["test_domain"]
        assert analysis.entities == ["甲"]
        assert analysis.doc_type_preferences == ["fact"]
        assert analysis.reality_preferences == ["objective"]

    def test_multi_entity_matches(self, analyzer: QueryAnalyzer):
        analysis = analyzer.analyze("甲和乙是什么关系")
        assert analysis.matched_domains == ["test_domain"]
        assert set(analysis.entities) >= {"甲", "乙"}
        assert "relation" in analysis.doc_type_preferences

    def test_story_title_match(self, analyzer: QueryAnalyzer):
        analysis = analyzer.analyze("1测试卷讲了什么故事")
        assert analysis.matched_domains == ["test_domain"]
        assert analysis.story_hits == ["1测试卷"]

    def test_narrative_preferences(self, analyzer: QueryAnalyzer):
        analysis = analyzer.analyze("甲在梦里发生了什么")
        assert "fictional" in analysis.reality_preferences
        analysis2 = analyzer.analyze("甲过去发生了什么")
        assert "flashback" in analysis2.temporal_preferences


# ---------------------------------------------------------------------------
# 混合检索
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    @pytest.fixture()
    def retriever(self, domain_config) -> HybridRetriever:
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        from knowledge.rag_pipeline.pipeline import DomainRuntime

        runtime = DomainRuntime(domain_config, index, provider)
        return runtime.retriever

    def test_sparse_mode(self, retriever: HybridRetriever, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        results = retriever.search(analysis, top_k=3, mode="sparse")
        assert results, "sparse 模式应可执行"

    def test_vector_mode(self, retriever: HybridRetriever, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲的身份是什么")
        results = retriever.search(analysis, top_k=3, mode="vector")
        assert results

    def test_hybrid_mode_fusion_scores(self, retriever: HybridRetriever, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        results = retriever.search(analysis, top_k=3, mode="hybrid")
        assert results
        for candidate in results:
            assert candidate.rrf_score > 0
            assert candidate.fused_score >= candidate.rrf_score * 0.5  # 层级乘数下限

    def test_metadata_filter_document_type(self, retriever: HybridRetriever, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        results = retriever.search(analysis, top_k=3, mode="hybrid", filters={"document_type": "relation"})
        assert results
        assert all(c.document.document_type == "relation" for c in results)

    def test_vector_channel_failure_degrades_to_sparse(self, domain_config):
        class BrokenProvider(FakeEmbeddingProvider):
            def embed_query(self, query: str) -> np.ndarray:
                raise RuntimeError("embedding 不可用")

        provider = BrokenProvider()
        index = build_test_index(domain_config, provider)
        from knowledge.rag_pipeline.pipeline import DomainRuntime

        retriever = DomainRuntime(domain_config, index, provider).retriever
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        results = retriever.search(analysis, top_k=3, mode="hybrid")
        # 向量通道失败，稀疏通道仍应返回结果
        assert results


# ---------------------------------------------------------------------------
# 重排
# ---------------------------------------------------------------------------


class TestReranker:
    def _analysis(self, domain_config) -> Any:
        return QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")

    def _candidates(self, domain_config, retriever_like) -> list:
        analysis = self._analysis(domain_config)
        return retriever_like.search(analysis, top_k=10, mode="hybrid")

    def test_deterministic_fallback_stable(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        from knowledge.rag_pipeline.pipeline import DomainRuntime

        runtime = DomainRuntime(domain_config, index, provider)
        analysis = self._analysis(domain_config)
        candidates = runtime.retriever.search(analysis, top_k=10, mode="hybrid")

        reranker = PipelineReranker(cross_encoder_enabled=False)
        result1 = reranker.rerank(analysis, list(candidates), top_k=3)
        result2 = reranker.rerank(analysis, list(candidates), top_k=3)
        assert [c.document.id for c in result1] == [c.document.id for c in result2]
        assert all(c.rerank_score is not None for c in result1)
        # 原始分数保留
        assert all(c.fused_score is not None for c in result1)
        assert reranker.uses_cross_encoder is False

    def test_cross_encoder_injection(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        from knowledge.rag_pipeline.pipeline import DomainRuntime

        runtime = DomainRuntime(domain_config, index, provider)
        analysis = self._analysis(domain_config)
        candidates = runtime.retriever.search(analysis, top_k=10, mode="hybrid")

        class FakeEncoder:
            def rerank(self, query, docs, top_k=5):
                ranked = [{**d, "rerank_score": 10.0 - i * 0.5} for i, d in enumerate(docs)]
                return ranked[:top_k]

        reranker = PipelineReranker(cross_encoder=FakeEncoder())
        result = reranker.rerank(analysis, list(candidates), top_k=2)
        assert len(result) == 2
        assert reranker.uses_cross_encoder is True
        assert result[0].rerank_score >= result[1].rerank_score

    def test_cross_encoder_without_model_falls_back(self, domain_config):
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        from knowledge.rag_pipeline.pipeline import DomainRuntime

        runtime = DomainRuntime(domain_config, index, provider)
        analysis = self._analysis(domain_config)
        candidates = runtime.retriever.search(analysis, top_k=10, mode="hybrid")

        class NoModelEncoder:
            def rerank(self, query, docs, top_k=5):
                return [dict(d) for d in docs[:top_k]]  # 无 rerank_score：模型未加载

        reranker = PipelineReranker(cross_encoder=NoModelEncoder())
        result = reranker.rerank(analysis, list(candidates), top_k=2)
        assert reranker.uses_cross_encoder is False  # 降级到确定性重排
        assert all(c.rerank_score is not None for c in result)

    def test_deterministic_reranker_preserves_relation_hit(self, domain_config):
        """精确实体+关系方向命中不应被长 evidence 文档吞掉。"""
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        from knowledge.rag_pipeline.retrieval import RetrievalCandidate

        relation_doc = next(d for d in domain_config.loader(domain_config.source_root) if d.document_type == "relation")
        event_doc = next(d for d in domain_config.loader(domain_config.source_root) if d.document_type == "event")
        candidates = [
            RetrievalCandidate(row=0, document=event_doc, fused_score=0.05),
            RetrievalCandidate(row=1, document=relation_doc, fused_score=0.04),
        ]
        reranker = DeterministicReranker()
        scored = reranker.rerank(analysis, candidates)
        best_doc = scored[0][0].document
        assert best_doc.document_type == "relation"

    def test_deterministic_reranker_keeps_meaningful_single_char_term(self, domain_config):
        """单字实体另行计分，但“死”等单字语义词不能从覆盖率中丢失。"""
        from knowledge.rag_pipeline.retrieval import RetrievalCandidate

        documents = domain_config.loader(domain_config.source_root)
        death_fact = next(d for d in documents if d.document_type == "fact")
        death_fact.title = "甲死于交通事故"
        death_fact.summary = "甲的死亡原因是交通事故"
        unrelated_event = next(d for d in documents if d.document_type == "event")
        unrelated_event.title = "家庭会议"
        unrelated_event.summary = "甲最终接受了家庭会议的决定"
        candidates = [
            RetrievalCandidate(row=0, document=unrelated_event, fused_score=0.05),
            RetrievalCandidate(row=1, document=death_fact, fused_score=0.04),
        ]
        analysis = QueryAnalyzer([domain_config]).analyze("甲最终是怎么死的")
        scored = DeterministicReranker().rerank(analysis, candidates)
        assert scored[0][0].document.title == "甲死于交通事故"


# ---------------------------------------------------------------------------
# 上下文构建
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def _candidates(self, domain_config, analysis) -> list:
        provider = FakeEmbeddingProvider()
        index = build_test_index(domain_config, provider)
        from knowledge.rag_pipeline.pipeline import DomainRuntime

        runtime = DomainRuntime(domain_config, index, provider)
        return runtime.retriever.search(analysis, top_k=10, mode="hybrid")

    def test_budget_respected(self, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        candidates = self._candidates(domain_config, analysis)
        builder = ContextBuilder(budget_chars=300, max_items=5)
        context = builder.build(analysis, candidates, ["test_domain"])
        assert len(context.context_text) <= 300 + len("\n\n") * 5  # 允许少量分隔符余量
        assert context.used_chars > 0

    def test_no_duplicate_docs(self, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        candidates = self._candidates(domain_config, analysis)
        doubled = list(candidates) + list(candidates)
        builder = ContextBuilder()
        context = builder.build(analysis, doubled, ["test_domain"])
        ids = [item["id"] for item in context.items]
        assert len(ids) == len(set(ids))

    def test_equivalent_content_dedup(self, domain_config):
        """同主体+谓词的等价事实卡不重复占用上下文。"""
        from knowledge.rag_pipeline.retrieval import RetrievalCandidate

        docs = domain_config.loader(domain_config.source_root)
        fact1 = next(d for d in docs if d.document_type == "fact")
        fact2 = KnowledgeIndexDocument.from_dict(fact1.to_dict())
        fact2.id = "fact_test_dup"
        fact2.summary = fact1.summary + "（重复变体）"
        candidates = [
            RetrievalCandidate(row=0, document=fact1, fused_score=0.05),
            RetrievalCandidate(row=1, document=fact2, fused_score=0.045),
        ]
        analysis = QueryAnalyzer([domain_config]).analyze("甲的身份是什么")
        builder = ContextBuilder()
        context = builder.build(analysis, candidates, ["test_domain"])
        assert len(context.items) == 1
        assert context.skipped_duplicates == 1

    def test_citations_carry_source_info(self, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲与乙是什么关系")
        candidates = self._candidates(domain_config, analysis)
        builder = ContextBuilder()
        context = builder.build(analysis, candidates, ["test_domain"])
        assert context.citations
        for citation in context.citations:
            assert citation["source_id"]
            assert citation["source_path"]
            assert citation["domain_id"] == "test_domain"
            assert "score" in citation

    def test_evidence_wrapped_as_quote_not_instruction(self, domain_config):
        analysis = QueryAnalyzer([domain_config]).analyze("甲的身份是什么")
        candidates = self._candidates(domain_config, analysis)
        builder = ContextBuilder()
        context = builder.build(analysis, candidates, ["test_domain"])
        for item in context.items:
            block = item["block"]
            if "证据" in block:
                assert "「" in block, "evidence 必须以引用格式呈现"


# ---------------------------------------------------------------------------
# 管线与业务兼容 bundle
# ---------------------------------------------------------------------------


class TestPipeline:
    @pytest.fixture()
    def pipeline(self, domain_config) -> RagPipeline:
        build_test_index(domain_config, FakeEmbeddingProvider())
        registry = DomainRegistry()
        registry.register(domain_config)
        pipeline = RagPipeline(
            registry=registry,
            embedding_provider=FakeEmbeddingProvider(),
        )
        assert pipeline.load_indexes()
        return pipeline

    def test_no_domain_match_returns_none(self, pipeline: RagPipeline):
        bundle = pipeline.retrieve("今天天气怎么样")
        assert bundle is None

    def test_domain_match_returns_bundle_contract(self, pipeline: RagPipeline):
        bundle = pipeline.retrieve("甲与乙是什么关系")
        assert bundle is not None
        for key in ("results", "citations", "confidence", "abstained", "context_text", "domains"):
            assert key in bundle
        assert bundle["domains"] == ["test_domain"]
        assert bundle["results"]
        assert bundle["context_text"]

    def test_explicit_domain_retrieval(self, pipeline: RagPipeline):
        bundle = pipeline.retrieve("普通查询不带实体", domain_id="test_domain")
        # 显式指定域时即使无实体信号也执行检索
        assert bundle is not None

    def test_index_unavailable_degrades(self, tmp_path: Path):
        registry = DomainRegistry()
        config = make_domain_config(Path("unused"), tmp_path / "empty_index")
        registry.register(config)
        pipeline = RagPipeline(registry=registry, embedding_provider=FakeEmbeddingProvider())
        # 索引不存在 → 不可用 → retrieve 返回 None（不抛错）
        assert not pipeline.is_available()
        assert pipeline.retrieve("甲与乙是什么关系") is None

    def test_multi_domain_isolation(self, tmp_path: Path):
        """两个域各自独立注册、独立索引，互不串扰。"""
        registry = DomainRegistry()
        provider = FakeEmbeddingProvider()
        domain_ids = []
        for i in range(2):
            domain_id = f"test_domain_{i}"
            domain_ids.append(domain_id)
            source = tmp_path / f"approved_{i}"
            source.mkdir()
            for name, card in [
                ("facts_approved.jsonl", FACT_CARD),
                ("relations_approved.jsonl", RELATION_CARD),
                ("events_approved.jsonl", EVENT_CARD),
            ]:
                card_copy = dict(card)
                card_copy["id"] = f"{card['id']}_{i}"
                with open(source / name, "w", encoding="utf-8") as f:
                    f.write(json.dumps(card_copy, ensure_ascii=False) + "\n")
            config = make_domain_config(source, tmp_path / f"index_{i}", domain_id=domain_id)
            registry.register(config)
            build_test_index(config, provider)
        pipeline = RagPipeline(registry=registry, embedding_provider=provider)
        assert pipeline.load_indexes()
        assert len(pipeline._runtimes) == 2
        # 各域文档数量闭合且 id 不串扰
        for i, domain_id in enumerate(domain_ids):
            docs = pipeline._runtimes[domain_id].index.documents
            assert len(docs) == 3
            assert all(d.id.endswith(f"_{i}") for d in docs)
        # 显式指定域时只检索该域
        bundle = pipeline.retrieve("甲的身份是什么", domain_id=domain_ids[0])
        assert bundle["domains"] == [domain_ids[0]]
        assert all(r["domain_id"] == domain_ids[0] for r in bundle["results"])


# ---------------------------------------------------------------------------
# 业务入口（api/generate._retrieve_rag_bundle）
# ---------------------------------------------------------------------------


class TestBusinessIntegration:
    @pytest.fixture()
    def fake_service(self):
        class FakeService:
            def __init__(self):
                self.calls: list[str] = []

            def is_available(self) -> bool:
                return True

            def retrieve_with_citations(self, query, top_k=5, filters=None, domain_id=None):
                self.calls.append(query)
                return {
                    "results": [
                        {
                            "id": "fact_x1",
                            "domain_id": "tsukiyashiro_kisaki",
                            "document_type": "fact",
                            "title": "月社妃的身份",
                            "summary": "妃是鹰山学园三年级生",
                            "content": "妃是鹰山学园三年级生",
                            "score": 0.9,
                        }
                    ],
                    "citations": [
                        {
                            "source_id": "fact_x1",
                            "source_title": "月社妃的身份",
                            "evidence_excerpt": "妃是鹰山学园三年级生",
                            "score": 0.9,
                            "source_path": "gametext/纸上魔法使/5磷灰石的怠惰现象.txt",
                            "source_line": 1,
                        }
                    ],
                    "confidence": 0.85,
                    "abstained": False,
                    "context_text": "【事实】月社妃的身份\n妃是鹰山学园三年级生",
                    "domains": ["tsukiyashiro_kisaki"],
                }

        return FakeService()

    @pytest.mark.asyncio
    async def test_p6_bundle_used_when_domain_matched(self, fake_service, monkeypatch):
        pytest.importorskip("api.generate")
        import api.generate as generate_module
        import knowledge.rag_pipeline.service as service_module

        monkeypatch.setattr(service_module, "get_rag_pipeline_service", lambda: fake_service)
        bundle = await generate_module._retrieve_rag_bundle("月社妃是什么身份", 3, None)
        assert fake_service.calls == ["月社妃是什么身份"]
        assert bundle["citations"]
        assert bundle["context_text"]
        assert bundle["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_fallback_to_legacy_when_no_domain(self, monkeypatch):
        pytest.importorskip("api.generate")
        import api.generate as generate_module
        import knowledge.rag_pipeline.service as service_module

        class UnavailableService:
            def is_available(self) -> bool:
                return False

        monkeypatch.setattr(service_module, "get_rag_pipeline_service", lambda: UnavailableService())

        legacy_bundle = {
            "results": [{"id": "legacy_1", "title": "旧链路结果", "content": "c"}],
            "citations": [],
            "confidence": 0.5,
            "abstained": False,
        }

        import knowledge.rag_helper as rag_helper_module

        class FakeHelper:
            def retrieve_with_citations(self, query, top_k=5, filters=None, threshold=0.3):
                return legacy_bundle

        monkeypatch.setattr(rag_helper_module, "get_rag_helper", lambda: FakeHelper())
        monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")

        bundle = await generate_module._retrieve_rag_bundle("今天天气怎么样", 3, None)
        assert bundle is legacy_bundle

    @pytest.mark.asyncio
    async def test_p6_exception_falls_back_to_legacy(self, fake_service, monkeypatch):
        pytest.importorskip("api.generate")
        import api.generate as generate_module
        import knowledge.rag_pipeline.service as service_module

        class BrokenService:
            def is_available(self) -> bool:
                return True

            def retrieve_with_citations(self, query, top_k=5, filters=None, domain_id=None):
                raise RuntimeError("索引故障")

        monkeypatch.setattr(service_module, "get_rag_pipeline_service", lambda: BrokenService())

        legacy_bundle = {
            "results": [],
            "citations": [],
            "confidence": 0.0,
            "abstained": True,
        }
        import knowledge.rag_helper as rag_helper_module

        class FakeHelper:
            def retrieve_with_citations(self, query, top_k=5, filters=None, threshold=0.3):
                return legacy_bundle

        monkeypatch.setattr(rag_helper_module, "get_rag_helper", lambda: FakeHelper())
        monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")

        bundle = await generate_module._retrieve_rag_bundle("月社妃是什么身份", 3, None)
        assert bundle is legacy_bundle


# ---------------------------------------------------------------------------
# 通用评估框架契约
# ---------------------------------------------------------------------------


class TestEvaluationContract:
    def test_criteria_resolution(self, domain_config):
        sys.path.insert(0, str(BACKEND_DIR / "scripts"))
        try:
            import evaluate_rag_retrieval as evaluator

            documents = domain_config.loader(domain_config.source_root)
            expected = evaluator.resolve_expected_ids(
                documents, {"document_type": "relation", "subject": "甲", "relation_type": "朋友"}
            )
            assert expected == {"rel_test_0001"}

            expected_any = evaluator.resolve_expected_ids(
                documents,
                {
                    "either_of": [
                        {"document_type": "fact", "subject": "甲", "predicate": "身份"},
                        {"document_type": "event", "title_contains": "初次见面"},
                    ]
                },
            )
            assert expected_any == {"fact_test_0001", "event_test_0001"}
        finally:
            sys.path.remove(str(BACKEND_DIR / "scripts"))
