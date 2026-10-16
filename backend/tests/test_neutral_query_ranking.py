"""Literal evidence remains available without invented query categories."""

from types import SimpleNamespace

import numpy as np
import pytest

from knowledge.multiscale_rag.runtime import _memory_index
from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.query import QueryAnalysis, QueryAnalyzer
from knowledge.retrieval_core.registry import KnowledgeDomainConfig
from knowledge.retrieval_core.rerank import DeterministicReranker
from knowledge.retrieval_core.retrieval import HybridRetriever, RetrievalCandidate

BODY = "林栖是河湾馆登记员；林栖与穆远是工作搭档，没有亲属关系。昨晚仅复核错字，不新增授权。"
QUERIES = [
    "概括《虚构登记故事》的经过。",
    "不要把这段当作过去的假设；按完整原文核对《虚构登记故事》。",
    "林栖是谁？请保留原文中的条件和否定。",
]


def arrange(tmp_path):
    config = KnowledgeDomainConfig(
        "synthetic",
        tmp_path,
        lambda _: [],
        aliases={"林栖": "林栖", "小林": "林栖", "穆远": "穆远"},
        story_titles=["虚构登记故事"],
    )
    docs = [
        KnowledgeIndexDocument(
            kind,
            "synthetic",
            kind,
            "虚构登记故事",
            BODY,
            BODY,
            BODY,
            entities=["林栖", "穆远"],
            metadata=dict(story_title="虚构登记故事"),
        )
        for kind in ["fact", "scene", "story", "event"]
    ]
    vectors = np.ones((4, 384), dtype=np.float32) / np.sqrt(384)
    index = _memory_index(config.domain_id, docs, vectors)
    calls = []

    def encode(query):
        calls.append(query)
        return vectors[0].copy()

    retriever = HybridRetriever(config, index, SimpleNamespace(embed_query=encode))
    return SimpleNamespace(config=config, docs=docs, analyzer=QueryAnalyzer([config]), retriever=retriever, calls=calls)


@pytest.mark.parametrize("query", QUERIES)
def test_equivalent_evidence_views_are_not_promoted_by_intent_words(tmp_path, query):
    rig = arrange(tmp_path)
    analysis = rig.analyzer.analyze(query)
    candidates = rig.retriever.search(analysis, top_k=4)
    assert [c.document.id for c in candidates] == ["fact", "scene", "story", "event"]
    ranked = DeterministicReranker().rerank(analysis, candidates)
    assert [c.document.id for c, _ in ranked] == ["fact", "scene", "story", "event"]
    assert len({score for _, score in ranked}) == 1
    assert all(c.document.content == BODY for c, _ in ranked)


def test_full_original_and_registered_alias_are_the_only_query_views(tmp_path):
    rig = arrange(tmp_path)
    original = "  不要自行补充。\n小林与穆远没有亲属关系吗？仅核对原文。  "
    analysis = rig.analyzer.analyze(original)
    assert analysis.original_query == original
    assert analysis.normalized_query == original.replace("小林", "林栖")
    rig.retriever.search(analysis, top_k=4)
    assert rig.calls == [original]


def test_explicit_metadata_filter_remains_authoritative(tmp_path):
    rig = arrange(tmp_path)
    analysis = rig.analyzer.analyze(QUERIES[0])
    result = rig.retriever.search(analysis, top_k=4, filters={"document_type": "event"})
    assert [c.document.id for c in result] == ["event"]
    assert result[0].document.content == BODY


def test_long_complete_evidence_is_not_penalized_for_its_length():
    analysis = QueryAnalysis(original_query="林栖是谁？", normalized_query="林栖是谁？", entities=["林栖"])
    long = "\n".join([BODY] * 100 + ["最终限定：没有亲属关系。"])
    docs = [
        KnowledgeIndexDocument(identity, "synthetic", "fact", "登记材料", body, body, body, entities=["林栖"])
        for identity, body in [("complete-long", long), ("complete-short", BODY)]
    ]
    candidates = [RetrievalCandidate(row, doc, fused_score=0.7) for row, doc in enumerate(docs)]
    ranked = DeterministicReranker().rerank(analysis, candidates)
    assert [c.document.id for c, _ in ranked] == ["complete-long", "complete-short"]
    assert ranked[0][1] == ranked[1][1]
    assert ranked[0][0].document.content == long
