"""Real chunk IDs keep complete evidence distinct through native retrieval and task merging."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.index_lifecycle import _vector_chunk_document
from knowledge import rag_helper
from knowledge.vector_db import IndexConfig, VectorDatabase

QUERY = "档案馆 周一开放时间 雨天预约例外 借阅期限 自动续借"
VIEWS = ("周一开放时间 雨天预约例外", "借阅期限 自动续借")
TEXTS = [
    "周一九点开放，必须预约；雨天停办，预约不豁免雨天关闭。",
    "借阅期限最多十四天，不支持自动续借；续借必须另行申请，周末不受理。",
]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    vector._model = object()
    embeddings = np.zeros((2, 384), dtype=np.float32)
    embeddings[0, 0] = 1
    embeddings[1, :2] = [0.6, 0.8]
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embeddings)
    monkeypatch.setattr(vector, "_get_embedding", Mock(return_value=embeddings[0]))
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: vector)
    base = dict(documentId=901, doc_title="虚构档案馆规则", doc_category="制度", doc_kb_id=73)
    documents = [
        _vector_chunk_document(dict(base, chunkIndex=i, content=text), {73: "虚构知识库"})
        for i, text in enumerate(TEXTS)
    ]
    vector.add_documents(documents)
    helper = rag_helper.RAGHelper()
    return SimpleNamespace(vector=vector, documents=documents, helper=helper)


def test_same_document_chunks_remain_distinct_with_complete_negative_conditions(rig):
    rows = rig.helper.retrieve_context(QUERY, top_k=2, filters={"knowledge_base_id": 73}, enable_rerank=False)
    assert {row["id"] for row in rows} == {"doc_901_chunk_0", "doc_901_chunk_1"}
    assert {row["document_id"] for row in rows} == {901}
    assert all(any(text in row["content"] for row in rows) for text in TEXTS)
    assert all(row["knowledge_base_id"] == 73 for row in rows)
    cached = rig.helper.retrieve_context(QUERY, top_k=2, filters={"knowledge_base_id": 73}, enable_rerank=False)
    assert cached == rows and cached is not rows


def test_multiple_task_views_merge_same_chunk_only_and_retain_both_fact_groups(rig):
    rows = rig.helper.retrieve_context(
        QUERY, top_k=2, additional_queries=VIEWS, filters={"knowledge_base_id": 73}, enable_rerank=False
    )
    assert len(rows) == 2 and {row["id"] for row in rows} == {"doc_901_chunk_0", "doc_901_chunk_1"}
    assert all(row["retrieval_tasks"] == (0, 1, 2) for row in rows)
    assert all(any(text in row["content"] for row in rows) for text in TEXTS)


def test_native_index_rejects_missing_identity_before_embedding_or_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "invalid"), IndexConfig(index_type="flat"))
    encode = Mock(side_effect=AssertionError("Invalid source must not reach embedding"))
    monkeypatch.setattr(vector, "_get_embeddings_batch", encode)
    before = vector.snapshot_path.read_bytes() if vector.snapshot_path.exists() else None
    with pytest.raises(KeyError, match="id"):
        vector.add_documents([{"title": "虚构档案馆规则", "content": TEXTS[0]}])
    encode.assert_not_called()
    assert vector.index.ntotal == 0 and vector.metadata == []
    after = vector.snapshot_path.read_bytes() if vector.snapshot_path.exists() else None
    assert after == before


@pytest.mark.parametrize("legacy", [{"chunk_id": "legacy"}, {"document_id": 901}, {}])
def test_malformed_result_cannot_get_a_substitute_identity_or_success_cache(monkeypatch, legacy):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    row = dict(legacy, title="虚构档案馆规则", content=TEXTS[0], score=0.8, fused_score=0.8)
    vector = SimpleNamespace(cache_generation=1, hybrid_search=Mock(return_value=[row]))
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: vector)
    helper = rag_helper.RAGHelper()
    with pytest.raises(KeyError, match="id"):
        helper.retrieve_context(QUERY, enable_rerank=False)
    assert not helper._query_cache


@pytest.mark.parametrize("field", ["id", "title", "content"])
def test_citation_requires_source_fields_without_aliases_or_empty_defaults(field):
    row = dict(id="source", title="虚构档案馆规程", content=TEXTS[0], score=.8,
               chunk_id="legacy-alias", original_title="legacy-title")
    del row[field]
    with pytest.raises(KeyError, match=field):
        object.__new__(rag_helper.RAGHelper).build_citations([row])


@pytest.mark.parametrize("scores, expected", [({"fused_score": .6}, .6), ({"score": 1.7}, 1.)])
def test_raw_producer_citation_uses_existing_absolute_score_projection(scores, expected):
    row = dict(id="source", title="虚构档案馆规程", content=TEXTS[0], **scores)
    helper = object.__new__(rag_helper.RAGHelper)
    assert helper.build_citations([row])[0]["score"] == helper.compute_confidence([row]) == expected


def test_missing_citation_score_is_an_error_instead_of_a_fabricated_zero():
    row = dict(id="source", title="虚构档案馆规程", content=TEXTS[0])
    with pytest.raises(ValueError, match="RAG retrieval score must be a finite number"):
        object.__new__(rag_helper.RAGHelper).build_citations([row])
