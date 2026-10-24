"""Required retrieval channels cannot silently disappear from a hybrid query."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from faiss import IndexFlatIP

from knowledge.multiscale_rag.vector_runtime import MatrixInnerProductIndex, attach_vectors
from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.errors import RetrievalIndexError
from knowledge.retrieval_core.index import BM25Index, DomainIndex
from knowledge.retrieval_core.query import QueryAnalysis
from knowledge.retrieval_core.registry import KnowledgeDomainConfig
from knowledge.retrieval_core.retrieval import HybridRetriever

QUERY = "青岚计划周五离线执行，出差时顺延。"


@pytest.fixture
def retrieval(tmp_path):
    index = DomainIndex("fixture", dimension=4)
    index.documents = [KnowledgeIndexDocument(
        id=f"fixture:{position}", domain_id="fixture", document_type="fact",
        title=body, summary=body, content=body, embedding_text=body,
    ) for position, body in enumerate([QUERY, "完成书面许可核对，尚未执行计划。"])]
    index._rebuild_entity_index()
    index.bm25.build(index.documents)
    attach_vectors(index, np.eye(2, 4, dtype=np.float32))
    provider = SimpleNamespace(embed_query=Mock(return_value=np.array([1, 0, 0, 0], dtype=np.float32)))
    config = KnowledgeDomainConfig("fixture", tmp_path, lambda _: [])
    retriever = HybridRetriever(config, index, provider)
    return retriever, QueryAnalysis(original_query=QUERY, normalized_query=QUERY), provider


@pytest.mark.parametrize("mode,corruption", [
    ("hybrid", "missing_vector"), ("vector", "missing_vector"),
    ("hybrid", "empty_vector"), ("hybrid", "partial_vector"),
    ("sparse", "missing_sparse"), ("hybrid", "missing_sparse"), ("hybrid", "partial_sparse"),
])
def test_required_index_failure_is_not_successful_partial_retrieval(retrieval, mode, corruption):
    retriever, analysis, _ = retrieval
    index = retriever.index
    if corruption == "missing_vector":
        index._faiss_index = None
    elif corruption == "empty_vector":
        index._faiss_index = IndexFlatIP(4)
    elif corruption == "partial_vector":
        index._faiss_index = MatrixInnerProductIndex(np.eye(1, 4, dtype=np.float32))
    elif corruption == "missing_sparse":
        index.bm25 = BM25Index()
    else:
        index.bm25.build(index.documents[:1])
    with pytest.raises(RetrievalIndexError) as caught:
        retriever.search(analysis, top_k=2, mode=mode)
    assert caught.value.channel == ("vector" if "vector" in corruption else "sparse")
    assert caught.value.reason == ("not_initialized" if corruption.startswith("missing_") else "row_count_mismatch")
    assert not index.is_healthy()


def test_unloaded_empty_domain_is_not_an_explicit_empty_index(retrieval):
    retriever, analysis, provider = retrieval
    retriever.index = DomainIndex("fixture", dimension=4)
    with pytest.raises(RuntimeError):
        retriever.search(analysis, top_k=2)
    provider.embed_query.assert_not_called()


def test_unknown_mode_is_not_implicitly_hybrid(retrieval):
    retriever, analysis, _ = retrieval
    with pytest.raises(ValueError, match="mode"):
        retriever.search(analysis, top_k=2, mode="synthetic-invalid-mode")


@pytest.mark.parametrize("mode", ["sparse", "vector", "hybrid"])
def test_healthy_channels_keep_the_real_document_and_native_score(retrieval, mode):
    retriever, analysis, provider = retrieval
    results = retriever.search(analysis, top_k=2, mode=mode)
    assert results and results[0].document.content == QUERY
    assert retriever.index.is_healthy()
    if mode == "sparse":
        provider.embed_query.assert_not_called()
    else:
        provider.embed_query.assert_called_once_with(QUERY)
    if mode == "vector":
        assert [row.vector_score for row in results] == [1.0, 0.0]


@pytest.mark.parametrize("mode", ["sparse", "vector", "hybrid"])
def test_initialized_empty_index_is_a_normal_empty_result(retrieval, mode):
    retriever, analysis, provider = retrieval
    index = DomainIndex("fixture", dimension=4)
    index.bm25.build([])
    attach_vectors(index, np.empty((0, 4), dtype=np.float32))
    retriever.index = index
    assert retriever.search(analysis, top_k=2, mode=mode) == []
    assert index.is_healthy()
    provider.embed_query.assert_not_called()


@pytest.mark.parametrize("mode", ["sparse", "vector"])
def test_explicit_single_channel_does_not_require_the_unused_channel(retrieval, mode):
    retriever, analysis, provider = retrieval
    if mode == "sparse":
        retriever.index._faiss_index = None
    else:
        retriever.index.bm25 = BM25Index()
    assert retriever.search(analysis, top_k=2, mode=mode)
    if mode == "sparse":
        provider.embed_query.assert_not_called()


def test_successful_search_can_have_no_matching_evidence(retrieval):
    retriever, _, _ = retrieval
    analysis = QueryAnalysis(original_query="unrelated_criterion_719", normalized_query="unrelated_criterion_719")
    assert retriever.search(analysis, top_k=2, mode="sparse") == []
    assert retriever.search(analysis, top_k=2, mode="vector", vector_threshold=1.1) == []


def test_standalone_sparse_initialization_and_empty_build_are_distinct():
    index = BM25Index()
    with pytest.raises(RuntimeError, match="not initialized"):
        index.search(QUERY, top_k=2)
    index.build([])
    assert index.search(QUERY, top_k=2) == []
    assert index.stats()["built"] is True


def test_zero_requested_results_do_not_start_embedding(retrieval):
    retriever, analysis, provider = retrieval
    assert retriever.search(analysis, top_k=0) == []
    provider.embed_query.assert_not_called()


def test_repaired_index_can_be_used_by_a_later_request(retrieval):
    retriever, analysis, _ = retrieval
    retriever.index._faiss_index = None
    with pytest.raises(RuntimeError, match="Vector index"):
        retriever.search(analysis, top_k=2)
    attach_vectors(retriever.index, np.eye(2, 4, dtype=np.float32))
    assert retriever.search(analysis, top_k=2)
    assert retriever.index.is_healthy()


@pytest.mark.parametrize("channel", ["sparse", "vector", "documents"])
async def test_generation_stops_when_a_required_index_is_missing(retrieval, semantic_plan, monkeypatch, channel):
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from api import generate
    from db.schemas import MessageRequest
    from knowledge.turn_dependencies import query_segments

    retriever, analysis, _ = retrieval
    if channel == "vector":
        retriever.index._faiss_index = None
    elif channel == "sparse":
        retriever.index.bm25 = BM25Index()
    else:
        monkeypatch.setattr(retriever.index.bm25, "search", lambda *a: [(2, 1.0)])
    semantic_plan(QUERY, dict(unresolved_source=list(range(len(query_segments(QUERY))))))

    async def retrieve(query, top_k, filters):
        rows = retriever.search(analysis, top_k=top_k)
        return dict(results=[row.to_dict() for row in rows], citations=[], confidence=.9,
                    abstained=False, context_text="\n".join(row.document.content for row in rows))

    lookup = AsyncMock(side_effect=retrieve)
    model = AsyncMock(side_effect=AssertionError("Required index failure must stop answer generation"))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", lookup)
    monkeypatch.setattr(generate, "_rag_retrieval_timeout", lambda: 30.0)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(MessageRequest(message=QUERY), None,
            runtime_config={"useKnowledgeBase": True}, model_generate=model)
    assert caught.value.status_code == 503
    assert isinstance(caught.value.__cause__, RetrievalIndexError)
    assert caught.value.detail == RetrievalIndexError(channel, "invalid_row" if channel == "documents" else "not_initialized").public_detail()
    assert QUERY not in str(caught.value.detail)
    lookup.assert_awaited_once()
    model.assert_not_awaited()


@pytest.mark.parametrize("channel", ["sparse", "vector", "entity"])
@pytest.mark.parametrize("row", [-1, 2])
def test_invalid_document_row_is_not_a_successful_empty_result(retrieval, monkeypatch, channel, row):
    retriever, analysis, _ = retrieval
    if channel == "sparse":
        monkeypatch.setattr(retriever.index, "search_sparse", lambda *a: [(row, 1.0)])
    elif channel == "vector":
        monkeypatch.setattr(retriever.index, "search_vector", lambda *a, **kw: [(row, 1.0)])
    else:
        analysis.entities = ["青岚"]
        retriever.index._entity_rows = {"青岚": [row]}
    with pytest.raises(RetrievalIndexError) as caught:
        retriever.search(analysis, top_k=2, mode=channel if channel != "entity" else "hybrid")
    assert caught.value.channel == "documents" and caught.value.reason == "invalid_row"
    assert QUERY not in str(caught.value) and "重建" in caught.value.public_detail()["message"]
