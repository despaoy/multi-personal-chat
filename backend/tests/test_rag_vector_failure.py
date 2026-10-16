"""A required vector failure cannot become successful sparse-only retrieval."""
from unittest.mock import Mock

import numpy as np
import pytest
from test_multiscale_rag_runtime import FakeQueryEmbeddingProvider

from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime
from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider

QUERY = '妃和琉璃是什么关系？'


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv('RERANKER_ENABLED', 'false')
    provider = LocalMeanPoolingEmbeddingProvider(model_path='synthetic-fixed-model')
    monkeypatch.setattr(provider, 'embed_query', FakeQueryEmbeddingProvider().embed_query)
    monkeypatch.setattr('knowledge.multiscale_rag.runtime.LocalMeanPoolingEmbeddingProvider',
                        lambda **kwargs: provider)
    instance = MultiScaleRagRuntime()
    assert instance.is_available()
    return instance


def search(runtime, mode, **kwargs):
    service = runtime._service
    analysis = service.analyzer.analyze(QUERY, domain_id=service.config.domain_id)
    return service.retriever.search(analysis, top_k=3, mode=mode, **kwargs)


@pytest.mark.parametrize('mode', ['hybrid', 'vector', 'runtime'])
@pytest.mark.parametrize('error_type', [OSError, ValueError])
def test_required_vector_failure_propagates_without_partial_results(runtime, mode, error_type, caplog):
    failure = error_type('private-embedding-detail')
    runtime._provider.embed_query = Mock(side_effect=failure)
    with pytest.raises(error_type) as caught:
        if mode == 'runtime':
            runtime.retrieve_with_citations(QUERY, top_k=3)
        else:
            search(runtime, mode)
    assert caught.value is failure
    runtime._provider.embed_query.assert_called_once()
    assert 'private-embedding-detail' not in caplog.text


def test_explicit_sparse_mode_does_not_need_embedding(runtime):
    runtime._provider.embed_query = Mock(side_effect=AssertionError('unused channel'))
    assert search(runtime, 'sparse')
    runtime._provider.embed_query.assert_not_called()


def test_completed_vector_search_can_legitimately_have_no_matches(runtime):
    assert search(runtime, 'vector', vector_threshold=1.1) == []


def test_healthy_runtime_still_returns_evidence(runtime):
    result = runtime.retrieve_with_citations(QUERY, top_k=3)
    assert result['results'] and result['context_trust'] == 'untrusted_retrieved_evidence'


@pytest.mark.parametrize('mode', ['hybrid', 'vector', 'runtime'])
@pytest.mark.parametrize('kind', ['nan', 'inf', 'negative_inf', 'zero', 'overflow', 'wrong_dimension'])
def test_invalid_query_vector_cannot_become_successful_retrieval(runtime, mode, kind):
    value = {'nan': np.nan, 'inf': np.inf, 'negative_inf': -np.inf, 'zero': 0.0,
             'overflow': 1e30, 'wrong_dimension': 1.0}[kind]
    vector = np.full(768 if kind == 'wrong_dimension' else 384, value, dtype=np.float32)
    runtime._provider.embed_query = Mock(return_value=vector)
    with pytest.raises(ValueError, match='embedding (vector|dimension)'):
        if mode == 'runtime':
            runtime.retrieve_with_citations(QUERY, top_k=3)
        else:
            search(runtime, mode)
    runtime._provider.embed_query.assert_called_once_with(QUERY)


def test_vector_validation_preserves_inner_product_scores_and_threshold(runtime):
    index = runtime._service.retriever.index
    query = 2.0 * FakeQueryEmbeddingProvider().embed_query(QUERY)
    expected = np.sort(index._faiss_index.matrix @ query)[::-1][:3]
    result = index.search_vector(query, top_k=3)
    assert np.allclose([score for _, score in result], expected)
    threshold = expected[0] * 0.75
    assert index.search_vector(query, top_k=3, threshold=threshold)[0][1] == pytest.approx(expected[0])
