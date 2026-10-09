"""A required vector failure cannot become successful sparse-only retrieval."""
from unittest.mock import Mock

import pytest
from test_multiscale_rag_runtime import FakeQueryEmbeddingProvider

from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime
from knowledge.multiscale_rag.service import CARD_TYPES, analyze_explicit_domain

QUERY = '妃和琉璃是什么关系？'


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv('RERANKER_ENABLED', 'false')
    monkeypatch.setattr('knowledge.multiscale_rag.runtime.LocalMeanPoolingEmbeddingProvider',
                        lambda **kwargs: FakeQueryEmbeddingProvider())
    instance = MultiScaleRagRuntime()
    assert instance.is_available()
    return instance


def search(runtime, mode, **kwargs):
    service = runtime._service
    analysis = analyze_explicit_domain(service.analyzer, service.config.domain_id, QUERY)
    return service.retrievers[CARD_TYPES].search(analysis, top_k=3, mode=mode, **kwargs)


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
