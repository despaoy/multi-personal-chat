"""Required generic retrieval work must complete before results can be cached."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge import rag_helper

QUERY = '档案馆周一开门时间和借阅期限'
ROWS = [
    dict(id='opening', content='档案馆周一九点开门。', score=.8),
    dict(id='loan', content='借阅期限为十四天。', score=.7),
]


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setenv('RERANKER_ENABLED', 'false')
    index = SimpleNamespace(cache_generation=1, hybrid_search=Mock(return_value=ROWS))
    factory = Mock(return_value=index)
    monkeypatch.setattr(rag_helper, 'get_vector_db', factory)
    helper = rag_helper.RAGHelper()
    helper.enable_query_expansion = False
    return helper, index, factory


def test_enabled_reranker_initialization_failure_is_not_disabled(monkeypatch):
    monkeypatch.setenv('RERANKER_ENABLED', 'true')
    error = OSError('private-initialization-detail')
    monkeypatch.setattr(rag_helper, 'get_reranker', Mock(side_effect=error))
    with pytest.raises(OSError) as caught:
        rag_helper.RAGHelper()
    assert caught.value is error


@pytest.mark.parametrize('stage', ['index', 'search', 'rerank'])
@pytest.mark.parametrize('with_citations', [False, True])
def test_required_failure_propagates_without_results_or_cache(prepared, stage, with_citations):
    helper, index, factory = prepared
    error = OSError('private-retrieval-detail')
    operation = Mock(side_effect=error)
    if stage == 'index':
        factory.side_effect = error
    elif stage == 'search':
        index.hybrid_search = operation
    else:
        helper.enable_reranking = True
        helper.reranker = SimpleNamespace(rerank=operation)
    invoke = helper.retrieve_with_citations if with_citations else helper.retrieve_context
    with pytest.raises(OSError) as caught:
        invoke(QUERY)
    assert caught.value is error
    assert not helper._query_cache


def test_later_expanded_query_failure_cannot_return_partial_evidence(prepared):
    helper, index, _ = prepared
    helper.enable_query_expansion = True
    helper.query_expander.expand_query = lambda q: [q, '档案馆借阅期限']
    error = TimeoutError('second search failed')
    index.hybrid_search.side_effect = [ROWS[:1], error]
    with pytest.raises(TimeoutError) as caught:
        helper.retrieve_with_citations(QUERY)
    assert caught.value is error and not helper._query_cache


def test_legitimate_empty_search_keeps_abstention(prepared):
    helper, index, _ = prepared
    index.hybrid_search.return_value = []
    result = helper.retrieve_with_citations(QUERY)
    assert result['results'] == [] and result['citations'] == [] and result['abstained']


def test_empty_reranker_output_is_failure(prepared):
    helper, _, _ = prepared
    helper.enable_reranking = True
    helper.reranker = SimpleNamespace(rerank=Mock(return_value=[]))
    with pytest.raises(RuntimeError, match='non-empty candidates'):
        helper.retrieve_context(QUERY)
    assert not helper._query_cache


def test_recovery_does_not_reuse_failed_partial_result(prepared):
    helper, index, _ = prepared
    helper.enable_reranking = True
    helper.reranker = SimpleNamespace(rerank=Mock(side_effect=TimeoutError('failure')))
    with pytest.raises(TimeoutError):
        helper.retrieve_context(QUERY)
    helper.reranker.rerank.side_effect = lambda q, rows, top_k: [dict(row, rerank_score=1.0) for row in rows[:top_k]]
    assert len(helper.retrieve_context(QUERY)) == 2
    assert index.hybrid_search.call_count == 2


def test_actual_reranking_mode_separates_cache_and_preserves_scores(prepared):
    helper, index, _ = prepared
    assert helper.retrieve_context(QUERY)[0]['id'] == 'opening'
    helper.enable_reranking = True
    helper.reranker = SimpleNamespace(rerank=Mock(side_effect=lambda q, rows, top_k: [
        dict(rows[1], rerank_score=2.0), dict(rows[0], rerank_score=-1.0),
    ]))
    result = helper.retrieve_context(QUERY)
    assert [r['id'] for r in result] == ['loan', 'opening']
    assert [r['normalized_score'] for r in result] == [1.0, 0.0]
    assert index.hybrid_search.call_count == 2
    result[0]['content'] = 'modified caller copy'
    cached = helper.retrieve_context(QUERY)
    assert cached[0]['content'] == ROWS[1]['content']
    helper.reranker.rerank.assert_called_once()
    assert all('normalized_score' not in r and 'final_score' not in r for r in ROWS)


def test_explicit_no_reranking_does_not_call_required_encoder(prepared):
    helper, _, _ = prepared
    helper.enable_reranking = True
    helper.reranker = SimpleNamespace(rerank=Mock(side_effect=AssertionError('disabled')))
    assert len(helper.retrieve_context(QUERY, enable_rerank=False)) == 2
    helper.reranker.rerank.assert_not_called()


def test_single_candidate_still_uses_enabled_reranker(prepared):
    helper, index, factory = prepared
    index.hybrid_search.return_value = ROWS[:1]
    helper.enable_reranking = True
    helper.reranker = SimpleNamespace(rerank=Mock(side_effect=lambda q, rows, top_k: [dict(rows[0], rerank_score=.8)]))
    assert helper.retrieve_context(QUERY)[0]['rerank_score'] == .8
    helper.reranker.rerank.assert_called_once()
    factory.assert_called_once()
