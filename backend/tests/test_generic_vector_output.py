"""Bad generic model output cannot publish a successful retrieval or index write."""
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge import rag_helper
from knowledge.vector_db import IndexConfig, VectorDatabase

QUERY = '档案馆周一开门时间和借阅期限'
DOCUMENT = dict(id='archive-rule', title='档案馆规定', content='档案馆周一九点开门，借阅期限为十四天。')


def healthy(texts, **kwargs):
    return np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv('RERANKER_ENABLED', 'false')
    monkeypatch.setattr(VectorDatabase, '_check_gpu_availability', lambda self: False)
    vector = VectorDatabase(str(tmp_path / 'vectors'), IndexConfig(index_type='flat', batch_encode_size=1))
    encode = Mock(side_effect=healthy)
    vector._model = SimpleNamespace(encode=encode)
    vector.add_documents([dict(DOCUMENT)])
    monkeypatch.setattr(rag_helper, 'get_vector_db', lambda: vector)
    helper = rag_helper.RAGHelper()
    encode.reset_mock()
    return vector, encode, helper


def invalid(kind):
    value = {'nan': np.nan, 'inf': np.inf, 'negative_inf': -np.inf, 'zero': 0.0, 'overflow': 1e30}.get(kind, 1.0)
    shape = {'wrong_width': (1, 12), 'extra_rows': (2, 384), 'missing_rows': (0, 384)}.get(kind, (1, 384))
    return np.full(shape, value, dtype=np.float32)


@pytest.mark.parametrize('entry', ['vector', 'hybrid', 'context', 'citations'])
@pytest.mark.parametrize('kind', ['nan', 'inf', 'negative_inf', 'zero', 'overflow',
                                  'wrong_width', 'extra_rows', 'missing_rows'])
def test_bad_model_output_fails_before_keyword_fusion_or_caching(prepared, entry, kind):
    vector, encode, helper = prepared
    encode.side_effect = lambda *args, **kwargs: invalid(kind)
    invoke = {'vector': vector.search, 'hybrid': vector.hybrid_search,
              'context': helper.retrieve_context, 'citations': helper.retrieve_with_citations}[entry]
    with pytest.raises(ValueError, match='embedding (vector|shape)'):
        invoke(QUERY)
    assert encode.call_count == 1
    assert not vector._query_cache and not helper._query_cache
    encode.side_effect = healthy
    result = invoke(QUERY)
    assert result['results'] if entry == 'citations' else result
    assert encode.call_count == 2
    invoke(QUERY)
    assert encode.call_count == 2


@pytest.mark.parametrize('kind', ['zero', 'extra_rows'])
def test_later_batch_failure_cannot_publish_partial_index_or_change_existing_cache(prepared, kind):
    vector, encode, _ = prepared
    vector.search(QUERY)
    before = hashlib.sha256(vector.snapshot_path.read_bytes()).digest()
    generation = vector.cache_generation
    cached = list(vector._query_cache)
    encode.reset_mock(side_effect=True)
    encode.side_effect = [healthy(['first']), invalid(kind)]
    documents = [dict(id='first-new', title='新规定', content='周二十点开门。'),
                 dict(id='second-new', title='新规定', content='周三十一点开门。')]
    with pytest.raises(ValueError, match='embedding (vector|shape)'):
        vector.add_documents(documents)
    assert encode.call_count == 2
    assert vector.index.ntotal == len(vector.metadata) == len(vector._id_to_index) == 1
    assert len(vector.bm25.corpus) == 1 and not vector._dirty
    assert vector.cache_generation == generation and list(vector._query_cache) == cached
    assert hashlib.sha256(vector.snapshot_path.read_bytes()).digest() == before
    encode.side_effect = healthy
    vector.add_documents(documents)
    assert vector.index.ntotal == len(vector.metadata) == 3
    assert vector.cache_generation == generation + 1 and not vector._query_cache


def test_batch_boundaries_are_validated_before_next_encode(prepared):
    vector, encode, _ = prepared
    vector.config.batch_encode_size = 2
    # A missing row in one batch cannot be hidden by an extra row in the next.
    encode.side_effect = [healthy(['one']), healthy(['two', 'three', 'four'])]
    with pytest.raises(ValueError, match='embedding shape'):
        vector._get_embeddings_batch(['one', 'two', 'three', 'four'])
    assert encode.call_count == 1


def test_healthy_batches_preserve_values_dtype_and_single_loading_boundary(prepared, monkeypatch):
    vector, encode, _ = prepared
    load = Mock(wraps=vector._load_model)
    monkeypatch.setattr(vector, '_load_model', load)
    encode.side_effect = lambda texts, **kwargs: 2.0 * healthy(texts).astype(np.float64)
    vector.add_documents([dict(id='next-rule', title='次日规定', content='周二十点开门。')])
    assert load.call_count == 1
    rows = vector._get_embeddings_batch(['first', 'second'])
    assert rows.shape == (2, 384) and rows.dtype == np.float32
    assert np.array_equal(rows, 2.0 * healthy(['first', 'second']))
    assert encode.call_count == 3


def test_empty_library_and_legitimate_threshold_miss_do_not_fake_failure(prepared):
    vector, encode, _ = prepared
    assert vector.search(QUERY, threshold=1.1) == []
    assert encode.call_count == 1
    vector.clear_all()
    encode.side_effect = AssertionError('empty index must not call encoder')
    assert vector.search(QUERY) == vector.hybrid_search(QUERY) == []
    vector.add_documents([])
    assert encode.call_count == 1
