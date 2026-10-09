"""A long-lived character encoder keeps bounded, isolated query vectors."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider


def test_query_cache_evicts_old_entries_and_returns_copies(monkeypatch):
    provider=LocalMeanPoolingEmbeddingProvider(model_path='synthetic-fixed-model')
    encode=Mock(side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32))
    monkeypatch.setattr(provider,'embed_texts',encode)
    for i in range(256):
        provider.embed_query(f'complete synthetic query {i}')
    result=provider.embed_query('complete synthetic query 0')
    result[0]=99
    provider.embed_query('complete synthetic query 256')
    assert provider.embed_query('complete synthetic query 0')[0]==1
    assert encode.call_count==257
    provider.embed_query('complete synthetic query 1')
    assert encode.call_count==258


def test_failed_encoding_does_not_cache_or_replace_success(monkeypatch):
    provider=LocalMeanPoolingEmbeddingProvider(model_path='synthetic-fixed-model')
    error=OSError('synthetic encode failure')
    encode=Mock(side_effect=[np.ones((1,384),dtype=np.float32),error,np.zeros((1,384),dtype=np.float32)])
    monkeypatch.setattr(provider,'embed_texts',encode)
    provider.embed_query('existing complete query')
    with pytest.raises(OSError) as caught:
        provider.embed_query('new complete query')
    assert caught.value is error
    assert provider.embed_query('existing complete query')[0]==1
    assert provider.embed_query('new complete query')[0]==0
    assert encode.call_count==3


def test_concurrent_identical_queries_encode_once(monkeypatch):
    provider=LocalMeanPoolingEmbeddingProvider(model_path='synthetic-fixed-model')
    start=Barrier(8)
    def encode(texts):
        sleep(.03)  # Simulate an in-flight encoder while peer requests arrive.
        return np.ones((1,384),dtype=np.float32)
    encoder=Mock(side_effect=encode)
    monkeypatch.setattr(provider,'embed_texts',encoder)
    def query(_):
        start.wait(timeout=3)
        return provider.embed_query('same complete synthetic query')
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(query,range(8)))
    assert encoder.call_count==1
    results[0][0]=99
    assert all(result[0]==1 for result in results[1:])
