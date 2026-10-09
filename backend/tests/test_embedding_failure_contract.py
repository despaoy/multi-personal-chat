"""Local inference failures propagate without changing device or replaying work."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.retrieval_core import embedding
from knowledge.vector_db import VectorDatabase


@pytest.mark.parametrize('caller',['shared','vector'])
@pytest.mark.parametrize('message',['CUDA out of memory','invalid model configuration'])
def test_model_load_failure_never_switches_device(monkeypatch,caller,message):
    error=RuntimeError(message)
    constructor=Mock(side_effect=error)
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=constructor))
    if caller=='shared':
        provider=embedding.SentenceTransformerEmbeddingProvider(model_path='fixture',device='cuda')
        load=provider._load_model
    else:
        monkeypatch.setattr('knowledge.vector_db.resolve_local_model_path',lambda:'fixture')
        provider=VectorDatabase.__new__(VectorDatabase)
        provider._model=None
        provider._use_gpu=True
        load=provider._load_model
    with pytest.raises(RuntimeError) as caught:
        load()
    assert caught.value is error
    constructor.assert_called_once_with('fixture',device='cuda')
    assert provider._model is None

@pytest.mark.parametrize('available',[False,True])
def test_auto_device_selection_still_follows_availability(monkeypatch,available):
    monkeypatch.setattr(embedding,'_cuda_available',lambda:available)
    constructor=Mock()
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=constructor))
    provider=embedding.SentenceTransformerEmbeddingProvider(model_path='fixture')
    assert provider._load_model() is constructor.return_value
    assert provider._load_model() is constructor.return_value
    constructor.assert_called_once_with('fixture',device='cuda' if available else 'cpu')

@pytest.mark.parametrize('caller',['shared','vector'])
def test_device_probe_failure_is_not_cpu_availability(monkeypatch,caller):
    error=RuntimeError('device probe failed')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(side_effect=error))))
    with pytest.raises(RuntimeError) as caught:
        if caller=='shared':
            embedding._cuda_available()
        else:
            VectorDatabase.__new__(VectorDatabase)._check_gpu_availability()
    assert caught.value is error

@pytest.mark.parametrize('failure',['encode','dimension','timeout'])
def test_encoding_failure_does_not_repeat_batches(monkeypatch,failure):
    provider=embedding.SentenceTransformerEmbeddingProvider(model_path='fixture',expected_dim=2,batch_size=1,timeout_seconds=1)
    encode=Mock(return_value=np.ones((1,3 if failure=='dimension' else 2),dtype=np.float32))
    provider._model=SimpleNamespace(encode=encode)
    if failure=='encode':
        encode.side_effect=ValueError('invalid input')
    if failure=='timeout':
        times=iter([0,2])
        monkeypatch.setattr(embedding.time,'monotonic',lambda:next(times))
    expected={'encode':ValueError,'dimension':embedding.EmbeddingModelError,'timeout':TimeoutError}[failure]
    with pytest.raises(expected):
        provider.embed_texts(['complete synthetic input'])
    assert encode.call_count==1

def test_successful_batches_and_empty_input_keep_contract():
    provider=embedding.SentenceTransformerEmbeddingProvider(model_path='fixture',expected_dim=2,batch_size=1)
    encode=Mock(side_effect=[np.array([[1.,0.]]),np.array([[0.,1.]])])
    provider._model=SimpleNamespace(encode=encode)
    assert provider.embed_texts([]).shape==(0,2)
    encode.assert_not_called()
    np.testing.assert_array_equal(provider.embed_texts(['first','second']),np.eye(2,dtype=np.float32))
    assert encode.call_count==2
