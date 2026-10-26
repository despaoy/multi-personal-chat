"""Local inference failures propagate without changing device or replaying work."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.retrieval_core import embedding
from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize('caller',['shared','vector'])
@pytest.mark.parametrize('message',['CUDA out of memory','invalid model configuration'])
def test_model_load_failure_never_switches_device(monkeypatch,caller,message):
    error=RuntimeError(message)
    constructor=Mock(side_effect=error)
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=constructor))
    provider=embedding.SentenceTransformerEmbeddingProvider(model_path='fixture',device='cuda')
    if caller=='shared':
        load=provider._load_model
    else:
        monkeypatch.setattr(embedding,'_default_embedding_provider',provider)
        database=VectorDatabase.__new__(VectorDatabase)
        database.config=IndexConfig()
        load=lambda:database._get_embeddings_batch(['complete synthetic source'])
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

@pytest.mark.parametrize("error", [ValueError("invalid input"), TimeoutError("encoding deadline")])
def test_encoding_failure_does_not_repeat_or_change_the_error(monkeypatch, error):
    provider = embedding.SentenceTransformerEmbeddingProvider(model_path="fixture")
    provider._model = object()
    encode = Mock(side_effect=error)
    monkeypatch.setattr(embedding, "sentence_transformer_embeddings", encode)
    with pytest.raises(type(error)) as caught:
        provider.embed_texts(["complete synthetic input"])
    assert caught.value is error and encode.call_count == 1


def test_successful_batches_and_empty_input_keep_contract(monkeypatch):
    provider = embedding.SentenceTransformerEmbeddingProvider(model_path="fixture", expected_dim=2, batch_size=1)
    provider._model = object()
    matrix = np.eye(2, dtype=np.float32)
    encode = Mock(return_value=matrix)
    monkeypatch.setattr(embedding, "sentence_transformer_embeddings", encode)
    assert provider.embed_texts([]).shape == (0, 2)
    encode.assert_not_called()
    assert provider.embed_texts(["first", "second"]) is matrix
    encode.assert_called_once_with(provider._model, ["first", "second"], batch_size=1,
                                   dimension=2, timeout_seconds=300.0)
