import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge.retrieval_core import embedding


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'cuda:1', 'mps'])
def test_configured_device_skips_automatic_gpu_probe(monkeypatch, device):
    monkeypatch.setenv('EMBEDDING_DEVICE', device)
    probe = Mock(side_effect=AssertionError('must not probe CUDA'))
    monkeypatch.setattr(embedding, '_cuda_available', probe)
    constructor = Mock()
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    embedding.SentenceTransformerEmbeddingProvider(model_path='fixture')._load_model()
    constructor.assert_called_once_with('fixture', device=device)
    probe.assert_not_called()


def test_explicit_device_overrides_environment(monkeypatch):
    monkeypatch.setenv('EMBEDDING_DEVICE', 'cuda')
    constructor = Mock()
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    embedding.SentenceTransformerEmbeddingProvider(model_path='fixture', device='cpu')._load_model()
    constructor.assert_called_once_with('fixture', device='cpu')


@pytest.mark.parametrize('device', ['', 'cpou', 'cuda:-1'])
def test_invalid_device_fails_before_model_load(monkeypatch, device):
    monkeypatch.setenv('EMBEDDING_DEVICE', device)
    constructor = Mock()
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    with pytest.raises(embedding.EmbeddingModelError, match='EMBEDDING_DEVICE'):
        embedding.SentenceTransformerEmbeddingProvider(model_path='fixture')._load_model()
    constructor.assert_not_called()
