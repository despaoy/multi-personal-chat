"""Exercise actual pooling and cache publication with isolated model outputs."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider

QUERY = '妃和琉璃是什么关系？'


def tokenizer_control(attention):
    """Complete two-pass tokenization for the declared pooling-only fixture."""
    def tokenize(texts, **options):
        if options.get("add_special_tokens") is False:
            return {"input_ids": [[1] * int(row.sum()) for row in attention[:len(texts)]]}
        return dict(input_ids=torch.ones_like(attention), attention_mask=attention,
                    special_tokens_mask=torch.zeros_like(attention),
                    overflow_to_sample_mapping=torch.arange(len(texts)))

    tokenizer = Mock(side_effect=tokenize)
    tokenizer.num_special_tokens_to_add.return_value = 0
    return tokenizer


@pytest.mark.parametrize('kind', ['nan', 'inf', 'negative_inf', 'zero', 'overflow',
                                  'wrong_width', 'extra_rows', 'missing_rows'])
def test_bad_model_output_fails_before_caching_and_next_request_can_recover(monkeypatch, kind):
    provider = LocalMeanPoolingEmbeddingProvider(model_path='synthetic-fixed-model')
    value = {'nan': float('nan'), 'inf': float('inf'), 'negative_inf': -float('inf'),
             'zero': 0.0, 'overflow': 1e30}.get(kind, 1.0)
    shape = {'wrong_width': (1, 2, 12), 'extra_rows': (2, 2, 384),
             'missing_rows': (0, 2, 384)}.get(kind, (1, 2, 384))
    tokenizer = tokenizer_control(torch.ones((1, 2), dtype=torch.long))
    model = Mock(side_effect=[SimpleNamespace(last_hidden_state=torch.full(shape, value)),
                              SimpleNamespace(last_hidden_state=torch.ones((1, 2, 384)))])
    monkeypatch.setattr(provider, '_load', lambda: (tokenizer, model))
    with pytest.raises(ValueError, match='embedding (vector|shape)'):
        provider.embed_query(QUERY)
    assert QUERY not in provider._query_cache
    result = provider.embed_query(QUERY)
    assert np.isfinite(result).all() and np.linalg.norm(result) == pytest.approx(1.0)
    result[0] = 99.0
    assert provider.embed_query(QUERY)[0] == pytest.approx(1 / np.sqrt(384))
    assert model.call_count == 2 and tokenizer.call_count == 4


def test_healthy_batches_and_empty_input_preserve_encoder_contract(monkeypatch):
    provider = LocalMeanPoolingEmbeddingProvider(model_path='synthetic-fixed-model', batch_size=2)
    tokenizer = tokenizer_control(torch.tensor([[1, 0], [1, 1]]))
    hidden = torch.ones((2, 2, 384))
    hidden[:, 1] = 3.0
    model = Mock(return_value=SimpleNamespace(last_hidden_state=hidden))
    monkeypatch.setattr(provider, '_load', lambda: (tokenizer, model))
    assert provider.embed_texts([]).shape == (0, 384)
    assert not model.called
    result = provider.embed_texts([QUERY, '琉璃与妃的关系是什么？'] * 2)
    assert result.shape == (4, 384) and result.dtype == np.float32
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
    assert model.call_count == 2 and tokenizer.call_count == 4
