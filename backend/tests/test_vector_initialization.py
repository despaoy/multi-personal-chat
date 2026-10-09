"""A failed model initialization cannot become the next request's cached model."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider


@pytest.mark.parametrize('stage',['tokenizer','model','eval','device'])
def test_failed_initialization_is_not_published_and_next_request_reloads(monkeypatch,stage):
    error=RuntimeError('synthetic initialization failure')
    first_tokenizer,second_tokenizer=object(),object()
    first_model,second_model=Mock(),Mock()
    tokenizers=Mock(side_effect=[first_tokenizer,second_tokenizer])
    models=Mock(side_effect=[first_model,second_model])
    if stage=='tokenizer':
        tokenizers.side_effect=[error,second_tokenizer]
        models.side_effect=[second_model]
    elif stage=='model':
        models.side_effect=[error,second_model]
    elif stage=='eval':
        first_model.eval.side_effect=error
    else:
        first_model.to.side_effect=error
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizers),
        AutoModel=SimpleNamespace(from_pretrained=models)))
    provider=LocalMeanPoolingEmbeddingProvider(model_path='synthetic-local-model')
    with pytest.raises(RuntimeError) as caught:
        provider._load()
    assert caught.value is error
    assert provider._tokenizer is provider._model is None
    assert tokenizers.call_count==1
    assert models.call_count==(stage!='tokenizer')
    assert provider._load()==(second_tokenizer,second_model)
    second_model.eval.assert_called_once_with()
    second_model.to.assert_called_once_with('cpu')
    assert provider._load()==(second_tokenizer,second_model)
    assert tokenizers.call_count==2
    assert models.call_count==(1 if stage=='tokenizer' else 2)
    for factory in (tokenizers,models):
        for call in factory.call_args_list:
            assert call.args==('synthetic-local-model',)
            assert call.kwargs=={'local_files_only':True}
