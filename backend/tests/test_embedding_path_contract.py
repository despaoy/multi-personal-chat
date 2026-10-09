"""Embedding lookup must preserve requested model identity and explicit config."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider
from knowledge.retrieval_core import embedding
from knowledge.vector_db import VectorDatabase


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(Path,'home',classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(embedding,'_LOCAL_MODEL_SEARCH_PATHS',[tmp_path/'default'])
    for name in ('EMBEDDING_MODEL_PATH','ALLOW_REMOTE_EMBEDDING_MODEL','MULTIPERSONAL_LAB_ROOT','QQCHAT_LAB_ROOT'):
        monkeypatch.delenv(name,raising=False)
    return tmp_path

def model(path):
    path.mkdir(parents=True)
    (path/'config.json').write_text('{}')
    (path/'model.safetensors').touch()
    return str(path)

@pytest.mark.parametrize('kind',['missing','file','empty_directory'])
def test_explicit_invalid_path_cannot_fall_through(paths,monkeypatch,kind):
    model(paths/'default')
    target=paths/kind
    if kind=='file':
        target.touch()
    if kind=='empty_directory':
        target.mkdir()
    monkeypatch.setenv('EMBEDDING_MODEL_PATH',str(target))
    monkeypatch.setenv('ALLOW_REMOTE_EMBEDDING_MODEL','true')
    with pytest.raises(embedding.EmbeddingModelError,match='EMBEDDING_MODEL_PATH'):
        embedding.resolve_local_model_path()

def test_explicit_valid_path_has_priority(paths,monkeypatch):
    expected=model(paths/'explicit')
    model(paths/'default')
    monkeypatch.setenv('EMBEDDING_MODEL_PATH',expected)
    assert embedding.resolve_local_model_path()==expected

@pytest.mark.parametrize('model_id',['custom-model','other-org/custom-model'])
def test_custom_model_never_uses_default_cache(paths,model_id):
    model(paths/'default')
    model(paths/'.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/one')
    with pytest.raises(embedding.EmbeddingModelError):
        embedding.resolve_local_model_path(model_id)

@pytest.mark.parametrize('model_id',['custom-model','other-org/custom-model'])
def test_exact_requested_hub_cache(paths,model_id):
    identity=model_id if '/' in model_id else 'sentence-transformers/'+model_id
    expected=model(paths/'.cache/huggingface/hub'/('models--'+identity.replace('/','--'))/'snapshots/one')
    assert embedding.resolve_local_model_path(model_id)==expected

def test_remote_opt_in_returns_requested_model(paths,monkeypatch):
    monkeypatch.setenv('ALLOW_REMOTE_EMBEDDING_MODEL','true')
    assert embedding.resolve_local_model_path('other-org/model')=='other-org/model'

@pytest.mark.parametrize('root_name',['MULTIPERSONAL_LAB_ROOT','QQCHAT_LAB_ROOT'])
def test_existing_lab_default_location_is_preserved(paths,monkeypatch,root_name):
    expected=model(paths/'lab/models'/embedding.DEFAULT_EMBEDDING_MODEL_ID)
    monkeypatch.setenv(root_name,str(paths/'lab'))
    assert embedding.resolve_local_model_path()==expected

def test_both_vector_callers_use_same_explicit_path(paths,monkeypatch):
    expected=model(paths/'explicit')
    monkeypatch.setenv('EMBEDDING_MODEL_PATH',expected)
    constructor=Mock(return_value=object())
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=constructor))
    database=VectorDatabase.__new__(VectorDatabase)
    database._model=None
    database._use_gpu=False
    database._load_model()
    constructor.assert_called_once_with(expected,device='cpu')
    assert LocalMeanPoolingEmbeddingProvider().model_path==expected
