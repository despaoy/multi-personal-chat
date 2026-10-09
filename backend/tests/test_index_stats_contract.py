"""Statistics expose the actual FAISS type, including reloaded snapshots."""
import pytest

from api import knowledge
from app import config
from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize('kind',['flat','ivf','hnsw'])
async def test_statistics_preserve_native_index_identity_after_reload(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(VectorDatabase,'_check_gpu_availability',lambda self:False)
    monkeypatch.setattr(knowledge,'VECTOR_DB_AVAILABLE',True)
    vector=VectorDatabase(str(tmp_path/'index'),IndexConfig(index_type=kind))
    assert vector.get_stats()['index_type']==kind
    vector._save_index()
    reloaded=VectorDatabase(str(tmp_path/'index'))
    monkeypatch.setattr(config,'get_vector_db',lambda:reloaded)
    response=await knowledge.get_vector_stats(dict(role='admin'))
    assert response['success'] and response['stats']['index_type']==kind
    assert response['stats']['total_documents']==response['stats']['index_size']==response['stats']['bm25_corpus_size']==0
