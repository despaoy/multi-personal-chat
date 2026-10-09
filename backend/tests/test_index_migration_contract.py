"""Migration compares native index identity and preserves rollback state."""
import hashlib
from unittest.mock import Mock

import faiss
import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.fixture
def vector(tmp_path,monkeypatch):
    old_threads=faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    monkeypatch.setattr(VectorDatabase,'_check_gpu_availability',lambda self:False)
    v=VectorDatabase(str(tmp_path/'vectors'),IndexConfig(index_type='flat',nlist=2,nprobe=2,auto_switch_threshold=256))
    v._model=object()
    monkeypatch.setattr(v,'_get_embeddings_batch',lambda texts:np.tile(np.eye(1,384,dtype=np.float32),(len(texts),1)))
    v.add_documents([dict(id=i,title='Synthetic',content=f'完整合成规则编号{i}') for i in range(255)])
    yield v
    faiss.omp_set_num_threads(old_threads)

@pytest.mark.parametrize('kind',['flat','auto'])
def test_add_crossing_threshold_respects_explicit_or_auto_choice(vector,kind):
    vector.config.index_type=kind
    vector.add_documents([dict(id=255,title='Synthetic',content='完整合成规则编号255')])
    native=faiss.downcast_index(vector.index.index)
    assert isinstance(native,faiss.IndexFlatIP if kind=='flat' else faiss.IndexIVFFlat)
    if kind=='auto':
        assert native.nlist==2 and native.nprobe==2
    assert vector.index.ntotal==len(vector.metadata)==256
    loaded=VectorDatabase(str(vector.db_path))
    assert loaded.metadata==vector.metadata and loaded.index.ntotal==256

def test_hnsw_migration_preserves_real_index_and_snapshot(vector):
    before=list(vector.metadata)
    vector._migrate_index('hnsw')
    assert isinstance(faiss.downcast_index(vector.index.index),faiss.IndexHNSWFlat)
    assert vector.metadata==before and vector.index.ntotal==255
    assert VectorDatabase(str(vector.db_path)).metadata==before

@pytest.mark.parametrize('stage',['create','encode','save'])
def test_failed_migration_restores_existing_index_without_false_success(vector,monkeypatch,stage):
    before=(vector.index,vector.metadata,vector._id_to_index)
    digest=hashlib.sha256(vector.snapshot_path.read_bytes()).digest()
    error=OSError('synthetic migration failure')
    method={'create':'_create_index','encode':'_get_embeddings_batch','save':'_save_index'}[stage]
    monkeypatch.setattr(vector,method,Mock(side_effect=error))
    with pytest.raises(OSError) as caught:
        vector._migrate_index('ivf')
    assert caught.value is error
    assert vector.index is before[0] and vector.metadata is before[1] and vector._id_to_index is before[2]
    assert hashlib.sha256(vector.snapshot_path.read_bytes()).digest()==digest

def test_failed_auto_migration_still_invalidates_cache_of_newly_committed_docs(vector,monkeypatch):
    vector.config.index_type='auto'
    generation=vector.cache_generation
    monkeypatch.setattr(vector,'_migrate_index',Mock(side_effect=RuntimeError('synthetic migration failure')))
    with pytest.raises(RuntimeError):
        vector.add_documents([dict(id=255,title='Synthetic',content='完整合成规则编号255')])
    assert vector.cache_generation>generation
    assert vector.index.ntotal==256
