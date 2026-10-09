"""Any attempted index mutation invalidates cached reads before it can fail."""
import hashlib
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.fixture
def vector(tmp_path,monkeypatch):
    monkeypatch.setattr(VectorDatabase,'_check_gpu_availability',lambda self:False)
    v=VectorDatabase(str(tmp_path/'vectors'),IndexConfig(index_type='flat'))
    v._model=object()
    monkeypatch.setattr(v,'_get_embeddings_batch',lambda texts:np.tile(np.eye(1,384,dtype=np.float32),(len(texts),1)))
    v.add_documents([dict(id=1,title='完整旧规则',content='10月12日办理，需书面确认。')])
    v._set_query_cache('synthetic-cached-query',[dict(id=1)])
    return v

@pytest.mark.parametrize('stage',['vector','keyword','save'])
def test_failed_write_marks_dirty_and_discards_old_cache(vector,monkeypatch,stage):
    generation=vector.cache_generation
    snapshot=hashlib.sha256(vector.snapshot_path.read_bytes()).digest()
    error=OSError('synthetic write failure')
    target,method={'vector':(vector.index,'add_with_ids'),'keyword':(vector.bm25,'add_documents'),'save':(vector,'_save_index')}[stage]
    monkeypatch.setattr(target,method,Mock(side_effect=error))
    with pytest.raises(OSError) as caught:
        vector.add_documents([dict(id=2,title='完整新规则',content='10月19日办理，仍需书面确认。')])
    assert caught.value is error
    assert vector._dirty and vector.cache_generation==generation+1
    assert vector._get_cached_query('synthetic-cached-query') is None
    assert hashlib.sha256(vector.snapshot_path.read_bytes()).digest()==snapshot
    assert VectorDatabase(str(vector.db_path)).index.ntotal==1

@pytest.mark.parametrize('deferred',[False,True])
def test_successful_write_and_explicit_deferred_flush_preserve_cache_contract(vector,deferred):
    generation=vector.cache_generation
    if deferred:
        vector.config.save_on_every_n_adds=100
    vector.add_documents([dict(id=2,title='完整新规则',content='10月19日办理，仍需书面确认。')])
    assert vector._dirty is deferred
    assert vector.cache_generation==generation+1
    assert vector._get_cached_query('synthetic-cached-query') is None
    vector.flush()
    assert not vector._dirty and VectorDatabase(str(vector.db_path)).index.ntotal==2
