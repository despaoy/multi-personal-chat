"""Invalid identities fail before encoding or changing an existing index."""
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize('damage',['missing','invalid','duplicate','existing'])
def test_invalid_batch_preserves_published_index(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(VectorDatabase,'_check_gpu_availability',lambda self:False)
    vector=VectorDatabase(str(tmp_path/'vectors'),IndexConfig(index_type='flat'))
    vector._model=object()
    monkeypatch.setattr(vector,'_get_embeddings_batch',lambda texts:np.tile(np.eye(1,384,dtype=np.float32),(len(texts),1)))
    vector.add_documents([dict(id=7,title='permit',content='permit requires written approval')])
    snapshot=vector.snapshot_path.read_bytes()
    metadata=deepcopy(vector.metadata)
    mapping=dict(vector._id_to_index)
    generation=vector.cache_generation
    cached=[dict(id=7,score=1)]
    vector._set_query_cache('complete-query',cached)
    docs=[dict(id=8,title='new permit',content='new permit still requires written approval')]
    if damage=='missing':
        del docs[0]['id']
    elif damage=='invalid':
        docs[0]['id']=None
    elif damage=='duplicate':
        docs.append(dict(docs[0],id='8'))
    else:
        docs[0]['id']='7'
    load=Mock(side_effect=AssertionError('invalid identity must not load the model'))
    encode=Mock(side_effect=AssertionError('invalid identity must not encode'))
    monkeypatch.setattr(vector,'_load_model',load)
    monkeypatch.setattr(vector,'_get_embeddings_batch',encode)
    with pytest.raises((KeyError,ValueError)):
        vector.add_documents(docs)
    load.assert_not_called()
    encode.assert_not_called()
    assert vector.metadata==metadata and vector._id_to_index==mapping
    assert vector.index.ntotal==1 and vector.bm25.corpus==['permit permit requires written approval']
    assert vector.cache_generation==generation and vector._get_cached_query('complete-query')==cached
    assert not vector._dirty and vector.snapshot_path.read_bytes()==snapshot
