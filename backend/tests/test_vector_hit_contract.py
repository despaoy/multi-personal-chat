"""Invalid vector hit identity must not be repaired by matching its text."""
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize('damage',['missing_id','unknown_id','missing_score'])
def test_malformed_vector_hit_fails_before_keyword_fusion(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(VectorDatabase,'_check_gpu_availability',lambda self:False)
    vector=VectorDatabase(str(tmp_path/'vectors'),IndexConfig(index_type='flat'))
    vector._model=object()
    monkeypatch.setattr(vector,'_get_embeddings_batch',lambda texts:np.tile(np.eye(1,384,dtype=np.float32),(len(texts),1)))
    vector.add_documents([dict(id='permit-current',title='permit',content='permit requires written approval',category='current')])
    hit=dict(vector.metadata[0],score=.9)
    if damage=='missing_id':
        del hit['id']
    elif damage=='unknown_id':
        hit['id']='permit-unknown'
    else:
        del hit['score']
    monkeypatch.setattr(vector,'search',Mock(return_value=[hit]))
    keyword=Mock(side_effect=AssertionError('malformed vector hits must fail before keyword fusion'))
    monkeypatch.setattr(vector.bm25,'search',keyword)
    with pytest.raises(KeyError):
        vector.hybrid_search('permit',top_k=1)
    keyword.assert_not_called()
