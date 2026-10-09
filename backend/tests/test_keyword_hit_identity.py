"""BM25 positions identify distinct sources with identical text."""
import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize('category',[None,'current','missing','invalid-position'])
def test_keyword_hits_keep_distinct_sources(tmp_path,monkeypatch,category):
    monkeypatch.setattr(VectorDatabase,'_check_gpu_availability',lambda self:False)
    vector=VectorDatabase(str(tmp_path/'vectors'),IndexConfig(index_type='flat'))
    vector._model=object()
    embeddings=np.tile(np.eye(1,384,dtype=np.float32),(2,1))
    monkeypatch.setattr(vector,'_get_embeddings_batch',lambda texts:embeddings)
    docs=[dict(title='permit',content='permit requires written approval',category=scope) for scope in ['archive','current']]
    for i,doc in enumerate(docs):
        doc['id']=i+10
    vector.add_documents(docs)
    monkeypatch.setattr(vector,'_get_embedding',lambda query:embeddings[0])
    # A threshold above all cosine scores deliberately isolates real BM25 recall.
    if category=='invalid-position':
        monkeypatch.setattr(vector.bm25,'search',lambda *args,**kwargs:[(99,1.0)])
        with pytest.raises(IndexError):
            vector.hybrid_search('permit',top_k=2,threshold=2,keyword_weight=1)
        return
    filters={'category':category} if category else None
    results=vector.hybrid_search('permit',top_k=2,threshold=2,keyword_weight=1,filters=filters)
    expected={'archive','current'} if category is None else ({'current'} if category=='current' else set())
    assert {row['category'] for row in results}==expected
    assert len(results)==len(expected)
    assert all(row['vector_score']==0 and row['bm25_score']>0 for row in results)
