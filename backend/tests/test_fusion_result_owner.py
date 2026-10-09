"""Fusion owns its scores and never mutates the cached vector hits."""
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize('fail_keyword',[False,True])
def test_fusion_keeps_prior_results_and_vector_cache_independent(tmp_path,monkeypatch,fail_keyword):
    monkeypatch.setattr(VectorDatabase,'_check_gpu_availability',lambda self:False)
    vector=VectorDatabase(str(tmp_path/'vectors'),IndexConfig(index_type='flat'))
    vector._model=object()
    embeddings=np.zeros((2,384),dtype=np.float32)
    embeddings[0,0]=1
    embeddings[1,:2]=[.6,.8]
    monkeypatch.setattr(vector,'_get_embeddings_batch',lambda texts:embeddings)
    vector.add_documents([dict(id=1,title='alpha',content='alpha permit requires written approval'),dict(id=2,title='beta',content='beta permit requires written approval')])
    encode=Mock(return_value=embeddings[0])
    monkeypatch.setattr(vector,'_get_embedding',encode)
    pure=vector.search('beta',top_k=2,threshold=0)
    before=deepcopy(pure)
    first=vector.hybrid_search('beta',top_k=2,threshold=0,keyword_weight=0)
    first_before=deepcopy(first)
    assert first[0]['id']==1
    if fail_keyword:
        error=OSError('synthetic keyword failure')
        monkeypatch.setattr(vector.bm25,'search',Mock(side_effect=error))
        with pytest.raises(OSError) as caught:
            vector.hybrid_search('beta',top_k=2,threshold=0,keyword_weight=1)
        assert caught.value is error
    else:
        second=vector.hybrid_search('beta',top_k=2,threshold=0,keyword_weight=1)
        assert second[0]['id']==2
        second[0]['fused_score']=-1
    assert first==first_before
    assert pure==before==vector.search('beta',top_k=2,threshold=0)
    assert encode.call_count==1
    assert all('fused_score' not in row and 'bm25_score' not in row for row in pure)


@pytest.mark.parametrize("mode", ["vector", "hybrid", "keyword_only"])
def test_callers_cannot_mutate_cached_or_indexed_nested_metadata(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    vector._model = object()
    embedding = np.eye(1, 384, dtype=np.float32)
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embedding)
    query = embedding[0].copy()
    if mode == "keyword_only":
        query[0], query[1] = 0, 1
    encode = Mock(return_value=query)
    monkeypatch.setattr(vector, "_get_embedding", encode)
    vector.add_documents([dict(id=1, title="alpha permission", content="alpha access requires written approval",
                              access_rules=dict(conditions=["written approval"]))])
    search = vector.search if mode == "vector" else vector.hybrid_search
    first = search("alpha", top_k=1, threshold=0.5)
    original = deepcopy(first)
    assert len(first) == 1
    first[0]["title"] = "changed title"
    first[0]["access_rules"]["conditions"].clear()
    first.clear()
    second = search("alpha", top_k=1, threshold=0.5)
    assert second == original
    second[0]["access_rules"]["conditions"].append("caller-only condition")
    vector.clear_cache()
    assert search("alpha", top_k=1, threshold=0.5) == original
    assert encode.call_count == 2
