"""Index construction uses requested types and effective native parameters."""
import faiss
import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


def database(kind='auto',count=256):
    result=VectorDatabase.__new__(VectorDatabase)
    result.config=IndexConfig(index_type=kind,nlist=2,nprobe=2)
    result.metadata=[None]*count
    result.index=object()
    return result

@pytest.mark.parametrize('kind',['flat','ivf','hnsw'])
def test_actual_index_trains_and_retrieves_requested_ids(kind):
    old_threads=faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        vector=database(kind)
        vector._create_index()
        native=faiss.downcast_index(vector.index.index)
        expected={'flat':faiss.IndexFlatIP,'ivf':faiss.IndexIVFFlat,'hnsw':faiss.IndexHNSWFlat}[kind]
        assert isinstance(native,expected)
        if kind=='ivf':
            assert native.nprobe==2 and native.nlist==2
        elif kind=='hnsw':
            assert native.hnsw.efConstruction==vector.config.ef_construction
            assert native.hnsw.efSearch==vector.config.ef_search
        vectors=np.random.default_rng(73).normal(size=(256,384)).astype(np.float32)
        faiss.normalize_L2(vectors)
        ids=np.arange(1000,1256,dtype=np.int64)
        if not vector.index.is_trained:
            vector.index.train(vectors)
        vector.index.add_with_ids(vectors,ids)
        scores,found=vector.index.search(vectors[:1],1)
        assert found[0,0]==1000 and scores[0,0]==pytest.approx(1,abs=1e-5)
    finally:
        faiss.omp_set_num_threads(old_threads)

@pytest.mark.parametrize('bad',['','unknown','FLAT'])
@pytest.mark.parametrize('explicit',[False,True])
def test_invalid_type_raises_without_replacing_current_index(bad,explicit):
    vector=database('flat' if explicit else bad)
    previous=vector.index
    with pytest.raises(ValueError,match='index_type'):
        vector._create_index(bad) if explicit else vector._create_index()
    assert vector.index is previous

@pytest.mark.parametrize('count,expected',[(0,faiss.IndexFlatIP),(10000,faiss.IndexIVFFlat),(100000,faiss.IndexHNSWFlat)])
def test_auto_policy_still_creates_expected_native_type(count,expected):
    vector=database(count=count)
    vector._create_index()
    assert isinstance(faiss.downcast_index(vector.index.index),expected)
