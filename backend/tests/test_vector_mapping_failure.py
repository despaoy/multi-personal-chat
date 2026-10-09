"""A corrupt vector hit cannot become an empty or keyword-only success."""
import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.fixture
def vector(tmp_path, monkeypatch):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    db = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    db._model = object()
    embedding = np.eye(1, 384, dtype=np.float32)
    monkeypatch.setattr(db, "_get_embeddings_batch", lambda texts: embedding)
    monkeypatch.setattr(db, "_get_embedding", lambda query: embedding[0])
    db.add_documents([dict(id=7, title="permit", content="permit requires written approval")])
    return db


@pytest.mark.parametrize("mode", ["search", "hybrid_search"])
@pytest.mark.parametrize("mapping", [None, -1, 1])
def test_corrupt_mapping_raises_and_repaired_query_recovers(vector, mode, mapping):
    if mapping is None:
        del vector._id_to_index[7]
    else:
        vector._id_to_index[7] = mapping
    search = getattr(vector, mode)
    with pytest.raises(ValueError, match="mapping.*rebuild"):
        search("permit", top_k=5)
    vector._id_to_index[7] = 0
    assert [doc["id"] for doc in search("permit", top_k=5)] == [7]


def test_faiss_padding_and_valid_no_match_remain_normal(vector):
    assert [doc["id"] for doc in vector.search("permit", top_k=5)] == [7]
    assert vector.search("permit", top_k=5, threshold=1.1) == []
    assert vector.search("permit", top_k=5, filters={"collection": "missing"}) == []
