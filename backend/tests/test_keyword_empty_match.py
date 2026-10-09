"""No lexical match is an empty result, not a zero-score candidate."""
import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize("query,expected", [("permit", [1]), ("zebra", [])])
def test_keyword_channel_does_not_fill_with_zero_score_documents(tmp_path, monkeypatch, query, expected):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    vector._model = object()
    embeddings = np.tile(np.eye(1, 384, dtype=np.float32), (2, 1))
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embeddings)
    vector.add_documents([dict(id=1, title="permit", content="Written permission required"),
                          dict(id=2, title="calendar", content="The workshop opens Monday")])
    orthogonal = np.zeros(384, dtype=np.float32)
    orthogonal[1] = 1
    monkeypatch.setattr(vector, "_get_embedding", lambda query: orthogonal)
    keyword = vector.bm25.search(query, top_k=5, threshold=0)
    assert [vector.metadata[i]["id"] for i, _ in keyword] == expected
    assert all(score > 0 for _, score in keyword)
    hybrid = vector.hybrid_search(query, top_k=5, threshold=0.5)
    assert [row["id"] for row in hybrid] == expected
    assert all(row["bm25_score"] > 0 for row in hybrid)
