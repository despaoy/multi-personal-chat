"""Eligible evidence beyond the unfiltered prefix must remain retrievable."""
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize("mode", ["vector", "hybrid_keyword"])
def test_filtered_recall_reaches_matching_evidence_after_unrelated_prefix(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    vector._model = object()
    embeddings = np.zeros((32, 384), dtype=np.float32)
    embeddings[:, 0] = np.linspace(1, 0.69, 32)
    embeddings[:, 1] = np.sqrt(1 - embeddings[:, 0] ** 2)
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embeddings)
    rows = [dict(id=i, title="permit", content="permit requires written approval", collection="target" if i == 31 else "other") for i in range(32)]
    vector.add_documents(rows)
    query = np.eye(1, 384, dtype=np.float32)[0]
    if mode == "hybrid_keyword":
        query[:] = 0
        query[2] = 1
    encode = Mock(return_value=query)
    monkeypatch.setattr(vector, "_get_embedding", encode)
    search = vector.search if mode == "vector" else vector.hybrid_search
    kwargs = {} if mode == "vector" else dict(keyword_weight=1)
    result = search("permit", top_k=1, threshold=0.5, filters={"collection": "target"}, **kwargs)
    assert [row["id"] for row in result] == [31]
    assert search("permit", top_k=1, threshold=0.5, filters={"collection": "missing"}, **kwargs) == []
    if mode == "vector":
        assert search("permit", top_k=1, threshold=0.9, filters={"collection": "target"}) == []
