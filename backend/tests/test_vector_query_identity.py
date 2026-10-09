"""Cache identity must match the original embedding input."""
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize("queries", [("US permission", "us permission"), ("A B", "A\nB")])
def test_distinct_model_inputs_cannot_share_cached_results(tmp_path, monkeypatch, queries):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    vector._model = object()
    embeddings = np.eye(2, 384, dtype=np.float32)
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embeddings)
    vector.add_documents([dict(id=1, title="US rules", content="US admission requires written permission"),
                          dict(id=2, title="Group rules", content="Our group requires individual approval")])
    encode = Mock(side_effect=lambda query: embeddings[queries.index(query)])
    monkeypatch.setattr(vector, "_get_embedding", encode)
    for _ in range(2):
        assert vector.search(queries[0], top_k=1)[0]["id"] == 1
        assert vector.search(queries[1], top_k=1)[0]["id"] == 2
    assert [call.args[0] for call in encode.call_args_list] == list(queries)


def test_equivalent_filter_order_uses_same_cached_query(tmp_path, monkeypatch):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    vector._model = object()
    embedding = np.eye(1, 384, dtype=np.float32)
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embedding)
    vector.add_documents([dict(id=1, title="rules", content="Written approval required", category="rules", version="1")])
    encode = Mock(return_value=embedding[0])
    monkeypatch.setattr(vector, "_get_embedding", encode)
    first = vector.search("rules", filters=dict(category="rules", version="1"))
    assert first[0]["id"] == 1
    assert vector.search("rules", filters=dict(version="1", category="rules")) == first
    assert encode.call_count == 1
