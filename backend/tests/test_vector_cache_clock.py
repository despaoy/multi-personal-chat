"""Wall-clock correction cannot change the vector cache lifetime."""
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge import vector_db


@pytest.mark.parametrize("wall_jump", [-10000, 10000])
def test_query_cache_expires_by_elapsed_time_at_ttl_boundary(tmp_path, monkeypatch, wall_jump):
    monkeypatch.setattr(vector_db.VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = vector_db.VectorDatabase(str(tmp_path / "vectors"), vector_db.IndexConfig(index_type="flat"))
    vector._model = object()
    embedding = np.eye(1, 384, dtype=np.float32)
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embedding)
    encode = Mock(return_value=embedding[0])
    monkeypatch.setattr(vector, "_get_embedding", encode)
    vector.add_documents([dict(id=1, title="permission", content="Written approval is required through October 12.")])
    clock = dict(wall=1000, elapsed=100)
    monkeypatch.setattr(vector_db.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(vector_db.time, "monotonic", lambda: clock["elapsed"])
    first = vector.search("permission")
    assert first[0]["id"] == 1 and encode.call_count == 1
    clock.update(wall=1000 + wall_jump, elapsed=399)
    assert vector.search("permission") == first and encode.call_count == 1
    clock["elapsed"] = 400
    assert vector.search("permission") == first and encode.call_count == 2
