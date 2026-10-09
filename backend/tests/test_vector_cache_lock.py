"""Cached retrieval must observe completed index mutations under the same lock."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase


@pytest.mark.parametrize("cached", [False, True])
def test_query_waits_for_index_clear_and_cannot_return_removed_document(tmp_path, monkeypatch, cached):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))
    vector._model = object()
    embedding = np.eye(1, 384, dtype=np.float32)
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: embedding)
    encode = Mock(return_value=embedding[0])
    monkeypatch.setattr(vector, "_get_embedding", encode)
    vector.add_documents([dict(id=1, title="Written permission", content="Access requires written permission through October 12.")])
    if cached:
        assert vector.search("permission", threshold=0)[0]["id"] == 1
    started = Event()
    def read():
        started.set()
        return vector.search("permission", threshold=0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with vector._lock:
            future = pool.submit(read)
            assert started.wait(1)
            try:
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.1)
            finally:
                vector.clear_all()
        assert future.result(timeout=2) == []
    assert vector.search("permission", threshold=0) == []
    assert encode.call_count == int(cached)
