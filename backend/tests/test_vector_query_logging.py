"""Cache diagnostics must not expose retrieval questions through normal or hybrid search."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge import vector_db


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_cached_search_keeps_diagnostics_and_results_without_query_text(tmp_path, monkeypatch, caplog, mode):
    monkeypatch.setattr(vector_db.VectorDatabase, "_check_gpu_availability", lambda self: False)
    index = vector_db.VectorDatabase(str(tmp_path / "vectors"), vector_db.IndexConfig(index_type="flat"))
    embedding = np.eye(1, 384, dtype=np.float32)
    monkeypatch.setattr(index, "_get_embeddings_batch", lambda texts: embedding)
    encode = Mock(return_value=embedding[0])
    monkeypatch.setattr(index, "_get_embedding", encode)
    index.add_documents([dict(id="synthetic-one", title="完整合成题目", content="三张票，每张两元，总额六元。")])
    query = "synthetic-private-question 三张票，每张两元，请计算总额。"
    caplog.set_level(logging.DEBUG, logger=vector_db.logger.name)
    if mode == "vector":
        invoke = lambda: index.search(query, top_k=1, threshold=0)
    else:
        invoke = lambda: index.hybrid_search(query, top_k=1, threshold=0, keyword_weight=0)
    first, second = invoke(), invoke()
    assert first == second and [item["id"] for item in second] == ["synthetic-one"]
    encode.assert_called_once_with(query)
    records = [record for record in caplog.records if record.name == vector_db.logger.name]
    assert records and any(record.levelno == logging.DEBUG for record in records)
    assert "synthetic-private-question" not in caplog.text and query not in caplog.text
