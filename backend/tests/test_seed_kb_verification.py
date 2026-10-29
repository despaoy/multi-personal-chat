"""CLI verification uses a real isolated index; only encoder weights are synthetic."""

import json
import logging
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from knowledge import vector_db


@pytest.mark.parametrize("miss", [False, True], ids=["all-hit", "one-miss"])
def test_seed_verification_exit_matches_retrieval(monkeypatch, tmp_path, caplog, miss):
    seed = Path(__file__).resolve().parents[1] / "data/kb_seed_documents.json"
    documents = json.loads(seed.read_text(encoding="utf-8"))["documents"]
    queries = {
        "胡桃的元素战技是什么？": "hutao_skill",
        "钟离的护盾怎么算？": "zhongli_shield",
        "七七的治疗量受什么影响？": "qiqi_heal",
        "元素反应有哪些类型？": "elemental_reactions",
        "树脂怎么恢复？": "resin_system",
    }
    identities = [document["id"] for document in documents]
    assert set(queries.values()) <= set(identities)
    assert 5 < len(documents) <= 384
    basis = np.eye(384, dtype=np.float32)
    text_vectors = {
        f"{document['title']} {document['content']}": basis[index] for index, document in enumerate(documents)
    }
    query_vectors = {query: basis[identities.index(identity)] for query, identity in queries.items()}
    if miss:
        query_vectors["树脂怎么恢复？"] = -np.ones(384, dtype=np.float32)
    monkeypatch.setattr(vector_db.VectorDatabase, "_check_gpu_availability", lambda self: False)
    vector = vector_db.VectorDatabase(str(tmp_path / "vectors"), vector_db.IndexConfig(index_type="flat"))
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: np.stack([text_vectors[text] for text in texts]))
    monkeypatch.setattr(vector, "_get_embedding", query_vectors.__getitem__)
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: vector)
    monkeypatch.setattr(sys, "argv", ["seed_kb_importer", "--seed-file", str(seed), "--verify"])
    with caplog.at_level(logging.INFO, logger="__main__"):
        if miss:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("knowledge.seed_kb_importer", run_name="__main__")
            assert caught.value.code == 1
        else:
            runpy.run_module("knowledge.seed_kb_importer", run_name="__main__")
    assert f"验证结果: {4 if miss else 5}/5 命中" in caplog.text
    for query in queries:
        assert f"query='{query}'" in caplog.text
    assert [document["id"] for document in vector.metadata] == identities
    assert vector.index.ntotal == len(documents)
    assert vector.bm25.get_stats()["total_docs"] == len(documents)
    assert vector.snapshot_path.exists() and not vector._dirty
    for query, identity in queries.items():
        if miss and query == "树脂怎么恢复？":
            assert vector.search(query, top_k=5, threshold=0.0) == []
        else:
            assert vector.search(query, top_k=5, threshold=0.0)[0]["id"] == identity
