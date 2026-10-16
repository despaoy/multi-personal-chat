"""Persisted FAISS keys and invalid-identity errors have one explicit contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from knowledge.vector_db import IndexConfig, VectorDatabase

GOLDEN = json.loads((Path(__file__).parent / "fixtures/vector_identity_keys.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", GOLDEN)
def test_persisted_valid_identity_keys_do_not_change(row):
    vector = object.__new__(VectorDatabase)
    value = vector._to_faiss_id(row["id"])
    assert type(value) is int and value == row["key"]


@pytest.mark.parametrize("value", [np.int64(7), np.uint64((1 << 64) - 1)])
def test_integral_array_identity_uses_the_same_integer_contract(value):
    vector = object.__new__(VectorDatabase)
    expected = 7 if int(value) == 7 else (1 << 63) - 1
    assert vector._to_faiss_id(value) == expected


@pytest.mark.parametrize("value", [None, True, 1.75, "", " \t", [], {}])
def test_invalid_identity_never_becomes_a_numeric_key_or_none(value, caplog):
    vector = object.__new__(VectorDatabase)
    with pytest.raises(ValueError, match="Document ID"):
        vector._to_faiss_id(value)
    assert not caplog.records


def test_string_encoding_error_is_not_swallowed_or_logged_with_raw_identity(caplog):
    vector = object.__new__(VectorDatabase)
    with pytest.raises(UnicodeEncodeError):
        vector._to_faiss_id("synthetic-invalid-\ud800-id")
    assert not caplog.records


def test_nonempty_snapshot_preserves_all_selected_original_identities_and_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda _: False)
    ids = [
        0,
        7,
        -2,
        -(1 << 63),
        (1 << 63) - 1,
        "abc",
        "0123456789abcdef0123456789abcdef",
        "doc_17_chunk_0",
        "虚构来源-甲",
        "abc-not-a-hex-id",
    ]
    docs = [
        dict(
            id=identity,
            title=f"完整虚构登记{index}",
            content=f"第{index}项业务只有取得书面确认才受理；雨天暂停，预约不豁免。",
        )
        for index, identity in enumerate(ids)
    ]
    vectors = np.eye(len(docs), 384, dtype=np.float32)
    vector = VectorDatabase(str(tmp_path), IndexConfig(index_type="flat"))
    monkeypatch.setattr(vector, "_get_embeddings_batch", lambda texts: vectors)
    vector.add_documents(docs)
    digest = hashlib.sha256(vector.snapshot_path.read_bytes()).hexdigest()
    reopened = VectorDatabase(str(tmp_path), IndexConfig(index_type="flat"))
    assert reopened.snapshot_validated and [row["id"] for row in reopened.metadata] == ids
    queries = {f"第{index}项完整查询": index for index in range(len(ids))}
    monkeypatch.setattr(reopened, "_get_embedding", lambda query: vectors[queries[query]])
    for query, index in queries.items():
        rows = reopened.search(query, top_k=1)
        assert rows[0]["id"] == ids[index] and rows[0]["content"] == docs[index]["content"]
    assert hashlib.sha256(reopened.snapshot_path.read_bytes()).hexdigest() == digest
