"""Targeted snapshot integrity and failure preservation tests; no model calls."""

import ast
import hashlib
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from knowledge import snapshot_store as store
from knowledge.vector_db import BM25Retriever, VectorDatabase


@pytest.fixture
def vector(tmp_path, monkeypatch):
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    v = VectorDatabase(str(tmp_path))
    v.metadata = [
        {"id": "one", "title": "Confirmed", "content": "LL-842-Q full current rules"},
        {"id": "two", "title": "Retired", "content": "LL-OLD-312 full retired rules"},
    ]
    v.index.add_with_ids(
        np.eye(2, 384, dtype=np.float32), np.array([v._to_faiss_id(d["id"]) for d in v.metadata], dtype=np.int64)
    )
    v.bm25.add_documents(v.metadata)
    return v


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reopen(v):
    return VectorDatabase(str(v.db_path))


def test_complete_round_trip_binds_keyword(vector):
    vector._save_index()
    loaded = reopen(vector)
    position, _ = loaded.bm25.search("LL-842-Q", top_k=1)[0]
    assert loaded.snapshot_validated and loaded.metadata == vector.metadata
    assert loaded.metadata[position]["id"] == "one"
    assert loaded.get_stats()["index_size"] == loaded.get_stats()["bm25_corpus_size"] == 2
    assert not list(vector.db_path.glob(".snapshot-*"))


@pytest.mark.parametrize("stage", ["_write_faiss", "_write_pickle", "replace"])
def test_failed_publication_retains_previous_snapshot(vector, monkeypatch, stage):
    vector._save_index()
    before = digest(vector.snapshot_path)
    vector.metadata[0]["content"] += " Updated"
    vector.bm25 = BM25Retriever()
    vector.bm25.add_documents(vector.metadata)

    def failure(*args, **kwargs):
        raise OSError("targeted disk failure")

    monkeypatch.setattr(store.os if stage == "replace" else store, stage, failure)
    with pytest.raises(OSError):
        vector._save_index()
    assert vector._dirty and digest(vector.snapshot_path) == before
    assert reopen(vector).metadata[0]["content"] == "LL-842-Q full current rules"
    assert not list(vector.db_path.glob(".snapshot-*"))


def test_equal_count_wrong_keyword_order_rejected(vector):
    vector.bm25 = BM25Retriever()
    vector.bm25.add_documents(list(reversed(vector.metadata)))
    with pytest.raises(ValueError, match="position"):
        vector._save_index()
    assert vector._dirty and not vector.snapshot_path.exists()


def test_equal_count_wrong_faiss_ids_rejected(vector):
    vector.index = faiss.IndexIDMap(faiss.IndexFlatIP(384))
    vector.index.add_with_ids(np.eye(2, 384, dtype=np.float32), np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="identity"):
        vector._save_index()


def test_duplicate_source_identity_rejected(vector):
    vector.metadata[1]["id"] = vector.metadata[0]["id"]
    with pytest.raises(ValueError, match="identity"):
        vector._save_index()


def test_keyword_statistics_corruption_rejected(vector):
    vector.bm25.doc_lens[0] += 1
    with pytest.raises(ValueError, match="position"):
        vector._save_index()


def test_corrupt_bundle_never_falls_back_to_old_files(vector):
    faiss.write_index(vector.index, str(vector.index_path))
    vector.metadata_path.write_bytes(pickle.dumps(vector.metadata))
    vector.bm25_path.write_bytes(pickle.dumps(vector._bm25_state()))
    vector.snapshot_path.write_bytes(b"broken complete bundle")
    before = digest(vector.snapshot_path)
    with pytest.raises(zipfile.BadZipFile):
        reopen(vector)
    assert digest(vector.snapshot_path) == before


def test_manifest_checksum_checked(vector):
    vector._save_index()
    with zipfile.ZipFile(vector.snapshot_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["metadata.pkl"] = pickle.dumps(list(reversed(vector.metadata)))
    with zipfile.ZipFile(vector.snapshot_path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="checksum"):
        reopen(vector)


def test_reader_pins_old_descriptor_across_new_publication(vector, monkeypatch):
    vector._save_index()
    old = [dict(d) for d in vector.metadata]
    vector.metadata = list(reversed(vector.metadata))
    vector.bm25 = BM25Retriever()
    vector.bm25.add_documents(vector.metadata)
    original = zipfile.ZipFile.open
    switched = False

    def observed(archive, name, *args, **kwargs):
        nonlocal switched
        result = original(archive, name, *args, **kwargs)
        if str(archive.filename) == str(vector.snapshot_path) and name == "faiss_index.bin" and not switched:
            switched = True
            vector._save_index()
        return result

    monkeypatch.setattr(zipfile.ZipFile, "open", observed)
    loaded = reopen(vector)
    assert switched and loaded.snapshot_validated and loaded.metadata == old
    assert loaded.bm25.corpus == [d["title"] + " " + d["content"] for d in old]
    assert reopen(vector).metadata == vector.metadata


def test_legacy_triple_cannot_authorize_database_reuse(vector):
    faiss.write_index(vector.index, str(vector.index_path))
    vector.metadata_path.write_bytes(pickle.dumps(vector.metadata))
    vector.bm25_path.write_bytes(pickle.dumps(vector._bm25_state()))
    loaded = reopen(vector)
    assert loaded.get_stats()["index_size"] == 2 and not loaded.snapshot_validated
    tree = ast.parse((Path(__file__).resolve().parents[1] / "api/knowledge.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_loaded_metadata_matches_database"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"db": SimpleNamespace(), "getattr": getattr}
    exec(compile(module, "scoped legacy gate", "exec"), namespace)
    assert namespace[node.name](loaded) is False
    vector._save_index()
    assert reopen(vector).snapshot_validated


def test_empty_snapshot_overrides_legacy_sources(vector):
    faiss.write_index(vector.index, str(vector.index_path))
    vector.metadata_path.write_bytes(pickle.dumps(vector.metadata))
    vector.bm25_path.write_bytes(pickle.dumps(vector._bm25_state()))
    vector.clear_all()
    loaded = reopen(vector)
    assert loaded.snapshot_validated and loaded.metadata == [] and loaded.bm25.corpus == []
    assert loaded.index.ntotal == 0


@pytest.mark.parametrize('component', ['index', 'metadata', 'bm25'])
@pytest.mark.parametrize('damage', ['missing', 'corrupt'])
def test_incomplete_legacy_index_raises_without_replacement(vector, component, damage):
    faiss.write_index(vector.index, str(vector.index_path))
    vector.metadata_path.write_bytes(pickle.dumps(vector.metadata))
    vector.bm25_path.write_bytes(pickle.dumps(vector._bm25_state()))
    paths = {'index': vector.index_path, 'metadata': vector.metadata_path, 'bm25': vector.bm25_path}
    target = paths[component]
    if damage == 'missing':
        target.unlink()
    else:
        target.write_bytes(b'broken component')
    before = {p.name: digest(p) for p in paths.values() if p.exists()}
    error = RuntimeError if component == 'index' else FileNotFoundError if damage == 'missing' else pickle.UnpicklingError
    with pytest.raises(error):
        reopen(vector)
    assert {p.name: digest(p) for p in paths.values() if p.exists()} == before
    assert not vector.snapshot_path.exists()


def test_new_empty_database_remains_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(VectorDatabase, '_check_gpu_availability', lambda self: False)
    empty = VectorDatabase(str(tmp_path))
    assert empty.metadata == [] and empty.bm25.corpus == [] and empty.index.ntotal == 0
    assert empty.search('没有资料') == []
