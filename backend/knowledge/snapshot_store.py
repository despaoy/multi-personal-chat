"""Publish and read one coherent FAISS/metadata/BM25 snapshot."""

import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import faiss

SNAPSHOT_NAME = "index_snapshot.zip"
MEMBERS = ("faiss_index.bin", "metadata.pkl", "bm25_state.pkl")


@contextmanager
def _workspace(root):
    root = Path(root).resolve()
    folder = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=root)).resolve()
    try:
        yield folder
    finally:
        if folder.parent != root or not folder.name.startswith(".snapshot-"):
            raise ValueError("Snapshot workspace escaped its index directory")
        shutil.rmtree(folder)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_file(path):
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_faiss(path, index):
    faiss.write_index(index, str(path))
    _sync_file(path)


def _write_pickle(path, value):
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())


def validate_snapshot(index, metadata, bm25, dimension, id_converter):
    """Counts alone cannot detect positional BM25/source mismatches."""
    if not isinstance(metadata, list) or not isinstance(bm25, dict):
        raise ValueError("Invalid snapshot record types")
    if index.d != dimension or index.ntotal != len(metadata) or not hasattr(index, "id_map"):
        raise ValueError("FAISS dimension or count mismatch")
    ids = [id_converter(record["id"]) for record in metadata]
    actual_ids = faiss.vector_to_array(index.id_map).tolist()
    if (
        None in ids
        or len(set(ids)) != len(ids)
        or len(set(actual_ids)) != len(actual_ids)
        or set(ids) != set(actual_ids)
    ):
        raise ValueError("FAISS/source identity mismatch")
    corpus = [f"{record.get('title', '')} {record.get('content', '')}" for record in metadata]
    tokens = bm25["tokenized_corpus"]
    lengths = bm25["doc_lens"]
    if bm25["corpus"] != corpus or len(tokens) != len(corpus) or lengths != [len(row) for row in tokens]:
        raise ValueError("BM25/source position mismatch")
    frequencies = Counter(token for row in tokens for token in set(row))
    average = sum(lengths) / len(lengths) if lengths else 0
    expected_idf = {
        token: math.log((len(corpus) - count + 0.5) / (count + 0.5) + 1) for token, count in frequencies.items()
    }
    if dict(bm25["doc_freqs"]) != dict(frequencies) or bm25["avgdl"] != average or bm25["idf"] != expected_idf:
        raise ValueError("BM25 statistics mismatch")
    if corpus and bm25["built"] is not True:
        raise ValueError("Nonempty keyword index is not built")


def save_snapshot(root, index, metadata, bm25, dimension, id_converter):
    validate_snapshot(index, metadata, bm25, dimension, id_converter)
    root = Path(root)
    with _workspace(root) as folder:
        _write_faiss(folder / MEMBERS[0], index)
        _write_pickle(folder / MEMBERS[1], metadata)
        _write_pickle(folder / MEMBERS[2], bm25)
        manifest = {
            "schema": 1,
            "dimension": dimension,
            "count": len(metadata),
            "sha256": {name: _hash(folder / name) for name in MEMBERS},
        }
        candidate = folder / SNAPSHOT_NAME
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in MEMBERS:
                archive.write(folder / name, arcname=name)
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        _sync_file(candidate)
        os.replace(candidate, root / SNAPSHOT_NAME)
        if os.name == "posix":
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def load_snapshot(root, dimension, id_converter):
    root = Path(root)
    # One open descriptor pins all members even if a writer replaces the path.
    with zipfile.ZipFile(root / SNAPSHOT_NAME, "r") as archive, _workspace(root) as folder:
        names = archive.namelist()
        if len(names) != 4 or set(names) != {*MEMBERS, "manifest.json"}:
            raise ValueError("Invalid snapshot members")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["schema"] != 1 or manifest["dimension"] != dimension or set(manifest["sha256"]) != set(MEMBERS):
            raise ValueError("Unsupported snapshot manifest")
        for name in MEMBERS:
            path = folder / name
            with archive.open(name) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if _hash(path) != manifest["sha256"][name]:
                raise ValueError("Snapshot member checksum mismatch")
        index = faiss.read_index(str(folder / MEMBERS[0]))
        with (folder / MEMBERS[1]).open("rb") as stream:
            metadata = pickle.load(stream)
        with (folder / MEMBERS[2]).open("rb") as stream:
            bm25 = pickle.load(stream)
        validate_snapshot(index, metadata, bm25, dimension, id_converter)
        if manifest["count"] != len(metadata):
            raise ValueError("Snapshot manifest count mismatch")
        return index, metadata, bm25
