"""Only persisted cross-worker authority, loaded evidence and cache freshness."""

from copy import deepcopy
from threading import RLock

import pytest

from api import knowledge
from db.database import SQLiteDB


class RecordingVector:
    """No embedding model: observe the index rebuild contract in unit tests."""

    def __init__(self):
        self._lock = RLock()
        self.metadata = []
        self.cache_generation = 0
        self.builds = 0
        self.on_add = None

    def get_stats(self):
        return dict(
            total_documents=len(self.metadata), index_size=len(self.metadata), bm25_corpus_size=len(self.metadata)
        )

    def clear_all(self):
        self.metadata = []
        self.cache_generation += 1

    def clear_cache(self):
        self.cache_generation += 1

    def add_documents(self, documents):
        self.metadata.extend(deepcopy(documents))
        self.builds += 1
        if self.on_add:
            callback = self.on_add
            self.on_add = None
            callback()

    def flush(self):
        pass


@pytest.fixture
def readers(tmp_path, monkeypatch):
    import app.config
    from knowledge import vector_db as vector_module

    path = tmp_path / "shared.db"
    reader = SQLiteDB(path)
    writer = SQLiteDB(path)
    old = reader.create_knowledge_base("旧库", "")
    new = reader.create_knowledge_base("新库", "")
    doc = reader.add_knowledge_document(
        dict(
            title="旧标题",
            content="完整课程规则。普通雨照常，红色暴雨停课。末尾仍须书面确认。",
            category="旧类",
            knowledge_base_id=old["id"],
            chunkCount=1,
        )
    )
    reader.add_knowledge_chunk(dict(documentId=doc["id"], chunkIndex=0, content=doc["content"]))
    vector = RecordingVector()
    monkeypatch.setattr(knowledge, "db", reader)
    monkeypatch.setattr(app.config, "VECTOR_DB_AVAILABLE", True)
    monkeypatch.setattr(vector_module, "get_vector_db", lambda: vector)
    monkeypatch.setattr(knowledge, "_vector_index_built", False)
    monkeypatch.setattr(knowledge, "_vector_index_revision", None)
    assert knowledge._ensure_vector_index()
    return reader, writer, doc, new, vector


def external_update(writer, doc, change, complete=False):
    writer.update_knowledge_document(doc["id"], change)
    if "content" in change:
        writer.execute_sql("DELETE FROM knowledge_chunks WHERE documentId = :doc_id", {"doc_id": doc["id"]})
        writer.add_knowledge_chunk(dict(documentId=doc["id"], chunkIndex=0, content=change["content"]))
    revision = int(writer.get_config_value(knowledge._VECTOR_REBUILD_REVISION_KEY, "0")) + 1
    writer.set_config_value(knowledge._VECTOR_REBUILD_REVISION_KEY, str(revision))
    # Unit-only simulated global marker; live probe builds a real second index.
    writer.set_config_value(
        knowledge._VECTOR_REBUILD_STATUS_KEY,
        f"complete:1:{knowledge._compute_chunk_fingerprint()}:{revision}" if complete else "dirty",
    )
    return revision


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("title", "新标题"),
        ("category", "新类"),
        ("knowledge_base_id", "new"),
        ("content", "完整新规则。旧安排作废，末尾必须书面确认。"),
    ],
)
def test_external_revision_refreshes_warm_reader_even_when_other_worker_complete(readers, field, value, complete):
    reader, writer, doc, new, vector = readers
    if value == "new":
        value = new["id"]
    old = deepcopy(vector.metadata)
    generation = vector.cache_generation
    builds = vector.builds
    revision = external_update(writer, doc, {field: value}, complete)
    assert knowledge._vector_index_built and vector.metadata == old
    assert knowledge._ensure_vector_index()
    assert vector.cache_generation > generation and vector.builds == builds + 1
    assert knowledge._vector_index_revision == revision and knowledge._read_rebuild_status()[3] == revision
    current = vector.metadata[0]
    if field == "content":
        assert value in current["content"]
    else:
        assert current[field] == value
    assert reader.get_knowledge_chunks(doc["id"])[0]["content"] in current["content"]


def test_unchanged_revision_uses_one_authority_read_without_chunk_scan(readers, monkeypatch):
    reader, writer, doc, new, vector = readers
    original = reader.get_config_value
    reads = []

    def read(key, default=None):
        reads.append(key)
        return original(key, default)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unchanged revision must not rescan or reembed")

    monkeypatch.setattr(reader, "get_config_value", read)
    monkeypatch.setattr(reader, "iter_chunks_with_document", forbidden)
    builds = vector.builds
    generation = vector.cache_generation
    assert knowledge._ensure_vector_index()
    assert (
        reads == [knowledge._VECTOR_REBUILD_REVISION_KEY]
        and vector.builds == builds
        and vector.cache_generation == generation
    )


def test_fresh_process_rejects_same_count_stale_local_disk_metadata(readers, monkeypatch):
    reader, writer, doc, new, vector = readers
    revision = external_update(writer, doc, dict(title="新标题"), True)
    monkeypatch.setattr(knowledge, "_vector_index_built", False)
    monkeypatch.setattr(knowledge, "_vector_index_revision", None)
    builds = vector.builds
    assert knowledge._ensure_vector_index() and vector.builds == builds + 1
    assert vector.metadata[0]["title"] == "新标题" and knowledge._vector_index_revision == revision


def test_valid_local_persisted_evidence_is_reused_without_reembedding(readers, monkeypatch):
    reader, writer, doc, new, vector = readers
    monkeypatch.setattr(knowledge, "_vector_index_built", False)
    monkeypatch.setattr(knowledge, "_vector_index_revision", None)
    builds = vector.builds
    generation = vector.cache_generation
    assert knowledge._ensure_vector_index() and vector.builds == builds
    assert (
        vector.cache_generation > generation and knowledge._vector_index_revision == knowledge._get_rebuild_revision()
    )


@pytest.mark.parametrize("failure", ["unreadable", "malformed", "negative"])
def test_failed_persisted_authority_never_authorizes_warm_cached_index(readers, monkeypatch, failure):
    reader, writer, doc, new, vector = readers
    if failure == "unreadable":

        def fail(*args, **kwargs):
            raise RuntimeError("DB unavailable")

        monkeypatch.setattr(reader, "get_config_value", fail)
    else:
        writer.set_config_value(knowledge._VECTOR_REBUILD_REVISION_KEY, "bad" if failure == "malformed" else "-1")
    builds = vector.builds
    metadata = deepcopy(vector.metadata)
    assert knowledge._ensure_vector_index() is False
    assert vector.builds == builds and vector.metadata == metadata


def test_external_update_during_rebuild_cannot_commit_an_old_revision(readers):
    reader, writer, doc, new, vector = readers
    external_update(writer, doc, dict(title="第一次更新"))
    vector.on_add = lambda: external_update(writer, doc, dict(title="重建中再次更新"))
    assert knowledge._ensure_vector_index() is False and not knowledge._vector_index_built
    assert knowledge._ensure_vector_index() and vector.metadata[0]["title"] == "重建中再次更新"
    assert knowledge._vector_index_revision == knowledge._get_rebuild_revision()


def test_external_last_document_delete_clears_old_index_and_cache(readers):
    reader, writer, doc, new, vector = readers
    generation = vector.cache_generation
    revision = knowledge._get_rebuild_revision() + 1
    writer.delete_knowledge_document(doc["id"])
    writer.set_config_value(knowledge._VECTOR_REBUILD_REVISION_KEY, str(revision))
    writer.set_config_value(knowledge._VECTOR_REBUILD_STATUS_KEY, "dirty")
    assert knowledge._ensure_vector_index() and vector.metadata == [] and vector.cache_generation > generation
    assert knowledge._vector_index_revision == revision and knowledge._read_rebuild_status() == (
        "complete",
        0,
        "empty",
        revision,
    )


def test_corrupt_local_identity_cannot_pass_global_complete_count(readers, monkeypatch):
    reader, writer, doc, new, vector = readers
    vector.metadata[0]["id"] = "wrong-local-id"
    monkeypatch.setattr(knowledge, "_vector_index_built", False)
    monkeypatch.setattr(knowledge, "_vector_index_revision", None)
    assert knowledge._ensure_vector_index() and vector.metadata[0]["id"] == f"doc_{doc['id']}_chunk_0"
