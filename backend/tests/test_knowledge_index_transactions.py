"""Persisted index authority transactions and the affected rebuild commits."""

import json
import sqlite3
import subprocess
import sys
from threading import RLock

import pytest

from api import knowledge
from db import knowledge_index_state as state
from db.database import SQLiteDB


def test_actual_independent_sqlite_processes_preserve_all_increments(tmp_path):
    path = tmp_path / "workers.db"
    database = SQLiteDB(path)
    code = """import json,sys
from db.database import SQLiteDB
db=SQLiteDB(sys.argv[1])
print(json.dumps([db.mark_knowledge_index_dirty() for _ in range(12)]))
db.close_connection()
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for _ in range(3)
    ]
    revisions = []
    for child in children:
        stdout, stderr = child.communicate(timeout=25)
        assert child.returncode == 0, stderr
        revisions.extend(json.loads(stdout))
    assert sorted(revisions) == list(range(1, 37))
    assert (
        database.get_config_value(state.REVISION_KEY) == 36 and database.get_config_value(state.STATUS_KEY) == "dirty"
    )


@pytest.mark.parametrize("count,fingerprint", [(0, "empty"), (3, "abc123def456")])
def test_current_completion_commits_and_next_writer_invalidates_it(tmp_path, count, fingerprint):
    database = SQLiteDB(tmp_path / "state.db")
    assert database.commit_knowledge_index_revision(0, count, fingerprint)
    assert database.get_config_value(state.STATUS_KEY) == f"complete:{count}:{fingerprint}:0"
    assert database.mark_knowledge_index_dirty() == 1
    assert database.get_config_value(state.STATUS_KEY) == "dirty"


def test_old_completion_cannot_replace_a_new_dirty_signal(tmp_path):
    database = SQLiteDB(tmp_path / "state.db")
    other = SQLiteDB(database.db_path)
    revision = database.mark_knowledge_index_dirty()
    assert other.mark_knowledge_index_dirty() == revision + 1
    assert database.commit_knowledge_index_revision(revision, 3, "abc123") is False
    assert (
        database.get_config_value(state.REVISION_KEY) == revision + 1
        and database.get_config_value(state.STATUS_KEY) == "dirty"
    )


def test_future_completion_cannot_authorize_a_revision_not_committed(tmp_path):
    database = SQLiteDB(tmp_path / "state.db")
    revision = database.mark_knowledge_index_dirty()
    assert database.commit_knowledge_index_revision(revision + 1, 1, "abc123") is False
    assert (
        database.get_config_value(state.REVISION_KEY) == revision
        and database.get_config_value(state.STATUS_KEY) == "dirty"
    )


@pytest.mark.parametrize("raw", ["bad", "-1", "1.5"])
@pytest.mark.parametrize("operation", ["dirty", "complete"])
def test_invalid_authority_rolls_back_instead_of_resetting_revision(tmp_path, raw, operation):
    database = SQLiteDB(tmp_path / "state.db")
    database.set_config_value(state.REVISION_KEY, raw)
    database.set_config_value(state.STATUS_KEY, "prior-state")
    with pytest.raises((ValueError, TypeError)):
        if operation == "dirty":
            database.mark_knowledge_index_dirty()
        else:
            database.commit_knowledge_index_revision(0, 1, "abc123")
    assert (
        str(database.get_config_value(state.REVISION_KEY)) == raw
        and database.get_config_value(state.STATUS_KEY) == "prior-state"
    )


def test_failed_dirty_write_rolls_back_revision_increment(tmp_path):
    database = SQLiteDB(tmp_path / "state.db")
    revision = database.mark_knowledge_index_dirty()
    assert database.commit_knowledge_index_revision(revision, 1, "abc123")
    database.execute_sql(
        "CREATE TRIGGER reject_dirty BEFORE INSERT ON config WHEN NEW.key='vector_index_rebuild_status' AND NEW.value='dirty' BEGIN SELECT RAISE(ABORT,'unit dirty write failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="dirty write failure"):
        database.mark_knowledge_index_dirty()
    assert (
        database.get_config_value(state.REVISION_KEY) == revision
        and database.get_config_value(state.STATUS_KEY) == f"complete:1:abc123:{revision}"
    )


def test_failed_completion_write_preserves_dirty_authority(tmp_path):
    database = SQLiteDB(tmp_path / "state.db")
    revision = database.mark_knowledge_index_dirty()
    database.execute_sql(
        "CREATE TRIGGER reject_complete BEFORE INSERT ON config WHEN NEW.key='vector_index_rebuild_status' AND NEW.value LIKE 'complete:%' BEGIN SELECT RAISE(ABORT,'unit completion write failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="completion write failure"):
        database.commit_knowledge_index_revision(revision, 1, "abc123")
    assert (
        database.get_config_value(state.REVISION_KEY) == revision
        and database.get_config_value(state.STATUS_KEY) == "dirty"
    )


class RecordingVector:
    def __init__(self):
        self._lock = RLock()
        self.metadata = []
        self.cache_generation = 0

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
        self.metadata.extend(documents)

    def flush(self):
        pass


@pytest.mark.parametrize("branch", ["empty", "rebuild", "reuse"])
def test_all_ready_branches_reject_real_revision_change_at_completion(tmp_path, monkeypatch, branch):
    import app.config
    from knowledge import vector_db as vector_module

    database = SQLiteDB(tmp_path / "state.db")
    other = SQLiteDB(database.db_path)
    vector = RecordingVector()
    if branch != "empty":
        doc = database.add_knowledge_document(
            dict(title="完整规则", content="完整安排。末尾须书面确认。", category="现行", chunkCount=1)
        )
        database.add_knowledge_chunk(dict(documentId=doc["id"], chunkIndex=0, content=doc["content"]))
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(app.config, "VECTOR_DB_AVAILABLE", True)
    monkeypatch.setattr(vector_module, "get_vector_db", lambda: vector)
    monkeypatch.setattr(knowledge, "_vector_index_built", False)
    monkeypatch.setattr(knowledge, "_vector_index_revision", None)
    if branch == "reuse":
        assert knowledge._ensure_vector_index()
        knowledge._vector_index_built = False
        knowledge._vector_index_revision = None
    original = database.commit_knowledge_index_revision

    def concurrent(expected, count, fingerprint):
        other.mark_knowledge_index_dirty()
        return original(expected, count, fingerprint)

    monkeypatch.setattr(database, "commit_knowledge_index_revision", concurrent)
    assert (
        knowledge._ensure_vector_index() is False
        and not knowledge._vector_index_built
        and knowledge._vector_index_revision is None
    )
    assert database.get_config_value(state.STATUS_KEY) == "dirty"


def test_api_dirty_marks_use_one_actual_database_transaction(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path / "state.db")
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(knowledge, "_vector_index_built", True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Separate config writes cannot be used for invalidation")

    monkeypatch.setattr(database, "set_config_value", forbidden)
    knowledge._mark_rebuild_dirty()
    assert database.get_config_value(state.REVISION_KEY) == 1 and database.get_config_value(state.STATUS_KEY) == "dirty"
    assert not knowledge._vector_index_built and knowledge._vector_index_revision is None
