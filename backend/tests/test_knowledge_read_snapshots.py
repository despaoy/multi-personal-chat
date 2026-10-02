"""Only cross-page source snapshots and their owned connection lifecycle."""

import os
import sqlite3

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://unused@127.0.0.1:1/unused")

from db.database import SQLiteDB  # noqa: E402
from db.pg_database import SyncPgAdapter  # noqa: E402


@pytest.fixture
def database(tmp_path):
    db = SQLiteDB(tmp_path / "snapshot.db")
    base = db.create_knowledge_base("Complete paired course", "")
    for title, content in [
        ("Arrangement", "Version ONE: date 2026-12-20; matched confirmation ONE required."),
        ("Confirmation", "Version ONE: deadline 2026-12-18 19:00; written confirmation required."),
    ]:
        db.save_knowledge_document(
            {"title": title, "content": content, "knowledge_base_id": base["id"]}, chunks=[content]
        )
    yield db
    db.close_connection()


@pytest.mark.parametrize("batch_size", [1, 2, 500])
def test_snapshot_keeps_paired_rules_across_two_commits(database, batch_size):
    rows = database.iter_chunks_with_document(batch_size=batch_size)
    first = next(rows)
    database.save_knowledge_document(
        {"title": "Arrangement TWO", "content": "Version TWO arrangement complete."},
        doc_id=1,
        chunks=["Version TWO arrangement complete."],
    )
    database.save_knowledge_document(
        {"title": "Confirmation TWO", "content": "Version TWO confirmation complete."},
        doc_id=2,
        chunks=["Version TWO confirmation complete."],
    )
    remaining = list(rows)
    assert first["doc_title"] == "Arrangement" and remaining[0]["doc_title"] == "Confirmation"
    assert all("Version ONE" in row["content"] for row in [first, *remaining])
    assert all("Version TWO" in row["content"] for row in database.iter_chunks_with_document(batch_size=1))


def test_deletion_does_not_shift_later_page(database):
    rows = database.iter_chunks_with_document(batch_size=1)
    first = next(rows)
    assert database.delete_knowledge_document(first["documentId"])
    rest = list(rows)
    assert [r["documentId"] for r in rest] == [2] and rest[0]["doc_title"] == "Confirmation"


def test_insert_not_in_existing_snapshot(database):
    rows = database.iter_chunks_with_document(batch_size=1)
    first = next(rows)
    database.save_knowledge_document(
        {"title": "New complete source", "content": "Version TWO independent complete source."},
        chunks=["Version TWO independent complete source."],
    )
    assert len([first, *rows]) == 2
    assert len(list(database.iter_chunks_with_document(batch_size=1))) == 3


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_page_size_rejected(database, batch_size):
    with pytest.raises(ValueError):
        next(database.iter_chunks_with_document(batch_size=batch_size))


def test_early_close_releases_sqlite_read_snapshot(database):
    rows = database.iter_chunks_with_document(batch_size=1)
    next(rows)
    rows.close()
    writer = sqlite3.connect(database.db_path, timeout=0.1, isolation_level=None)
    try:
        # WAL checkpoint can complete only after the owned read snapshot closes.
        result = writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert result[0] == 0
    finally:
        writer.close()


class BatchBackend:
    def __init__(self, fail=False):
        self.fail = fail
        self.reader_closed = False
        self.backend_closed = False
        self.batches_yielded = 0

    async def init(self):
        pass

    async def close(self):
        self.backend_closed = True

    async def iter_chunk_document_batches(self, batch_size=500):
        try:
            self.batches_yielded += 1
            yield [{"documentId": 1}, {"documentId": 2}]
            if self.fail:
                raise OSError("actual asynchronous page failure")
            self.batches_yielded += 1
            yield [{"documentId": 3}]
        finally:
            self.reader_closed = True


def test_sync_bridge_returns_one_batch_without_per_row_calls():
    pg = BatchBackend()
    adapter = SyncPgAdapter(pg)
    rows = adapter.iter_chunks_with_document()
    try:
        assert next(rows) == {"documentId": 1} and next(rows) == {"documentId": 2}
        assert pg.batches_yielded == 1
        assert list(rows) == [{"documentId": 3}] and pg.reader_closed
        assert not adapter._knowledge_readers and not adapter._pending
    finally:
        adapter.close()


def test_sync_early_close_closes_async_snapshot():
    pg = BatchBackend()
    adapter = SyncPgAdapter(pg)
    rows = adapter.iter_chunks_with_document()
    try:
        next(rows)
        rows.close()
        assert pg.reader_closed and not adapter._knowledge_readers and not adapter._pending
    finally:
        adapter.close()


def test_sync_page_failure_releases_async_snapshot():
    pg = BatchBackend(fail=True)
    adapter = SyncPgAdapter(pg)
    rows = adapter.iter_chunks_with_document()
    try:
        next(rows)
        next(rows)
        with pytest.raises(OSError):
            next(rows)
        assert pg.reader_closed and not adapter._knowledge_readers and not adapter._pending
    finally:
        adapter.close()


def test_adapter_shutdown_closes_idle_async_snapshot():
    pg = BatchBackend()
    adapter = SyncPgAdapter(pg)
    rows = adapter.iter_chunks_with_document()
    next(rows)
    adapter.close()
    rows.close()
    assert pg.reader_closed and pg.backend_closed and not adapter._knowledge_readers and not adapter._pending
