"""Each supported backup artifact must preserve SQLite schema and behavior."""
import sqlite3
from pathlib import Path

import pytest

from infra.backup_manager import BackupManager, BackupType
from scripts.restore_sqlite_backup import restore_sqlite_backup


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [BackupType.FULL, BackupType.INCREMENTAL])
async def test_backup_and_offline_restore_preserve_complete_schema_and_triggers(tmp_path, kind):
    source, target = tmp_path / "source.sqlite", tmp_path / "target.sqlite"
    with sqlite3.connect(source) as connection:
        connection.executescript("""
            CREATE TABLE events(id INTEGER PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE audit(event_id INTEGER NOT NULL);
            CREATE INDEX event_payload_idx ON events(payload);
            CREATE VIEW event_summary AS SELECT id,payload FROM events;
            CREATE TRIGGER event_write_log AFTER INSERT ON events
                BEGIN INSERT INTO audit VALUES(new.id); END;
            INSERT INTO events VALUES(1,'first complete fact'),(2,'second complete fact');
        """)
        expected_schema = connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
    with sqlite3.connect(target) as connection:
        connection.execute("CREATE TABLE previous(value TEXT)")
        connection.execute("INSERT INTO previous VALUES('original target')")

    manager = BackupManager(str(source), backup_dir=str(tmp_path / "archives"))
    artifact = await manager.backup(kind)
    assert artifact is not None and artifact.backup_type is kind
    safety = restore_sqlite_backup(Path(artifact.path), target)
    assert safety is not None
    with sqlite3.connect(safety) as connection:
        assert connection.execute("SELECT value FROM previous").fetchall() == [("original target",)]
    with sqlite3.connect(target) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall() == expected_schema
        assert connection.execute("SELECT * FROM event_summary ORDER BY id").fetchall() == [(1, "first complete fact"), (2, "second complete fact")]
        connection.execute("INSERT INTO events VALUES(3,'subsequent fact')")
        assert connection.execute("SELECT event_id FROM audit ORDER BY event_id").fetchall() == [(1,), (2,), (3,)]
