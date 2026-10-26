"""Each supported backup artifact must preserve SQLite schema and behavior."""

from app.runtime import RuntimeContainer
from infra.concurrency_control import inference_runtime
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


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', [BackupType.FULL, BackupType.INCREMENTAL])
@pytest.mark.parametrize('stage', ['missing', 'compression', 'destination'])
async def test_backup_failure_preserves_cause_and_releases_owned_files(tmp_path, monkeypatch, kind, stage):
    from api import enhanced
    from fastapi import HTTPException
    from infra import backup_manager as module
    source = tmp_path / 'source.sqlite'
    if stage != 'missing':
        with sqlite3.connect(source) as connection:
            connection.execute('CREATE TABLE evidence(value TEXT NOT NULL)')
            connection.execute("INSERT INTO evidence VALUES('complete synthetic source')")
    manager = BackupManager(str(source), backup_dir=str(tmp_path / 'archives'))
    runtime = RuntimeContainer(db=None, is_pg_mode=lambda: False, inference_runtime=inference_runtime)
    runtime.backup_mgr = manager
    failure = OSError('synthetic compression unavailable') if stage == 'compression' else sqlite3.OperationalError('synthetic destination unavailable')
    connections = []
    real_connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            return super().close()

    def connect(*args, **kwargs):
        if connections:
            raise failure
        connection = real_connect(*args, **kwargs, factory=TrackedConnection)
        connections.append(connection)
        return connection
    if stage == 'destination':
        monkeypatch.setattr(sqlite3, 'connect', connect)
    if stage == 'compression':

        def compress(*args, **kwargs):
            raise failure
        monkeypatch.setattr(module.gzip, 'open', compress)
    with pytest.raises(HTTPException) as caught:
        await enhanced.create_backup(kind.value, {'role': 'admin'}, runtime=runtime)
    assert caught.value.status_code == 500
    if stage == 'missing':
        assert isinstance(caught.value.__cause__, FileNotFoundError)
    else:
        assert caught.value.__cause__ is failure
    assert all((connection.closed for connection in connections))
    assert not list((tmp_path / 'archives').rglob('*'))
    assert manager.get_backup_stats()['total_backups'] == 0


@pytest.mark.asyncio
async def test_two_same_second_backups_preserve_both_versions_and_existing_names(tmp_path, monkeypatch):
    import datetime
    import gzip
    import shutil
    from infra import backup_manager as module

    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 19, 1, 2, 3, tzinfo=tz)

    monkeypatch.setattr(module, "datetime", Clock)
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE evidence(value TEXT)")
        connection.execute("INSERT INTO evidence VALUES('first complete state')")
    manager = BackupManager(str(source), backup_dir=str(tmp_path / "archives"))
    first = await manager.backup()
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE evidence SET value='second complete state'")
    second = await manager.backup()
    assert first.path != second.path
    for index, (artifact, expected) in enumerate([(first, "first complete state"), (second, "second complete state")]):
        target = tmp_path / f"readback{index}.sqlite"
        with gzip.open(artifact.path, "rb") as compressed:
            target.write_bytes(compressed.read())
        with sqlite3.connect(target) as connection:
            assert connection.execute("SELECT value FROM evidence").fetchall() == [(expected,)]
    legacy = tmp_path / "archives/source_20200101_010203_full.bak.gz"
    shutil.copyfile(first.path, legacy)
    listed = manager.list_backups()
    assert len(listed) == 3 and all(item.db_name == "source" for item in listed)
    assert {item.filename for item in listed} == {first.filename, second.filename, legacy.name}


@pytest.mark.asyncio
async def test_incomplete_backup_is_not_published(tmp_path, monkeypatch):
    import asyncio
    from threading import Event
    from infra import backup_manager as module

    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE evidence(value TEXT)")
        connection.execute("INSERT INTO evidence VALUES('complete synthetic state')")
    manager = BackupManager(str(source), backup_dir=str(tmp_path / "archives"))
    started, release = Event(), Event()
    copy = module.shutil.copyfileobj
    def blocked_copy(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return copy(*args, **kwargs)
    monkeypatch.setattr(module.shutil, "copyfileobj", blocked_copy)
    task = asyncio.create_task(manager.backup())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        assert manager.list_backups() == []
        assert not list((tmp_path / "archives").glob("*.bak.gz"))
    finally:
        release.set()
    artifact = await task
    assert Path(artifact.path).is_file() and len(manager.list_backups()) == 1


@pytest.mark.asyncio
async def test_unknown_backup_type_fails_before_creating_files(tmp_path):
    manager = BackupManager(str(tmp_path / "source.sqlite"), backup_dir=str(tmp_path / "archives"))
    with pytest.raises(ValueError):
        await manager.backup("unknown")
    assert not (tmp_path / "archives").exists()


@pytest.mark.asyncio
async def test_backup_list_io_failure_cannot_become_partial_success(tmp_path, monkeypatch):
    from api import enhanced
    source = tmp_path / 'source.sqlite'
    with sqlite3.connect(source) as connection:
        connection.execute('CREATE TABLE evidence(value TEXT)')
        connection.execute("INSERT INTO evidence VALUES('complete synthetic source')")
    manager = BackupManager(str(source), backup_dir=str(tmp_path / 'archives'))
    full = await manager.backup(BackupType.FULL)
    await manager.backup(BackupType.INCREMENTAL)
    failure = OSError('synthetic backup metadata unavailable')
    real_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path == Path(full.path):
            raise failure
        return real_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    runtime = RuntimeContainer(db=None, is_pg_mode=lambda: False, inference_runtime=inference_runtime)
    runtime.backup_mgr = manager
    with pytest.raises(OSError) as caught:
        await enhanced.list_backups({'role': 'admin'}, runtime=runtime)
    assert caught.value is failure
