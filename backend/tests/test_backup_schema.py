"""Each supported backup artifact must preserve SQLite schema and behavior."""

from app.runtime import RuntimeContainer
from infra.concurrency_control import inference_runtime
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from infra.backup_manager import BackupManager, BackupType
from scripts.restore_sqlite_backup import restore_sqlite_backup


@pytest.mark.asyncio
async def test_restore_keeps_chat_erasure_receipts_and_purges_older_session_rows(tmp_path):
    from character.library import CharacterSettings
    from db.database import SQLiteDB
    from repositories.character_chat import CharacterChatRepository
    from repositories.character_chat_proactive import CharacterChatProactiveRepository

    path = tmp_path / "chat.sqlite"
    db = SQLiteDB(path)
    try:
        chats = CharacterChatRepository(db)
        character = chats.characters.create("1", CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
        erased = chats.create("1", character["id"], "erased")
        kept = chats.create("1", character["id"], "kept")
        turn = chats.commit_turn(erased, "request", "deleted utterance", "deleted reply", "fixture", enqueue_enrichment=True)
        erased = chats.get("1", erased["id"])
        copy = chats.create("1", character["id"], "explicitly retained copy", source_session_id=erased["id"],
                            source_revision=erased["revision"], source_turn_id=turn["id"])
        proactive = CharacterChatProactiveRepository(db)
        proactive.set_policy("1", erased["id"], 0, enabled=True, cadence_hours=24, timezone="UTC",
                             quiet_start=None, quiet_end=None)
        proactive.schedule("1", erased["id"], 1, "old-job", "event", "deleted topic", 101, now=100)
        db.execute_sql("DROP TABLE character_chat_erasure_receipts")
        before = await BackupManager(str(path), backup_dir=str(tmp_path / "archives")).backup()
    finally:
        db.close_connection()
    db = SQLiteDB(path)
    try:
        CharacterChatRepository(db).delete_sessions("1", [{"id": erased["id"], "revision": erased["revision"]}])
    finally:
        db.close_connection()

    # Replaying the same old backup keeps both the deletion and its receipt.
    for _ in range(2):
        restore_sqlite_backup(Path(before.path), path)
        with closing(sqlite3.connect(path)) as connection:
            assert {row[0] for row in connection.execute("SELECT id FROM character_chat_sessions")} == {kept["id"], copy["id"]}
            assert connection.execute("SELECT session_id FROM character_chat_erasure_receipts").fetchall() == [(erased["id"],)]
            assert connection.execute("SELECT * FROM character_chat_turns WHERE session_id=?", (erased["id"],)).fetchall() == []
            assert connection.execute("SELECT * FROM character_chat_enrichment_jobs WHERE session_id=?", (erased["id"],)).fetchall() == []
            assert connection.execute("SELECT * FROM character_chat_proactive_jobs").fetchall() == []
            assert connection.execute("SELECT * FROM character_chat_proactive_policies").fetchall() == []

    # Recovery into a new file can use a surviving, newer SQLite receipt source.
    recovered = tmp_path / "recovered.sqlite"
    restore_sqlite_backup(Path(before.path), recovered, deletion_records_from=path)
    with closing(sqlite3.connect(recovered)) as connection:
        assert connection.execute("SELECT id FROM character_chat_sessions WHERE id=?", (erased["id"],)).fetchone() is None


@pytest.mark.asyncio
async def test_restore_does_not_regrant_old_outreach_consent(tmp_path):
    from character.library import CharacterSettings
    from db.database import SQLiteDB
    from repositories.character_chat_proactive import CharacterChatProactiveRepository

    path = tmp_path / "restore-consent.sqlite"
    db = SQLiteDB(path)
    try:
        repo = CharacterChatProactiveRepository(db)
        character = repo.chats.characters.create("1", CharacterSettings(name="fixture", persona="fixture", greeting="hi"))
        session = repo.chats.create("1", character["id"], "old consent")
        identity = session["id"]
        settings = dict(cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
        repo.set_policy("1", identity, 0, enabled=True, inactivity_hours=72, **settings)
        repo.chats.commit_turn(session, "activity", "hello", "hi", "fixture")
        repo.schedule("1", identity, 1, "delivered", "event", "已有正式记录", 101, now=100)
        repo.schedule("1", identity, 1, "waiting", "event", "原待发话题", 300, now=100)
        assert repo.deliver_due(now=101) == 1
        backup = await BackupManager(str(path), backup_dir=str(tmp_path / "archives")).backup()
        repo.set_policy("1", identity, 1, enabled=False, **settings)
    finally:
        db.close_connection()
    restore_sqlite_backup(Path(backup.path), path)
    db = SQLiteDB(path)
    try:
        repo = CharacterChatProactiveRepository(db)
        view = repo.view("1", identity)
        assert not view["policy"]["enabled"]
        assert [job["id"] for job in view["jobs"]] == ["delivered"]
        assert not repo.has_due(4102444800)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_restore_refuses_uncheckpointed_wal_without_replacing_target(tmp_path):
    source, target = tmp_path / "source.sqlite", tmp_path / "target.sqlite"
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("CREATE TABLE archived (value TEXT)")
    backup = await BackupManager(str(source), backup_dir=str(tmp_path / "archives")).backup()

    with closing(sqlite3.connect(target)) as connection, connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE live (value TEXT)")
        connection.execute("INSERT INTO live VALUES ('newer target')")
        connection.commit()
        assert target.with_name(target.name + "-wal").stat().st_size > 0
        with pytest.raises(RuntimeError, match="uncheckpointed WAL"):
            restore_sqlite_backup(Path(backup.path), target)
        assert connection.execute("SELECT value FROM live").fetchall() == [("newer target",)]
    assert not list(tmp_path.glob("*.safety-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [BackupType.FULL, BackupType.INCREMENTAL])
async def test_backup_and_offline_restore_preserve_complete_schema_and_triggers(tmp_path, kind):
    source, target = tmp_path / "source.sqlite", tmp_path / "target.sqlite"
    with closing(sqlite3.connect(source)) as connection, connection:
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
    with closing(sqlite3.connect(target)) as connection, connection:
        connection.execute("CREATE TABLE previous(value TEXT)")
        connection.execute("INSERT INTO previous VALUES('original target')")

    manager = BackupManager(str(source), backup_dir=str(tmp_path / "archives"))
    artifact = await manager.backup(kind)
    assert artifact is not None and artifact.backup_type is kind
    safety = restore_sqlite_backup(Path(artifact.path), target)
    assert safety is not None
    with closing(sqlite3.connect(safety)) as connection, connection:
        assert connection.execute("SELECT value FROM previous").fetchall() == [("original target",)]
    with closing(sqlite3.connect(target)) as connection, connection:
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
        with closing(sqlite3.connect(source)) as connection, connection:
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
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("CREATE TABLE evidence(value TEXT)")
        connection.execute("INSERT INTO evidence VALUES('first complete state')")
    manager = BackupManager(str(source), backup_dir=str(tmp_path / "archives"))
    first = await manager.backup()
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("UPDATE evidence SET value='second complete state'")
    second = await manager.backup()
    assert first.path != second.path
    for index, (artifact, expected) in enumerate([(first, "first complete state"), (second, "second complete state")]):
        target = tmp_path / f"readback{index}.sqlite"
        with gzip.open(artifact.path, "rb") as compressed:
            target.write_bytes(compressed.read())
        with closing(sqlite3.connect(target)) as connection, connection:
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
    with closing(sqlite3.connect(source)) as connection, connection:
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
    with closing(sqlite3.connect(source)) as connection, connection:
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
