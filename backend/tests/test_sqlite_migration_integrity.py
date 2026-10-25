"""Legacy schema migration must either finish or retain its original records."""

import sqlite3

import pytest

from db.database import SQLiteDB


def _legacy_training(path, with_id):
    with sqlite3.connect(path) as connection:
        connection.execute(f"""
            CREATE TABLE training_tasks (
                {"id TEXT," if with_id else ""}
                task_id TEXT PRIMARY KEY, lora_name TEXT NOT NULL,
                status TEXT DEFAULT 'pending', progress REAL DEFAULT 0,
                error_message TEXT DEFAULT '', config_json TEXT DEFAULT '{{}}',
                created_at TEXT DEFAULT '', updated_at TEXT DEFAULT ''
            )
        """)
        connection.execute("INSERT INTO training_tasks (task_id, lora_name) VALUES ('task', 'test')")
        if with_id:
            connection.execute("UPDATE training_tasks SET id = task_id")


@pytest.mark.parametrize("with_id", [False, True])
def test_legacy_training_schema_is_migrated_once_without_data_loss(tmp_path, with_id):
    path = tmp_path / "legacy.db"
    _legacy_training(path, with_id)
    for _ in range(2):
        database = SQLiteDB(path)
        try:
            assert database.get_training_task("task")["lora_name"] == "test"
            columns = {
                row["name"]: row for row in database.get_connection().execute("PRAGMA table_info(training_tasks)")
            }
            assert columns["id"]["pk"] == 1
            assert columns["task_id"]["pk"] == 0
        finally:
            database.close_connection()


@pytest.mark.parametrize("table", ["training_tasks", "knowledge_chunks"])
def test_failed_schema_rebuild_propagates_and_restores_original_table(tmp_path, monkeypatch, table):
    path = tmp_path / "legacy.db"
    if table == "training_tasks":
        _legacy_training(path, False)
    else:
        with sqlite3.connect(path) as connection:
            connection.execute("""CREATE TABLE knowledge_chunks (
                id INTEGER PRIMARY KEY, documentId INTEGER, chunkIndex INTEGER,
                content TEXT, embedding BLOB, createdAt TEXT
            )""")
            connection.execute("INSERT INTO knowledge_chunks VALUES (1, 1, 0, 'evidence', X'01', 'now')")

    original_connect = sqlite3.connect
    opened = []
    failure = sqlite3.OperationalError("synthetic migration write failure")

    class FaultCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if f"INSERT INTO {table} (" in sql:
                raise failure
            return super().execute(sql, parameters)

    class FaultConnection(sqlite3.Connection):
        def cursor(self):
            return super().cursor(factory=FaultCursor)

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs, factory=FaultConnection)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            SQLiteDB(path)
        assert caught.value is failure
        connection = opened[0]
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = ?", (table + "_old",)).fetchone() is None
        columns = {row[1]: row for row in connection.execute(f"PRAGMA table_info({table})")}
        if table == "training_tasks":
            assert "id" not in columns
        else:
            assert columns["embedding"][2] == "BLOB"
    finally:
        for connection in opened:
            connection.close()


def test_required_index_failure_stops_startup(tmp_path, monkeypatch):
    original_connect = sqlite3.connect
    opened = []

    class FaultCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if "CREATE INDEX IF NOT EXISTS idx_training_tasks_status" in sql:
                raise sqlite3.OperationalError("synthetic index failure")
            return super().execute(sql, parameters)

    class FaultConnection(sqlite3.Connection):
        def cursor(self):
            return super().cursor(factory=FaultCursor)

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs, factory=FaultConnection)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        with pytest.raises(sqlite3.OperationalError, match="synthetic index failure"):
            SQLiteDB(tmp_path / "index.db")
    finally:
        for connection in opened:
            connection.close()


def test_invalid_legacy_session_migration_stops_and_rolls_back(tmp_path):
    path = tmp_path / "sessions.db"
    with sqlite3.connect(path) as connection:
        # A malformed old schema must not become an empty-success conversation view.
        connection.execute("CREATE TABLE session_settings (sessionId TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO session_settings VALUES ('retained')")
    with pytest.raises(sqlite3.OperationalError, match="sessionType"):
        SQLiteDB(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT sessionId FROM session_settings").fetchall() == [("retained",)]
        assert [row[1] for row in connection.execute("PRAGMA table_info(session_settings)")] == ["sessionId"]
