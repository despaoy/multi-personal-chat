"""Training records use one decoder and reject invalid writes atomically."""

import json
from types import SimpleNamespace

import pytest

from db.database import SQLiteDB
from db.training_tasks import training_task_record


@pytest.fixture
def database(tmp_path):
    db = SQLiteDB(tmp_path / "training.db")
    db.add_training_task({"task_id": "task", "lora_name": "test", "config": {"steps": 3}})
    yield db
    db.close_connection()


@pytest.mark.parametrize("raw", ["broken", "[]", "null", "", '"text"'])
def test_all_training_reads_reject_corrupt_configuration(database, raw):
    database.get_connection().execute("UPDATE training_tasks SET config_json = ?", (raw,))
    for read in (
        lambda: database.get_training_task("task"),
        database.get_all_training_tasks,
        database.get_training_tasks,
        lambda: database.get_active_training_by_lora_name("test"),
    ):
        with pytest.raises(ValueError):
            read()


@pytest.mark.parametrize(
    "update",
    [
        {"config": {"bad": object()}},
        {"config": []},
        {"config_json": "broken"},
        {"config_json": "[]"},
    ],
)
def test_invalid_update_does_not_commit_other_fields(database, update):
    before = database.get_training_task("task")
    with pytest.raises((ValueError, TypeError)):
        database.update_training_task("task", {"status": "running", **update})
    assert database.get_training_task("task") == before


def test_explicit_serialized_config_takes_precedence(database):
    result = database.update_training_task("task", {"config": object(), "config_json": '{"steps": 5}'})
    assert result["config"] == {"steps": 5}


@pytest.mark.parametrize("config", [[], {"bad": object()}])
def test_invalid_save_preserves_existing_task(database, config):
    before = database.get_training_task("task")
    with pytest.raises((ValueError, TypeError)):
        database.save_training_task("task", {"status": "running", "config": config})
    assert database.get_training_task("task") == before


def test_nullable_config_remains_an_empty_object():
    assert training_task_record({"config_json": None})["config"] == {}


def test_postgres_uses_same_record_decoder(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/test_database")
    from db.pg_database import PgDatabase

    row = {"task_id": "task", "config_json": json.dumps({"steps": 3})}
    assert PgDatabase._normalize_training_task(SimpleNamespace(_mapping=row)) == training_task_record(row)
    with pytest.raises(ValueError):
        PgDatabase._normalize_training_task(SimpleNamespace(_mapping={"config_json": "broken"}))


@pytest.mark.asyncio
async def test_postgres_invalid_update_never_opens_transaction(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/test_database")
    from db.pg_database import PgDatabase

    db = object.__new__(PgDatabase)
    db.async_session = lambda: pytest.fail("Invalid update opened a transaction")
    with pytest.raises(TypeError):
        await db.update_training_task("task", {"status": "running", "config": {"bad": object()}})
