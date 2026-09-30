from datetime import datetime, timezone

import pytest

from db import memory_source
from db.database import SQLiteDB
from evaluation.memory_source_storage_probe import exercise

SCOPE = dict(character_id="role", platform="test", adapter="test", sender_id="user",
             conversation_type="private", conversation_id="room")


def test_linked_source_storage_contract(tmp_path):
    assert len(exercise(SQLiteDB(tmp_path / "sources.sqlite"))) == 15


@pytest.mark.parametrize("changes", [dict(character_id="*"), dict(adapter="narrative"),
    dict(conversation_type="unknown"), dict(sender_id=""), dict(conversation_id="*"),
    dict(conversation_type="group", conversation_id="")])
def test_invalid_or_branch_scope_is_not_invented(changes):
    with pytest.raises(ValueError):
        memory_source.source_scope(**(SCOPE | changes))


@pytest.mark.parametrize("limit", [0, 201, True, "20"])
def test_source_reads_are_bounded(tmp_path, limit):
    db = SQLiteDB(tmp_path / "bounds.sqlite")
    with pytest.raises(ValueError):
        db.list_memory_sources(**SCOPE, limit=limit)


def test_failed_capture_rolls_back_and_preserves_connection(tmp_path):
    db = SQLiteDB(tmp_path / "rollback.sqlite")
    with pytest.raises(ValueError):
        db.capture_memory_source(**SCOPE, source_message_id="source", body="原话", observed_at=datetime.now())
    assert db.capture_memory_source(**SCOPE, source_message_id="source", body="原话",
                                    observed_at=datetime.now(timezone.utc)) == "recorded"


def test_erase_rolls_back_source_purge_if_claim_delete_fails(tmp_path):
    db = SQLiteDB(tmp_path / "rollback-erase.sqlite")
    db.capture_memory_source(**SCOPE, source_message_id="source", body="原话",
                             observed_at=datetime.now(timezone.utc))
    target = db.append_character_memory_claim(**SCOPE, memory_type="user_fact", memory_key="key",
                                              content="fact", source_message_id="source")
    conn = db._get_connection()
    conn.execute("CREATE TRIGGER fail_erase BEFORE DELETE ON character_memories "
                 "BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    with pytest.raises(Exception, match="injected failure"):
        db.erase_character_memories(**SCOPE, memory_id=target["id"])
    assert db.list_memory_sources(**SCOPE)[0]["body"] == "原话"
    assert len(db.list_character_memory_claims(**SCOPE)) == 1
    assert conn.execute("SELECT revoked_before FROM memory_source_fences").fetchone()[0] == ""


def test_revoked_storage_retains_no_body_or_observation(tmp_path):
    db = SQLiteDB(tmp_path / "purge.sqlite")
    db.capture_memory_source(**SCOPE, source_message_id="source", body="不得残留的内容",
                             observed_at=datetime.now(timezone.utc))
    target = db.append_character_memory_claim(**SCOPE, memory_type="user_fact", memory_key="key",
                                              content="fact", source_message_id="source")
    db.erase_character_memories(**SCOPE, memory_id=target["id"])
    rows = db._get_connection().execute("SELECT body, observed_at, state FROM memory_sources").fetchall()
    assert [tuple(row) for row in rows] == [(None, None, "revoked")]
    assert not db._get_connection().execute("SELECT * FROM memory_source_links").fetchall()


def test_sqlite_schema_and_orm_source_tables_match(tmp_path):
    from sqlalchemy import create_engine, inspect

    from db.models import metadata

    db = SQLiteDB(tmp_path / "schema.sqlite")
    engine = create_engine("sqlite://")
    names = ("memory_sources", "memory_source_links", "memory_source_fences", "memory_source_terms")
    metadata.create_all(engine, tables=[metadata.tables[name] for name in names])
    orm = inspect(engine)
    for name in names:
        actual = db._get_connection().execute(f"PRAGMA table_info({name})").fetchall()
        assert {row["name"] for row in actual} == {column["name"] for column in orm.get_columns(name)}
        assert [row["name"] for row in actual if row["pk"]] == orm.get_pk_constraint(name)["constrained_columns"]
    engine.dispose()


@pytest.mark.parametrize("ids", ["source", [str(i) for i in range(201)], [""]])
def test_invalid_source_batch_rejected(tmp_path, ids):
    db = SQLiteDB(tmp_path / "invalid-batch.sqlite")
    with pytest.raises(ValueError):
        db.list_memory_sources(**SCOPE, source_message_ids=ids)


@pytest.mark.parametrize("ids,index", [(None, "idx_memory_sources_scope"),
                                       (["source", "missing"], "sqlite_autoindex_memory_sources_1")])
def test_source_read_uses_indexed_seek(tmp_path, ids, index):
    db = SQLiteDB(tmp_path / "query-plan.sqlite")
    plan = memory_source.read_plan(memory_source.source_scope(**SCOPE), source_message_ids=ids)
    sql, params = next(plan)
    rows = db._get_connection().execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    detail = " ".join(row["detail"] for row in rows)
    assert f"SEARCH memory_sources USING INDEX {index}" in detail
    assert "SCAN memory_sources" not in detail


def test_source_timestamp_normalizes_equal_instants(tmp_path):
    db = SQLiteDB(tmp_path / "source-time.sqlite")
    for stamp in ("2026-09-20T12:00:00+00:00", "2026-09-20T20:00:00+08:00"):
        assert db.capture_memory_source(**SCOPE, source_message_id="source", body="完整原话",
                                        observed_at=datetime.fromisoformat(stamp)) == "recorded"
    assert db.list_memory_sources(**SCOPE)[0]["observed_at"] == "2026-09-20T12:00:00.000000+00:00"
