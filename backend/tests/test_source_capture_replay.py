"""Repeated authorized capture reuses a complete source index.

Real SQLite, SQL plans, source tokenizer, repositories, complete_turn and the
scheduler are exercised. Only the semantic model response is synthetic.

baseline_preflight(path) is separately callable against the current baseline;
it returns safe counters instead of weakening the desired regression assertions.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services import delivery_memory as dm
from services.character_context import CharacterContextService
from test_delivery_memory import generated

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from db import memory_source_search
from db.database import SQLiteDB
from db.memory_source import source_identity, source_scope

FIELDS = dict(character_id="fiction-role", platform="web", adapter="capture-replay",
              sender_id="fiction-owner", conversation_type="group", conversation_id="fiction-room")
BODY = "我本人长期喜欢深蓝色油墨，这是明确的个人偏好。"
SOURCE_ID = "fiction-source"
STAMP = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def sql_counts(statements):
    """Never return expanded SQLite SQL: it contains complete source bodies."""
    sql = [" ".join(statement.upper().split()) for statement in statements]
    return dict(total=len(sql),
        posting_inserts=sum(item.startswith("INSERT INTO MEMORY_SOURCE_TERMS ") for item in sql),
        source_upserts=sum(item.startswith("INSERT INTO MEMORY_SOURCES ") for item in sql),
        owner_fence_reads=sum(item.startswith("SELECT REVOKED_BEFORE FROM MEMORY_SOURCE_FENCES ") for item in sql))


def observe_capture(database, patch):
    """Wrap the real tokenizer and trace every real SQLite worker connection."""
    calls, statements, connections = [], [], {}
    actual_terms = memory_source_search.terms
    actual_connection = database._get_connection

    def count_terms(text):
        calls.append(text)
        return actual_terms(text)

    def traced_connection():
        connection = actual_connection()
        if id(connection) not in connections:
            connections[id(connection)] = connection
            connection.set_trace_callback(statements.append)
        return connection

    patch.setattr(memory_source_search, "terms", count_terms)
    patch.setattr(database, "_get_connection", traced_connection)
    return SimpleNamespace(calls=calls, sql=statements, connections=connections, actual_terms=actual_terms)


def close_observed(database, probe):
    # Repository asyncio.to_thread calls use SQLite's actual thread-local pool.
    # Scheduler shutdown precedes this cleanup, so no connection is still busy.
    main = database._get_connection()
    for connection in tuple(probe.connections.values()):
        connection.set_trace_callback(None)
        if connection is not main:
            connection.close()
    database.close_connection()


@pytest.fixture
def capture_store(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path / "capture.sqlite")
    probe = observe_capture(database, monkeypatch)
    try:
        yield database, probe
    finally:
        close_observed(database, probe)


def capture(database, *, source_id=SOURCE_ID, body=BODY, stamp=STAMP, fields=FIELDS):
    return database.capture_memory_source(**fields, source_message_id=source_id,
                                          body=body, observed_at=stamp)


def postings(database, *, source_id=SOURCE_ID, fields=FIELDS):
    identity = source_identity(source_scope(**fields), source_id)
    return tuple(row[0] for row in database._get_connection().execute(
        "SELECT term FROM memory_source_terms WHERE scope_key = :scope_key AND source_key = :source_key ORDER BY term",
        identity))


def test_identical_source_replay_preserves_speech_without_retokenizing_or_posting_writes(capture_store):
    database, probe = capture_store
    expected_terms = probe.actual_terms(BODY)
    assert expected_terms
    assert capture(database) == "recorded"
    first = sql_counts(probe.sql)
    assert probe.calls.count(BODY) == 1 and first["posting_inserts"] > 0
    assert postings(database) == expected_terms
    original = database.list_memory_sources(**FIELDS)

    start = len(probe.sql)
    assert capture(database) == "recorded"
    replay = sql_counts(probe.sql[start:])
    assert probe.calls.count(BODY) == 1
    assert replay["posting_inserts"] == 0
    assert replay["owner_fence_reads"] == 1  # Reuse still requires current owner authority.
    assert database.list_memory_sources(**FIELDS) == original == [dict(
        source_message_id=SOURCE_ID, body=BODY, observed_at=STAMP.isoformat(timespec="microseconds"))]
    assert postings(database) == expected_terms


@pytest.mark.asyncio
async def test_complete_turn_and_scheduler_both_capture_but_index_original_only_once(
        capture_store, monkeypatch, no_live_context_review):
    database, probe = capture_store
    _, _, stored = generated(database)
    snapshot, turn = dm.thaw_completion(stored["context"])
    source_id = stored["context"]["request"]["sourceMessageId"]
    body = turn.message
    repo = DatabaseCharacterMemoryRepository(database)
    assert await repo.list_memory_records(snapshot.character_id, snapshot.user_scope) == []
    real_capture = repo.capture_source
    capture_spy = AsyncMock(wraps=real_capture)
    monkeypatch.setattr(repo, "capture_source", capture_spy)
    model_calls = []

    async def synthetic_complete(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["current_user_message"] == body
        model_calls.append(payload)
        return json.dumps({"memories": [dict(attributed_to="user", operation="ADD", kind="like",
            value="深蓝色油墨", evidence=body, confidence=.99)]}, ensure_ascii=False)

    completion = SimpleNamespace(complete=synthetic_complete, close=AsyncMock())
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(
        enabled=True, base_url="http://fixture.invalid/v1", model="fiction-local-writer", idle_seconds=0),
        completion=completion)
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", lambda: scheduler)
    service = CharacterContextService(object(), repo, object())
    receipt = asyncio.get_running_loop().create_future()
    start = len(probe.sql)
    try:
        outcome = await service.complete_turn(snapshot, turn,
            source_message_id=source_id, memory_receipt=receipt)
        assert outcome.source_capture == "recorded" and outcome.memory_enrichment_scheduled
        assert not outcome.failed_writes
        assert await scheduler.flush_memory(timeout=3)
        result = await asyncio.wait_for(asyncio.shield(receipt), timeout=1)
        assert result["status"] == "saved" and result["persisted"] == 1
        assert result["source_capture"] == "recorded"

        # Retain both real authorization/capture calls at their business boundaries.
        assert capture_spy.await_count == 2
        for call in capture_spy.await_args_list:
            assert call.args == (snapshot.character_id, snapshot.user_scope)
            assert call.kwargs == dict(source_message_id=source_id, body=body, observed_at=snapshot.received_at)
        assert probe.calls.count(body) == 1
        trace = sql_counts(probe.sql[start:])
        assert trace["posting_inserts"] == (len(probe.actual_terms(body)) + 99) // 100
        assert trace["owner_fence_reads"] >= 2
        assert len(model_calls) == 1

        assert await repo.list_sources(snapshot.character_id, snapshot.user_scope) == [dict(
            source_message_id=source_id, body=body,
            observed_at=snapshot.received_at.isoformat(timespec="microseconds"))]
        records = await repo.list_memory_records(snapshot.character_id, snapshot.user_scope)
        assert len(records) == 1 and records[0]["source_message_id"] == source_id
        assert "深蓝色油墨" in records[0]["content"]
    finally:
        await scheduler.shutdown(timeout=3)
    completion.close.assert_awaited_once()


def test_pre013_recorded_source_without_postings_backfills_then_becomes_searchable(capture_store):
    database, probe = capture_store
    expected_terms = probe.actual_terms(BODY)
    assert capture(database) == "recorded"
    identity = source_identity(source_scope(**FIELDS), SOURCE_ID)
    connection = database._get_connection()
    # 013 explicitly created the index without backfilling existing source rows.
    connection.execute("DELETE FROM memory_source_terms WHERE source_key = :source_key", identity)
    connection.commit()
    assert postings(database) == ()
    assert database.list_memory_sources(**FIELDS)[0]["body"] == BODY
    assert database.search_memory_sources(**FIELDS, query="深蓝色") == []

    before_calls, start = probe.calls.count(BODY), len(probe.sql)
    assert capture(database) == "recorded"
    assert probe.calls.count(BODY) == before_calls + 1
    assert sql_counts(probe.sql[start:])["posting_inserts"] > 0
    assert postings(database) == expected_terms
    matches = database.search_memory_sources(**FIELDS, query="深蓝色")
    assert len(matches) == 1 and matches[0]["source_message_id"] == SOURCE_ID and matches[0]["body"] == BODY

    start = len(probe.sql)
    assert capture(database) == "recorded"
    assert probe.calls.count(BODY) == before_calls + 1
    assert sql_counts(probe.sql[start:])["posting_inserts"] == 0


def test_legal_source_with_zero_terms_remains_recorded_and_replayable(capture_store):
    database, probe = capture_store
    body = "甲"  # A legal complete utterance with no bigram or Latin term.
    assert probe.actual_terms(body) == ()
    start = len(probe.sql)
    assert capture(database, body=body) == "recorded"
    assert capture(database, body=body) == "recorded"
    assert sql_counts(probe.sql[start:])["posting_inserts"] == 0
    assert postings(database) == ()
    assert database.list_memory_sources(**FIELDS) == [dict(source_message_id=SOURCE_ID,
        body=body, observed_at=STAMP.isoformat(timespec="microseconds"))]
    assert database.search_memory_sources(**FIELDS, query="甲") == []


def test_replay_fast_path_keeps_body_clock_revocation_and_owner_fence_checks(capture_store):
    database, probe = capture_store
    assert capture(database) == "recorded"
    indexed = postings(database)
    assert indexed
    start, calls = len(probe.sql), len(probe.calls)
    assert capture(database, body="我本人长期喜欢另一种颜色。") == "conflict"
    assert capture(database, stamp=STAMP + timedelta(seconds=1)) == "conflict"
    assert probe.calls[calls:] == []
    assert postings(database) == indexed
    assert database.list_memory_sources(**FIELDS)[0]["body"] == BODY

    assert database.erase_unlinked_memory_sources(**FIELDS, source_message_ids=(SOURCE_ID,)) == 1
    assert capture(database) == "revoked"
    assert capture(database, stamp=datetime.now(timezone.utc) + timedelta(seconds=1)) == "revoked"
    assert postings(database) == () and database.list_memory_sources(**FIELDS) == []

    # A sibling conversation retains recorded text, but the same owner's clear
    # watermark must reject even an exact-body replay before any fast return.
    sibling = dict(FIELDS, conversation_id="fiction-sibling-room")
    assert capture(database, source_id="sibling", fields=sibling) == "recorded"
    calls = len(probe.calls)
    database.clear_character_memories(**FIELDS)
    stale_start = len(probe.sql)
    assert capture(database, source_id="sibling", fields=sibling) == "stale"
    assert len(probe.calls) == calls
    assert sql_counts(probe.sql[stale_start:])["posting_inserts"] == 0
    assert database.list_memory_sources(**sibling) == []
    assert sql_counts(probe.sql[start:])["owner_fence_reads"] >= 5


def baseline_preflight(workdir):
    """Callable, never auto-run: return real first/replay work on a fresh DB.

    Baseline is expected to tokenize and issue posting INSERTs twice; after the
    fix the second capture keeps its fence read but does neither. All source
    text and expanded trace statements stay local to this function.
    """
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "capture-replay-preflight.sqlite"
    if path.exists():
        raise ValueError("preflight requires a fresh work directory")
    database = SQLiteDB(path)
    with pytest.MonkeyPatch.context() as patch:
        probe = observe_capture(database, patch)
        try:
            captures = []
            for _ in range(2):
                start_sql, start_calls = len(probe.sql), probe.calls.count(BODY)
                status = capture(database)
                captures.append(dict(status=status,
                    tokenizations=probe.calls.count(BODY) - start_calls,
                    **sql_counts(probe.sql[start_sql:])))
            sources = database.list_memory_sources(**FIELDS)
            assert len(sources) == 1 and sources[0]["body"] == BODY
            assert postings(database) == probe.actual_terms(BODY)
            return dict(captures=captures, source_rows=len(sources), posting_rows=len(postings(database)),
                        exact_body_and_clock=sources[0] == dict(source_message_id=SOURCE_ID,
                            body=BODY, observed_at=STAMP.isoformat(timespec="microseconds")),
                        model_calls=0)
        finally:
            close_observed(database, probe)


def test_second_posting_batch_failure_rolls_back_source_and_whole_index_then_retries(capture_store):
    import sqlite3

    database, probe = capture_store
    body = " ".join(f"capturetoken{index:03d}" for index in range(150))
    expected_terms = probe.actual_terms(body)
    assert len(expected_terms) == 150 and expected_terms[100] == "capturetoken100"
    connection = database._get_connection()
    # The trigger fires on the first term of the real second 100-term batch.
    # Its distinct failure message proves all first-batch postings were present
    # in this same transaction before SQLite aborted the second INSERT.
    connection.execute("""
        CREATE TRIGGER reject_second_capture_posting_batch
        BEFORE INSERT ON memory_source_terms
        WHEN NEW.term = 'capturetoken100'
        BEGIN
            SELECT CASE WHEN (
                SELECT COUNT(*) FROM memory_source_terms WHERE source_key = NEW.source_key
            ) = 100
            THEN RAISE(ABORT, 'fixture second posting batch')
            ELSE RAISE(ABORT, 'fixture invalid posting batch premise') END;
        END
    """)
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="^fixture second posting batch$"):
        capture(database, body=body)
    assert probe.calls.count(body) == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_sources").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM memory_source_terms").fetchone()[0] == 0
    assert database.list_memory_sources(**FIELDS) == []

    connection.execute("DROP TRIGGER reject_second_capture_posting_batch")
    connection.commit()
    assert capture(database, body=body) == "recorded"
    assert probe.calls.count(body) == 2
    assert postings(database) == expected_terms
    assert database.list_memory_sources(**FIELDS) == [dict(source_message_id=SOURCE_ID,
        body=body, observed_at=STAMP.isoformat(timespec="microseconds"))]

    start = len(probe.sql)
    assert capture(database, body=body) == "recorded"
    assert probe.calls.count(body) == 2
    assert sql_counts(probe.sql[start:])["posting_inserts"] == 0
    assert postings(database) == expected_terms
