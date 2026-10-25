"""Required invocation accounting fails explicitly without hiding model errors."""

from services import chat_answer

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection

from api import generate
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference import model_manager

FACTS = "虚构资料：林绮周四14:00开会，15:00结束；尚未出发，下雨改为线上。请准确复述时间、否定和条件。"
REPLY = "林绮周四14:00至15:00开会，尚未出发；下雨改为线上。"


def request():
    return MessageRequest(message=FACTS, loraName="default", platform="web", adapter="fixture",
        conversationId="fiction-meeting", sessionId="fiction-meeting", traceId="fiction-invocation-trace")


@pytest.fixture
def database(tmp_path):
    return SQLiteDB(tmp_path / "invocations.sqlite")


@pytest.fixture
def generation(monkeypatch, database):
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat"),
        set_lora_adapter=Mock(), get_status=lambda: {"currentProvider": "openai_compat",
            "providers": {"openai_compat": {"modelName": "fixture-model"}}})
    connection = cloud_connection("fixture-model")
    manager.get_current_provider = lambda: connection
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "db", database)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    model = AsyncMock(return_value=(REPLY, False, {"modelInvoked": True}))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', model)
    return model


@pytest.mark.asyncio
async def test_success_records_real_sqlite_row_without_copying_private_text(database):
    await generate._record_model_invocation(request(), "fixture-model", "default", 0.2,
        used_rag=True, database=database)
    rows = database.execute_sql("SELECT * FROM model_invocations")
    assert len(rows) == 1 and rows[0]["traceId"] == "fiction-invocation-trace"
    assert rows[0]["usedRag"] == 1 and rows[0]["errorType"] == ""
    assert rows[0]["totalTokens"] is rows[0]["promptTokens"] is rows[0]["completionTokens"] is None
    assert rows[0]["usageSource"] == "unavailable"
    assert FACTS not in str(rows) and REPLY not in str(rows)


@pytest.mark.asyncio
async def test_record_failure_after_generation_stops_before_archive(generation, database, monkeypatch, caplog):
    failure = OSError("fiction-private-invocation-storage")
    writer = Mock(side_effect=failure)
    monkeypatch.setattr(database, "add_model_invocation", writer)
    save = AsyncMock()
    monkeypatch.setattr(generate, "_save_message", save)
    counts = []
    monkeypatch.setattr(generate, "increment", counts.append)
    with pytest.raises(HTTPException) as error:
        await generate._generate_reply_impl(request(), enable_rag=False, message_db=generate.db)
    assert error.value.status_code == 503 and error.value.__cause__ is failure
    detail = error.value.detail
    assert detail["stage"] == "model_invocation_record" and detail["modelInvoked"] is True
    assert detail["messageSaved"] is False and detail["invocationRecorded"] is None
    assert detail["generationErrorType"] is None and counts == ["db_write_failures"]
    assert "fiction-private-invocation-storage" not in str(detail) + caplog.text
    writer.assert_called_once()
    generation.assert_awaited_once()
    save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("record_fault", [False, True])
async def test_model_failure_keeps_primary_cause_and_reports_recording_failure(generation, database, monkeypatch, caplog, record_fault):
    primary = RuntimeError("fiction-private-model-error")
    generation.side_effect = primary
    storage = OSError("fiction-private-accounting-error")
    original = database.add_model_invocation
    writer = Mock(side_effect=storage if record_fault else original)
    monkeypatch.setattr(database, "add_model_invocation", writer)
    counts = []
    monkeypatch.setattr(generate, "increment", counts.append)
    with pytest.raises(HTTPException) as error:
        await generate._generate_reply_impl(request(), enable_rag=False, message_db=generate.db)
    assert error.value.status_code == (503 if record_fault else 500)
    assert error.value.__cause__ is primary
    assert database.execute_sql("SELECT id FROM messages") == []
    writer.assert_called_once()
    if record_fault:
        assert error.value.detail["generationErrorType"] == "RuntimeError"
        assert error.value.detail["modelInvoked"] is None and error.value.detail["invocationRecorded"] is None
        assert error.value.__context__.__cause__ is storage
        assert counts == ["model_failures", "db_write_failures"]
    else:
        rows = database.execute_sql("SELECT * FROM model_invocations")
        assert len(rows) == 1 and rows[0]["errorType"] == "RuntimeError" and rows[0]["completionTokens"] is None and rows[0]["usageSource"] == "unavailable"
        assert counts == ["model_failures"]
    assert "fiction-private-model-error" not in str(error.value.detail) + caplog.text
    assert "fiction-private-accounting-error" not in str(error.value.detail) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("invoked,record", [(True, True), (False, True), (True, False)])
async def test_success_deterministic_and_explicit_internal_modes_remain_distinct(generation, database, invoked, record):
    generation.return_value = (REPLY, False, {"modelInvoked": invoked, "answerMode": "generated" if invoked else "current_constraint"})
    result = await generate._generate_reply_impl(request(), enable_rag=False, record_invocation=record, message_db=generate.db)
    assert result.reply == REPLY and result.modelInvoked is invoked
    assert len(database.execute_sql("SELECT id FROM model_invocations")) == int(invoked and record)
    assert len(database.execute_sql("SELECT id FROM messages")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancellation_does_not_claim_thread_rollback_or_retry(database, monkeypatch, cancel):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def delayed(record):
        calls.append(record)
        entered.set()
        assert release.wait(2)
        try:
            return database.add_model_invocation(record)
        finally:
            done.set()
    monkeypatch.setattr(generate, "_DB_WRITE_TIMEOUT", 1 if cancel else 0.02)
    task = asyncio.create_task(generate._record_model_invocation(request(), "fixture-model", "default", 0.2,
        database=SimpleNamespace(add_model_invocation=delayed)))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(HTTPException) as error:
                await task
            assert error.value.status_code == 503 and isinstance(error.value.__cause__, TimeoutError)
            assert error.value.detail["invocationRecorded"] is None
        assert database.execute_sql("SELECT id FROM model_invocations") == []
    finally:
        release.set()
        assert await asyncio.to_thread(done.wait, 2)
    assert len(calls) == len(database.execute_sql("SELECT id FROM model_invocations")) == 1
