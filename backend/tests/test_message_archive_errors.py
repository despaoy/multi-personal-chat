"""Required message writes fail explicitly before memory or delivery completion."""

from services import chat_answer

import asyncio
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from services.character_context import _TurnOutcome
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection
from services import turn_completion

from api import generate
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference import model_manager

FACTS = "虚构资料：陶弈的课程在周二09:00开始、11:00结束；尚未出发、尚未参加，暴雨时停课。请保留所有时间和否定。"
REPLY = "陶弈周二09:00上课、11:00结束；尚未出发和参加，暴雨时停课。"


def request():
    result = MessageRequest(message=FACTS, loraName="default", characterId="tsukiyashiro_kisaki",
        platform="web", adapter="fixture", senderId="fiction-owner", conversationId="fiction-course",
        sessionId="fiction-course", sourceMessageId="fiction-source", traceId="fiction-archive-trace")
    result._source_received_at = datetime.now(timezone.utc)
    return result


@pytest.fixture
def database(tmp_path):
    return SQLiteDB(tmp_path / "archive-control.sqlite")


@pytest.mark.asyncio
async def test_success_requires_real_receipt_and_keeps_scope(database):
    receipt = {"id": "stale-other-owner"}
    assert await generate._save_message(request(), REPLY, "fixture-model", "default", 0.1,
        database=database, saved_receipt=receipt, model_invoked=True) is True
    assert receipt["id"] != "stale-other-owner" and receipt["senderId"] == "fiction-owner"
    assert receipt["characterId"] == "tsukiyashiro_kisaki"
    assert receipt["message"] == FACTS and receipt["reply"] == REPLY
    rows = database.execute_sql("SELECT id,message,reply FROM messages WHERE id=:id", {"id": receipt["id"]})
    assert len(rows) == 1 and rows[0]["message"] == FACTS and rows[0]["reply"] == REPLY


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, False, 1, {}, {"id": ""}])
async def test_invalid_receipt_cannot_be_reported_as_saved(monkeypatch, result):
    receipt = {"id": "stale-other-owner"}
    store = SimpleNamespace(add_message=lambda record: result)
    with pytest.raises(HTTPException) as error:
        await generate._save_message(request(), REPLY, "fixture-model", "default", 0.1,
            database=store, saved_receipt=receipt, model_invoked=True)
    assert error.value.status_code == 503 and error.value.detail["stage"] == "message_archive"
    assert isinstance(error.value.__cause__, ValueError) and receipt == {}
    assert error.value.detail["modelInvoked"] is True and error.value.detail["messageSaved"] is None


@pytest.mark.asyncio
async def test_database_error_preserves_cause_without_private_error_text(monkeypatch, caplog):
    failure = OSError("fiction-private-write-error-text")
    def fail(record):
        raise failure
    counts = []
    monkeypatch.setattr(generate, "increment", counts.append)
    with pytest.raises(HTTPException) as error:
        await generate._save_message(request(), REPLY, "fixture-model", "default", 0.1,
            database=SimpleNamespace(add_message=fail), model_invoked=False)
    assert error.value.__cause__ is failure and error.value.detail["modelInvoked"] is False
    assert counts == ["db_write_failures"]
    assert "fiction-private-write-error-text" not in str(error.value.detail) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timed_out_or_cancelled_thread_is_not_claimed_rolled_back(database, monkeypatch, cancel):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    def delayed(record):
        entered.set()
        assert release.wait(2)
        try:
            return database.add_message(record)
        finally:
            done.set()
    monkeypatch.setattr(generate, "_DB_WRITE_TIMEOUT", 1 if cancel else 0.02)
    receipt = {"id": "stale"}
    operation = asyncio.create_task(generate._save_message(request(), REPLY, "fixture-model", "default", 0.1,
        database=SimpleNamespace(add_message=delayed), saved_receipt=receipt, model_invoked=True))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        if cancel:
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(HTTPException) as error:
                await operation
            assert error.value.status_code == 503 and isinstance(error.value.__cause__, TimeoutError)
            assert error.value.detail["messageSaved"] is None
        assert receipt == {} and database.execute_sql("SELECT id FROM messages") == []
    finally:
        release.set()
        assert await asyncio.to_thread(done.wait, 2)
    # The actual thread commits later; no synthetic receipt or retry was made.
    assert len(database.execute_sql("SELECT id FROM messages")) == 1 and receipt == {}


@pytest.fixture
async def generation(monkeypatch):
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat"),
        set_lora_adapter=lambda _: None,
        get_status=lambda: {"currentProvider": "openai_compat", "providers": {"openai_compat": {"modelName": "fixture-model"}}})
    connection = cloud_connection("fixture-model")
    manager.get_current_provider = lambda: connection
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    prepared = SimpleNamespace(character_id="tsukiyashiro_kisaki",
        compiled=SimpleNamespace(profile_context="", dynamic_context="", reference_context=""), history=())
    monkeypatch.setattr(generate, "_prepare_character_turn", AsyncMock(return_value=prepared))
    runtime = turn_completion.TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", lambda: runtime)
    service = SimpleNamespace(complete_turn=AsyncMock(return_value=_TurnOutcome(
        source_capture="recorded", memory_enrichment_status="skipped")))
    model = AsyncMock(return_value=(REPLY, False, {"modelInvoked": True}))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', model)
    yield SimpleNamespace(runtime=runtime, service=service, model=model)
    await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("invoked", [True, False])
@pytest.mark.parametrize("delivery", [False, True])
async def test_archive_failure_stops_fresh_generation_memory_and_delivery(generation, monkeypatch, invoked, delivery):
    failure = OSError("fiction-fresh-archive-error")
    def fail(record):
        raise failure
    store = SimpleNamespace(config={"useKnowledgeBase": False}, loras=[], add_message=fail)
    monkeypatch.setattr(generate, "db", store)
    counts = []
    monkeypatch.setattr(generate, "increment", counts.append)
    generation.model.return_value = (REPLY, False, {"modelInvoked": invoked, "answerMode": "generated" if invoked else "abstention"})
    context = {} if delivery else None
    with pytest.raises(HTTPException) as error:
        await generate._generate_reply_impl(request(), persist_message=True, record_invocation=False,
            character_service=generation.service, delivery_context=context)
    assert error.value.status_code == 503 and error.value.__cause__ is failure
    assert error.value.detail["stage"] == "message_archive" and error.value.detail["modelInvoked"] is invoked
    assert error.value.detail["messageSaved"] is None and counts == ["db_write_failures"]
    generation.service.complete_turn.assert_not_awaited()
    generation.model.assert_awaited_once()
    assert context is None or context == {}
    assert generation.runtime.active == generation.runtime.reserved == 0



@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [True, False])
async def test_success_and_internal_generation_keep_distinct_archive_semantics(generation, monkeypatch, database, persist):
    store = SimpleNamespace(config={"useKnowledgeBase": False}, loras=[], add_message=database.add_message)
    monkeypatch.setattr(generate, "db", store)
    result = await generate._generate_reply_impl(request(), persist_message=persist, record_invocation=False,
        character_service=generation.service)
    assert result.reply == REPLY and result.modelInvoked and result.warnings is None
    assert len(database.execute_sql("SELECT id FROM messages")) == int(persist)
    assert generation.service.complete_turn.await_count == int(persist)
    assert generation.runtime.active == generation.runtime.reserved == 0

