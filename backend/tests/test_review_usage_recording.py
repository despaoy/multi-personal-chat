"""Completed reviews remain accounted for when later preparation fails."""

from services import chat_answer

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from test_compiled_cloud_transport import installed as installed

from api import generate
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference.request_context import get_request_context
from inference.usage import has_pending_usage


@pytest.fixture
def review_usage(installed, monkeypatch, tmp_path):
    database = SQLiteDB(tmp_path / "review-usage.sqlite")
    database.update_config(installed.config)
    original_writer = database.add_model_invocation
    writer = Mock(wraps=original_writer)
    monkeypatch.setattr(database, "add_model_invocation", writer)
    monkeypatch.setattr(generate, "db", database)

    async def consume():
        provider = get_request_context().provider
        return await provider.async_complete(
            [{"role": "system", "content": "Return a structured review of complete synthetic facts."},
             {"role": "user", "content": "Synthetic facts: A is approved, B is not approved."}],
            temperature=0, max_tokens=128, top_p=1,
        )

    async def prepare(turn, role):
        raw, _elapsed = await consume()
        return json.loads(raw)

    service = SimpleNamespace(prepare_turn=prepare)
    request = MessageRequest(characterId="tsukiyashiro_kisaki", loraName="default",
        message="Summarize the complete supplied synthetic facts A and B.", traceId="synthetic-review-usage")

    async def run(record=True):
        return await generate._generate_reply_impl(request, persist_message=False, enable_rag=False,
            record_invocation=record, character_service=service, message_db=database)

    def rows():
        return database.execute_sql("SELECT * FROM model_invocations")

    return SimpleNamespace(database=database, writer=writer, original_writer=original_writer,
        consume=consume, service=service, run=run, rows=rows, installed=installed)


@pytest.mark.asyncio
@pytest.mark.parametrize("record", [True, False])
async def test_preparation_parse_failure_keeps_exact_review_usage_and_original_error(review_usage, record):
    rig = review_usage
    with pytest.raises(HTTPException) as caught:
        await rig.run(record)
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, json.JSONDecodeError)
    assert len(rig.installed.calls) == 1 and rig.writer.call_count == int(record)
    rows = rig.rows()
    assert len(rows) == int(record)
    if record:
        assert (rows[0]["promptTokens"], rows[0]["completionTokens"], rows[0]["totalTokens"]) == (100, 20, 120)
        assert rows[0]["modelName"] == "fixture-model" and rows[0]["usageSource"] == "provider"
        assert rows[0]["errorType"] == "JSONDecodeError"
    assert rig.database.execute_sql("SELECT id FROM messages") == []
    assert get_request_context() is None and not has_pending_usage()


@pytest.mark.asyncio
async def test_review_only_answer_is_accounted_without_claiming_main_generation(review_usage, monkeypatch):
    rig = review_usage
    async def prepare(turn, role):
        await rig.consume()
        return SimpleNamespace(character_id=role, memory_operation_receipt=None)
    rig.service.prepare_turn = prepare
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', AsyncMock(return_value=(
        "The currently stored synthetic facts were read.", False,
        {"modelInvoked": False, "answerMode": "memory_lookup"},
    )))
    result = await rig.run()
    assert not result.modelInvoked and result.model == "memory/lookup"
    assert len(rig.installed.calls) == rig.writer.call_count == 1
    row, = rig.rows()
    assert row["totalTokens"] == 120 and row["modelName"] == "fixture-model" and row["errorType"] == ""


@pytest.mark.asyncio
async def test_main_failure_does_not_record_completed_reviews_twice(review_usage):
    rig = review_usage
    async def prepare(turn, role):
        await rig.consume()
        rig.installed.state.finish = "length"
        return SimpleNamespace(character_id=role, memory_operation_receipt=None)
    rig.service.prepare_turn = prepare
    with pytest.raises(HTTPException) as caught:
        await rig.run()
    assert isinstance(caught.value.__cause__, ValueError)
    assert len(rig.installed.calls) == 2 and rig.writer.call_count == 1
    row, = rig.rows()
    assert row["totalTokens"] == 240 and row["errorType"] == "ValueError"


@pytest.mark.asyncio
async def test_storage_failure_preserves_primary_http_receipt_and_is_not_retried(review_usage):
    rig = review_usage
    detail = {"stage": "synthetic_review", "operation_receipt": {"status": "completed"}}
    async def prepare(turn, role):
        await rig.consume()
        raise HTTPException(503, detail=detail)
    rig.service.prepare_turn = prepare
    rig.writer.side_effect = OSError("synthetic-private-storage-detail")
    with pytest.raises(HTTPException) as caught:
        await rig.run()
    assert caught.value.status_code == 503 and caught.value.__cause__.detail == detail
    assert caught.value.detail["stage"] == "model_invocation_record"
    assert caught.value.detail["requestError"] == detail
    assert caught.value.detail["invocationRecorded"] is None
    assert rig.writer.call_count == 1 and rig.rows() == []
    assert "synthetic-private" not in str(caught.value.detail)


@pytest.mark.asyncio
async def test_timed_out_record_can_commit_later_but_is_never_written_twice(review_usage, monkeypatch):
    rig = review_usage
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    def delayed(record):
        entered.set()
        assert release.wait(2)
        try:
            return rig.original_writer(record)
        finally:
            done.set()
    rig.writer.side_effect = delayed
    monkeypatch.setattr(generate, "_DB_WRITE_TIMEOUT", .01)
    try:
        with pytest.raises(HTTPException) as caught:
            await rig.run()
        assert await asyncio.to_thread(entered.wait, 1)
        assert caught.value.detail["invocationRecorded"] is None
        assert isinstance(caught.value.__cause__.__cause__, json.JSONDecodeError)
        assert rig.writer.call_count == 1 and rig.rows() == []
    finally:
        release.set()
        assert await asyncio.to_thread(done.wait, 2)
    row, = rig.rows()
    assert rig.writer.call_count == 1 and row["totalTokens"] == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_fault", [False, True])
async def test_cancellation_after_review_keeps_its_semantics_and_does_not_retry_record(review_usage, storage_fault):
    rig = review_usage
    async def prepare(turn, role):
        await rig.consume()
        raise asyncio.CancelledError()
    rig.service.prepare_turn = prepare
    if storage_fault:
        rig.writer.side_effect = OSError("synthetic-cancel-storage-error")
    with pytest.raises(asyncio.CancelledError) as caught:
        await rig.run()
    assert rig.writer.call_count == 1
    if storage_fault:
        assert isinstance(caught.value.__cause__, HTTPException) and rig.rows() == []
    else:
        row, = rig.rows()
        assert row["errorType"] == "CancelledError" and row["totalTokens"] == 120
    assert get_request_context() is None and not has_pending_usage()


@pytest.mark.asyncio
async def test_rag_review_http_failure_keeps_its_error_and_usage(review_usage, monkeypatch):
    rig = review_usage
    rig.service.prepare_turn = AsyncMock(return_value=SimpleNamespace(character_id="tsukiyashiro_kisaki", memory_operation_receipt=None))
    detail = {"stage": "synthetic_rag_review", "message": "Evidence review failed"}
    failure = ValueError("Synthetic evidence review failure")
    async def retrieve(*args, **kwargs):
        await rig.consume()
        raise chat_answer.ChatRetrievalError(detail) from failure
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', retrieve)
    with pytest.raises(HTTPException) as caught:
        await rig.run()
    assert caught.value.status_code == 503 and caught.value.detail == detail
    row, = rig.rows()
    assert row["totalTokens"] == 120 and row["errorType"] == "ValueError"
    assert caught.value.__cause__ is failure
    assert rig.writer.call_count == len(rig.installed.calls) == 1


@pytest.mark.asyncio
async def test_pre_model_rejection_creates_no_invented_usage(review_usage):
    rig = review_usage
    rig.service.prepare_turn = AsyncMock(side_effect=HTTPException(503, "Synthetic unavailable source"))
    with pytest.raises(HTTPException):
        await rig.run()
    assert rig.rows() == [] and rig.writer.call_count == 0 and rig.installed.calls == []


@pytest.mark.asyncio
async def test_review_only_record_failure_does_not_claim_main_generation(review_usage, monkeypatch):
    rig = review_usage
    async def prepare(turn, role):
        await rig.consume()
        return SimpleNamespace(character_id=role, memory_operation_receipt=None)
    rig.service.prepare_turn = prepare
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', AsyncMock(return_value=(
        "The currently stored synthetic facts were read.", False,
        {"modelInvoked": False, "answerMode": "memory_lookup"},
    )))
    rig.writer.side_effect = OSError("synthetic-review-only-storage-error")
    with pytest.raises(HTTPException) as caught:
        await rig.run()
    assert caught.value.detail["stage"] == "model_invocation_record"
    assert caught.value.detail["modelInvoked"] is False and caught.value.detail["messageSaved"] is False
    assert rig.writer.call_count == len(rig.installed.calls) == 1 and rig.rows() == []



@pytest.mark.asyncio
@pytest.mark.parametrize("record", [True, False])
async def test_unknown_result_after_review_keeps_only_actual_usage_and_does_not_archive(review_usage, monkeypatch, record):
    rig = review_usage

    async def prepare(turn, role):
        await rig.consume()
        return SimpleNamespace(character_id=role, memory_operation_receipt=None)

    rig.service.prepare_turn = prepare
    monkeypatch.setattr(chat_answer, "generate_chat_answer", AsyncMock(return_value=(
        "Synthetic undeclared result.", False,
        {"modelInvoked": False, "answerMode": "synthetic-unknown-mode"},
    )))
    with pytest.raises(HTTPException) as caught:
        await rig.run(record)
    assert caught.value.status_code == 503 and caught.value.detail["stage"] == "generation_result"
    assert caught.value.detail["modelInvoked"] is False and caught.value.detail["messageSaved"] is False
    assert isinstance(caught.value.__cause__, KeyError)
    assert len(rig.installed.calls) == 1 and rig.writer.call_count == int(record)
    rows = rig.rows()
    assert len(rows) == int(record)
    if record:
        row, = rows
        assert (row["promptTokens"], row["completionTokens"], row["totalTokens"]) == (100, 20, 120)
        assert row["modelName"] == "fixture-model" and row["usageSource"] == "provider"
        assert row["errorType"] == "KeyError"
    assert rig.database.execute_sql("SELECT id FROM messages") == []
    assert get_request_context() is None and not has_pending_usage()
