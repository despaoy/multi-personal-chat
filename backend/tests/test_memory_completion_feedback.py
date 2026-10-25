"""Only completion feedback and its two generation consumers; no full suite."""

from services import chat_answer

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection

from db.schemas import MessageRequest
from services.character_context import _TurnOutcome


def request():
    return MessageRequest(
        message="朋友许澄的完整合成课程记录，尚未参加、尚未出发。",
        characterId="tsukiyashiro_kisaki",
        sourceMessageId="new-source",
        sessionId="synthetic",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capture,expected",
    [
        ("failed", "保存失败"),
        ("revoked", "已删除"),
        ("stale", "已删除"),
        ("conflict", "其他内容"),
        ("recorded", None),
        ("", None),
    ],
)
async def test_capture_outcome_is_not_discarded(capture, expected):
    from api import generate as gen

    service = SimpleNamespace(
        complete_turn=AsyncMock(
            return_value=_TurnOutcome(source_capture=capture, memory_enrichment_status="skipped")
        )
    )
    warning = await gen._complete_character_turn(
        SimpleNamespace(character_id="tsukiyashiro_kisaki"), request(), character_service=service
    )
    assert warning is None if expected is None else expected in warning
    assert service.complete_turn.await_args.kwargs["source_message_id"] == "new-source"


@pytest.mark.asyncio
async def test_real_coroutine_timeout_is_unknown_not_failed(monkeypatch):
    from api import generate as gen
    from services import turn_completion

    runtime = turn_completion.TurnCompletionRuntime()
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", lambda: runtime)
    monkeypatch.setattr(gen, "_DB_WRITE_TIMEOUT", 0.01)
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow(*args, **kwargs):
        started.set()
        await release.wait()
        finished.set()

    warning = await gen._complete_character_turn(
        SimpleNamespace(character_id="tsukiyashiro_kisaki"), request(),
        character_service=SimpleNamespace(complete_turn=slow),
    )
    assert started.is_set() and not finished.is_set() and runtime.active == 1
    assert "尚未确认" in warning and "保存失败" not in warning and "稍后检查" in warning
    release.set()
    await runtime.shutdown(timeout=1)
    assert finished.is_set() and runtime.cancelled == runtime.active == 0


@pytest.mark.asyncio
async def test_unexpected_completion_failure_does_not_leak_exception():
    from api import generate as gen

    service = SimpleNamespace(complete_turn=AsyncMock(side_effect=RuntimeError("private-internal-detail")))
    with pytest.raises(RuntimeError, match="private-internal-detail"):
        await gen._complete_character_turn(
            SimpleNamespace(character_id="tsukiyashiro_kisaki"), request(), character_service=service
        )


@pytest.mark.asyncio
async def test_external_cancel_is_not_converted_to_success_feedback():
    from api import generate as gen

    service = SimpleNamespace(complete_turn=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await gen._complete_character_turn(
            SimpleNamespace(character_id="tsukiyashiro_kisaki"), request(), character_service=service
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai_compat"])
@pytest.mark.parametrize("mode", ["failed", "recorded", "internal", "archive_failure", "completion_failure"])
async def test_generation_preserves_answer_and_existing_warnings(monkeypatch, provider, mode):
    from api import generate as gen
    from inference import model_manager as mm

    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value=provider),
        set_lora_adapter=lambda _: None,
        get_status=lambda: dict(currentProvider=provider, providers={provider: dict(modelName="unit-model")}),
    )
    connection = cloud_connection("unit-model")
    manager.get_current_provider = lambda: connection
    monkeypatch.setattr(mm, "get_model_manager", lambda: manager)
    monkeypatch.setattr(gen, "response_cache", None)
    monkeypatch.setattr(gen, "circuit_breaker_registry", None)
    monkeypatch.setattr(gen, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(gen, "get_vllm_client", AsyncMock(return_value=SimpleNamespace(generate=AsyncMock()) if provider == 'vllm' else None))
    meta = dict(warnings=["原有检索提示"], answerMode="unit", confidence=0.3, citations=[])
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', AsyncMock(return_value=("模型原始内容", False, meta)))
    failure = OSError("fiction-archive-write-error")
    if mode == "archive_failure":
        def fail(record):
            raise failure
        gen.db.add_message = fail
    else:
        monkeypatch.setattr(gen, "_save_message", AsyncMock(return_value=True))
    prepared = SimpleNamespace(
        memory_operation_receipt=None, character_id="tsukiyashiro_kisaki",
        compiled=SimpleNamespace(profile_context="", dynamic_context="", reference_context=""),
        history=(),
    )
    monkeypatch.setattr(gen, "_prepare_character_turn", AsyncMock(return_value=prepared))
    service = SimpleNamespace(
        complete_turn=AsyncMock(return_value=_TurnOutcome(source_capture=mode, memory_enrichment_status="skipped"))
    )
    if mode == "completion_failure":
        failure = TimeoutError("synthetic-private-completion-error")
        service.complete_turn.side_effect = failure
        with pytest.raises(HTTPException) as error:
            await gen._generate_reply_impl(request(), enable_rag=False, record_invocation=False,
                character_service=service, message_db=gen.db)
        assert error.value.status_code == 503 and error.value.detail["stage"] == "turn_completion"
        assert error.value.detail["modelInvoked"] is True and error.value.detail["messageSaved"] is True
        assert error.value.__cause__ is failure and "synthetic-private" not in str(error.value.detail)
        gen._save_message.assert_awaited_once()
        service.complete_turn.assert_awaited_once()
        return
    if mode == "archive_failure":
        with pytest.raises(HTTPException) as error:
            await gen._generate_reply_impl(request(), enable_rag=False, record_invocation=False, character_service=service, message_db=gen.db)
        assert error.value.status_code == 503 and error.value.detail["stage"] == "message_archive"
        assert error.value.detail["modelInvoked"] is True and error.value.detail["messageSaved"] is None
        assert error.value.__cause__ is failure
        service.complete_turn.assert_not_awaited()
        return
    result = await gen._generate_reply_impl(
        request(),
        persist_message=mode != "internal",
        enable_rag=False,
        record_invocation=False,
        character_service=service,
        message_db=gen.db,
    )
    assert result.warnings[0] == "原有检索提示" and result.confidence == 0.3 and result.citations == []
    if mode == "failed":
        assert result.reply.startswith("模型原始内容\n\n保存提示：") and "保存失败" in result.reply
        assert len(result.warnings) == 2 and result.warnings[1] in result.reply
    else:
        assert result.reply == "模型原始内容" and result.warnings == ["原有检索提示"]
    assert meta["warnings"] == ["原有检索提示"]
    if mode == "internal":
        service.complete_turn.assert_not_awaited()
    else:
        service.complete_turn.assert_awaited_once()


async def test_unknown_capture_is_not_converted_to_success_feedback():
    from api import generate as gen

    service = SimpleNamespace(complete_turn=AsyncMock(return_value=_TurnOutcome(
        source_capture="unsupported_scope", memory_enrichment_status="skipped")))
    with pytest.raises(ValueError, match="Unrecognized memory completion result"):
        await gen._complete_character_turn(SimpleNamespace(character_id="role"), request(), character_service=service)
