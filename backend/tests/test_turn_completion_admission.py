"""Only generation admission, slot handoff, and failed-request cleanup."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from services.turn_completion import CompletionUnavailable, TurnCompletionRuntime


@pytest.mark.asyncio
async def test_reservations_compete_with_active_tasks_before_generation():
    runtime = TurnCompletionRuntime(capacity=2)
    first, second = runtime.reserve(), runtime.reserve()
    assert runtime.active == 0 and runtime.reserved == 2
    with pytest.raises(CompletionUnavailable):
        runtime.reserve()
    first.release()
    first.release()
    release = asyncio.Event()
    with pytest.raises(TimeoutError):
        await runtime.run(release.wait, timeout=0.005, reservation=second)
    second.release()
    assert runtime.active == 1 and runtime.reserved == 0
    third = runtime.reserve()
    with pytest.raises(CompletionUnavailable):
        runtime.reserve()
    third.release()
    release.set()
    await runtime.shutdown(timeout=1)
    assert runtime.active == runtime.reserved == runtime.cancelled == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["foreign", "released", "consumed"])
async def test_slot_authority_rejects_without_second_work(invalid):
    runtime = TurnCompletionRuntime()
    other = TurnCompletionRuntime()
    slot = (other if invalid == "foreign" else runtime).reserve()
    if invalid == "released":
        slot.release()
    if invalid == "consumed":
        await runtime.run(_empty, timeout=1, reservation=slot)
    invoked = []

    def forbidden():
        invoked.append(True)
        return _empty()

    with pytest.raises(RuntimeError):
        await runtime.run(forbidden, timeout=1, reservation=slot)
    assert not invoked
    slot.release()
    await runtime.shutdown(timeout=1)
    await other.shutdown(timeout=1)


async def _empty():
    return None


@pytest.mark.asyncio
async def test_factory_failure_releases_its_pre_generation_reservation():
    runtime = TurnCompletionRuntime(capacity=1)
    slot = runtime.reserve()

    def broken():
        raise ValueError("factory-failed")

    with pytest.raises(ValueError):
        await runtime.run(broken, timeout=1, reservation=slot)
    assert runtime.active == runtime.reserved == 0
    replacement = runtime.reserve()
    replacement.release()
    await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_shutdown_invalidates_unused_slots_and_prevents_reopen():
    runtime = TurnCompletionRuntime()
    slot = runtime.reserve()
    await runtime.shutdown(timeout=1)
    assert slot.state == "released" and runtime.reserved == 0
    with pytest.raises(CompletionUnavailable):
        await runtime.run(_empty, timeout=1, reservation=slot)


def generation_environment(monkeypatch, provider):
    from api import generate as gen
    from db.schemas import MessageRequest
    from inference import model_manager as mm
    from services import turn_completion

    runtime = TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(turn_completion, "get_turn_completion_runtime", lambda: runtime)
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value=provider),
        set_lora_adapter=lambda _: None,
        get_status=lambda: dict(currentProvider=provider, providers={provider: dict(modelName="unit-model")}),
    )
    monkeypatch.setattr(mm, "get_model_manager", lambda: manager)
    monkeypatch.setattr(gen, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(gen, "response_cache", None)
    monkeypatch.setattr(gen, "circuit_breaker_registry", None)
    monkeypatch.setattr(gen, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(gen, "_ensure_vllm", AsyncMock(return_value=provider == "vllm"))
    monkeypatch.setattr(gen, "_vllm_client", object())
    prepared = SimpleNamespace(
        character_id="tsukiyashiro_kisaki",
        compiled=SimpleNamespace(profile_context="", dynamic_context="", reference_context=""),
        history=(),
    )
    preparation = AsyncMock(return_value=prepared)
    storage = AsyncMock(return_value=True)
    generation = AsyncMock(
        return_value=(
            "真实单元输出",
            False,
            dict(warnings=["检索提示"], answerMode="unit", confidence=0.3, citations=[]),
        )
    )
    monkeypatch.setattr(gen, "_prepare_character_turn", preparation)
    monkeypatch.setattr(gen, "_generate_with_vllm", generation)
    monkeypatch.setattr(gen, "_generate_with_retrieval", generation)
    monkeypatch.setattr(gen, "_save_message", storage)
    monkeypatch.setattr(gen, "_persist_completion_feedback", AsyncMock(return_value=True))
    request = MessageRequest(
        message="朋友陶弈本人收到完整书面预约，回执EN-935-V，有效、未参加、未出发。",
        characterId="tsukiyashiro_kisaki",
        sourceMessageId="target",
        sessionId="scope-owned",
    )
    request._source_received_at = datetime.now(timezone.utc)
    return gen, runtime, request, preparation, generation, storage


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai_compat"])
async def test_busy_route_rejects_before_prepare_model_and_archive(monkeypatch, provider):
    gen, runtime, request, prepare, generation, storage = generation_environment(monkeypatch, provider)
    occupied = runtime.reserve()
    with pytest.raises(HTTPException) as error:
        await gen._generate_reply_impl(request, enable_rag=False, record_invocation=False)
    assert error.value.status_code == 503 and error.value.detail["code"] == "turn_completion_busy"
    prepare.assert_not_awaited()
    generation.assert_not_awaited()
    storage.assert_not_awaited()
    assert runtime.reserved == 1 and runtime.active == 0
    occupied.release()
    await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai_compat"])
async def test_same_pre_generation_slot_survives_response_deadline(monkeypatch, provider):
    gen, runtime, request, prepare, generation, storage = generation_environment(monkeypatch, provider)
    monkeypatch.setattr(gen, "_DB_WRITE_TIMEOUT", 0.005)
    release = asyncio.Event()
    observed = []

    async def complete(prepared, turn, reply, *, source_message_id):
        observed.append((runtime.reserved, runtime.active, turn.message, turn.received_at, source_message_id, reply))
        await release.wait()
        return SimpleNamespace(source_capture="recorded")

    result = await gen._generate_reply_impl(
        request, enable_rag=False, record_invocation=False, character_service=SimpleNamespace(complete_turn=complete)
    )
    assert result.reply.startswith("真实单元输出") and "尚未确认" in result.reply and result.warnings[0] == "检索提示"
    assert observed == [(0, 1, request.message, request._source_received_at, "target", "真实单元输出")]
    assert runtime.active == 1 and runtime.reserved == 0
    generation.assert_awaited_once()
    prepare.assert_awaited_once()
    storage.assert_awaited_once()
    release.set()
    await runtime.shutdown(timeout=1)
    assert runtime.completed == 1 and runtime.active == runtime.reserved == runtime.cancelled == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [HTTPException(503, "context unavailable"), asyncio.CancelledError()])
async def test_preparation_error_or_cancel_releases_capacity(monkeypatch, failure):
    gen, runtime, request, prepare, generation, storage = generation_environment(monkeypatch, "openai_compat")
    prepare.side_effect = failure
    with pytest.raises(type(failure)):
        await gen._generate_reply_impl(request, enable_rag=False, record_invocation=False)
    assert runtime.active == runtime.reserved == 0
    generation.assert_not_awaited()
    storage.assert_not_awaited()
    await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_archive_failure_releases_unused_slot_without_complete(monkeypatch):
    gen, runtime, request, prepare, generation, storage = generation_environment(monkeypatch, "openai_compat")
    storage.return_value = False
    complete = AsyncMock()
    await gen._generate_reply_impl(
        request, enable_rag=False, record_invocation=False, character_service=SimpleNamespace(complete_turn=complete)
    )
    complete.assert_not_awaited()
    assert runtime.active == runtime.reserved == 0
    await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_internal_no_archive_does_not_reserve_completion(monkeypatch):
    gen, runtime, request, prepare, generation, storage = generation_environment(monkeypatch, "openai_compat")
    occupied = runtime.reserve()
    complete = AsyncMock()
    await gen._generate_reply_impl(
        request,
        persist_message=False,
        enable_rag=False,
        record_invocation=False,
        character_service=SimpleNamespace(complete_turn=complete),
    )
    complete.assert_not_awaited()
    storage.assert_not_awaited()
    assert runtime.reserved == 1 and runtime.active == 0
    occupied.release()
    await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_relation_failure_keeps_formatted_root_diagnostic(monkeypatch, caplog):
    from character import memory_llm
    from character.models import RelationshipState, UserScope
    from services.character_context import CharacterContextService, TurnInput

    service = object.__new__(CharacterContextService)
    service._memory_repo = SimpleNamespace(
        increment_interaction=AsyncMock(return_value=1),
        get_relationship_record=AsyncMock(return_value=None),
        upsert_relationship=AsyncMock(side_effect=RuntimeError("synthetic-relation-failure")),
    )
    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: SimpleNamespace(enabled=False))
    scope = UserScope("web", "unit", "owner", "owner", "private")
    prepared = SimpleNamespace(
        character_id="role",
        user_scope=scope,
        received_at=datetime.now(timezone.utc),
        memory_operation_receipt=None,
        relationship=RelationshipState(),
        compiled=SimpleNamespace(used_memory_ids=()),
    )
    turn = TurnInput("请叫我杉岚。", "web", "unit", "owner", "owner", "private")
    result = await service.complete_turn(prepared, turn, "ack", source_message_id="source")
    records = [r for r in caplog.records if r.name == "services.character_context" and "角色关系更新失败" in str(r.msg)]
    assert result.interaction_count == 1 and len(records) == 1
    assert records[0].getMessage() == "角色关系更新失败 character=role"
    assert isinstance(records[0].exc_info[1], RuntimeError)
