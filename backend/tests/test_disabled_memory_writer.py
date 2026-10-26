"""Disabled automatic memory is visible and never becomes rule writes.

Two real SQLite chains exercise complete_turn through the Web completion/archive
consumer and through the durable delivery worker. Context preparation and reply
generation use explicitly synthetic fixtures; every actual model boundary and
the automatic rule writer are hard-blocked. No paid model calls are used.

"""

from services import chat_answer
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from provider_fixtures import cloud_connection
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services import delivery_memory as dm
from services import turn_completion
from services.character_context import CharacterContextService, PreparedCharacterTurn
from test_delivery_memory import deliver, generated

from api import generate
from character.models import build_user_scope
from character.memory_extractor import extract_memories
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import CompiledCharacterContext, RelationshipState
from db.database import SQLiteDB
from db.schemas import MessageRequest
from services.chat_answer import ChatAnswer

BODY = "我喜欢咖啡。"  # A real rule-extractor match, so the former fallback would try to write.
REPLY = "这是一条预先提供的合成回复。"
ROLE = "tsukiyashiro_kisaki"


@pytest.fixture
async def disabled_writer(tmp_path, monkeypatch, no_live_context_review):
    """Use a real disabled scheduler, never a fake disabled outcome."""
    from inference import model_manager
    from knowledge.retrieval_core import embedding
    for key in ('CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'CONTEXTUAL_DECISION_POLICY_ENABLED', 'DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED', 'MEMORY_SOURCE_RECALL_ENABLED'):
        monkeypatch.setenv(key, 'false')
    database = SQLiteDB(tmp_path / 'disabled-writer.sqlite')
    connections = {}
    actual_connection = database._get_connection

    def connection():
        result = actual_connection()
        connections[id(result)] = result
        return result
    monkeypatch.setattr(database, '_get_connection', connection)
    config = MemoryLlmConfig.from_env({'MEMORY_LLM_ENABLED': 'false'})
    assert config.enabled is False
    memory_model = AsyncMock(side_effect=AssertionError('disabled memory must not invoke a model'))
    completion = SimpleNamespace(complete=memory_model, close=AsyncMock())
    scheduler = MemoryEnrichmentScheduler(config=config, completion=completion)
    monkeypatch.setattr('character.memory_llm.get_memory_enrichment_scheduler', lambda _owner_database=None: scheduler)
    rule_writer = AsyncMock(side_effect=AssertionError('disabled memory must not invoke the automatic rule writer'))
    monkeypatch.setattr('character.rule_memory_writer.write_rule_memory', rule_writer)
    embedding_factory = Mock(side_effect=AssertionError('this test must not construct an embedding model'))
    monkeypatch.setattr(embedding, 'get_default_embedding_provider', embedding_factory)
    connection_fixture = cloud_connection('fiction-reply-model')
    local_model = AsyncMock(side_effect=AssertionError('this test must not invoke a local model'))
    manager = SimpleNamespace(_current_provider=SimpleNamespace(value='openai_compat'), get_current_provider=lambda: connection_fixture, async_generate=local_model, set_lora_adapter=lambda _: None, get_status=lambda: dict(currentProvider='openai_compat', providers={'openai_compat': dict(modelName='fiction-reply-model')}))
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: manager)
    runtime = turn_completion.TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(turn_completion, 'get_turn_completion_runtime', lambda _owner_database=None: runtime)
    monkeypatch.setattr(dm, 'get_turn_completion_runtime', lambda _owner_database=None: runtime)
    repo = DatabaseCharacterMemoryRepository(database)
    service = CharacterContextService(object(), repo, object())
    actual_complete = service.complete_turn
    outcomes = []

    async def complete(*args, **kwargs):
        outcome = await actual_complete(*args, **kwargs)
        outcomes.append(outcome)
        return outcome
    monkeypatch.setattr(service, 'complete_turn', complete)
    rig = SimpleNamespace(database=database, repo=repo, service=service, scheduler=scheduler, runtime=runtime, outcomes=outcomes, memory_model=memory_model, rule_writer=rule_writer, provider=connection_fixture, local_model=local_model, embedding_factory=embedding_factory)
    try:
        assert extract_memories(BODY)
        yield rig
    finally:
        await runtime.shutdown(timeout=2)
        await scheduler.shutdown(timeout=2)
        main = database._get_connection()
        for current in tuple(connections.values()):
            if current is not main:
                current.close()
        database.close_connection()


def assert_no_models_or_rule_fallback(rig):
    rig.memory_model.assert_not_awaited()
    rig.provider.async_complete.assert_not_awaited()
    rig.local_model.assert_not_awaited()
    rig.embedding_factory.assert_not_called()
    rig.rule_writer.assert_not_awaited()
    assert rig.scheduler.status.saved == rig.scheduler.status.failed == 0
    assert rig.scheduler.status.queued == rig.scheduler.status.processing == rig.scheduler.status.buffered == 0


@pytest.mark.asyncio
async def test_real_disabled_completion_is_visible_in_web_reply_and_exact_archive(disabled_writer, monkeypatch):
    rig = disabled_writer
    request = MessageRequest(message=BODY, characterId=ROLE, loraName="default",
        platform="web", adapter="web-character", senderId="fiction-owner", userId="fiction-owner",
        sessionId="fiction-owner", conversationId="fiction-owner", conversationType="private",
        sourceMessageId="fiction-disabled-web", traceId="fiction-disabled-web-trace")
    request._source_received_at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    scope = build_user_scope(request.platform, request.adapter, request.senderId,
                             request.conversationId, request.conversationType)
    prepared = PreparedCharacterTurn(character_id=ROLE, user_scope=scope,
        compiled=CompiledCharacterContext("", "", "", ()), history=(),
        relationship=RelationshipState(), memory_candidates=0, interaction_count=0,
        reply_guard=None, received_at=request._source_received_at)
    monkeypatch.setattr(generate, "_prepare_character_turn", AsyncMock(return_value=prepared))
    # This is the existing generation boundary fixture, not a real provider
    # response or a model-quality test. Completion, archive and feedback stay real.
    synthetic_reply = AsyncMock(return_value=ChatAnswer(reply=REPLY, used_rag=False, model_invoked=False, metadata=dict(answerMode='current_constraint', warnings=['合成生成夹具提示'])))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', synthetic_reply)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    result = await generate._generate_reply_impl(request, persist_message=True,
        enable_rag=False, record_invocation=False, character_service=rig.service, message_db=rig.database)

    synthetic_reply.assert_awaited_once()
    assert len(rig.outcomes) == 1
    outcome = rig.outcomes[0]
    assert outcome.memory_enrichment_status == outcome.memory_enrichment_mode == "disabled"
    assert not outcome.memory_enrichment_scheduled
    assert outcome.source_capture == "" and outcome.failed_writes == () and outcome.interaction_count == 1
    assert result.modelInvoked is False and result.warnings[0] == "合成生成夹具提示"
    warning = result.warnings[-1]
    assert "未启用" in warning and "不代表已保存" in warning
    assert "保存失败" not in warning and "重试" not in warning
    assert result.reply == f"{REPLY}\n\n保存提示：{warning}"

    rows = [row for row in rig.database.get_messages(limit=10) if row["traceId"] == request.traceId]
    assert len(rows) == 1 and rows[0]["message"] == BODY and rows[0]["reply"] == result.reply
    history = rig.database.list_conversation_history(request.platform, request.adapter,
        request.senderId, request.conversationType, request.conversationId, character_id=ROLE)
    assert history[-1] == dict(role="assistant", content=result.reply)
    assert await rig.repo.list_memory_records(ROLE, scope, include_inactive=True) == []
    assert await rig.repo.list_sources(ROLE, scope) == []
    relationship = await rig.repo.get_relationship_record(ROLE, scope)
    assert relationship["interaction_count"] == 1
    assert rig.runtime.active == rig.runtime.reserved == 0
    assert_no_models_or_rule_fallback(rig)


@pytest.mark.asyncio
async def test_real_disabled_delivery_is_terminal_with_reason_and_never_replayed(disabled_writer, monkeypatch):
    rig = disabled_writer
    key, owner, stored = generated(rig.database)
    context = stored["context"]
    context["request"] = MessageRequest.model_validate(dict(context["request"], message=BODY)).model_dump()
    snapshot, _ = dm.thaw_completion(context)
    original_reply = dict(stored["reply"])
    deliver(rig.database, key, owner, stored)
    monkeypatch.setattr(dm, "POLL_SECONDS", 3600)
    pauses = asyncio.Queue()
    actual_pause = dm.DeliveryMemoryWorker._pause

    async def pause(worker):
        pauses.put_nowait(None)
        await actual_pause(worker)

    monkeypatch.setattr(dm.DeliveryMemoryWorker, "_pause", pause)
    parked = rig.runtime.reserve()
    worker = dm.DeliveryMemoryWorker(rig.database, rig.service)
    try:
        # The actual owner loop sees the occupied runtime and enters its pause;
        # drive this one real CAS job explicitly, without racing the loop.
        await asyncio.wait_for(pauses.get(), timeout=1)
        parked.release()
        slot = rig.runtime.reserve()
        try:
            await worker._process(rig.database.integration_receipt("get", key=key), rig.runtime, slot)
        finally:
            slot.release()
        row = rig.database.integration_receipt("get", key=key)
        response = json.loads(row["response"])
        marker = response["context"]["memory_completion"]
        assert row["status"] == "delivered"
        assert response["memory_completion_state"] == marker["state"] == "blocked"
        assert marker["reason"] == "memory_disabled" and marker["attempts"] == 1
        assert marker["semantic_receipt"]["status"] == "disabled"
        outcome = marker["completion_outcome"]
        assert outcome["memory_enrichment_status"] == outcome["memory_enrichment_mode"] == "disabled"
        assert outcome["memory_enrichment_scheduled"] is False
        assert outcome["failed_writes"] == [] and outcome["interaction_count"] == 1
        assert marker["turn_finished"] is True and response["reply"] == original_reply
        assert len(rig.outcomes) == 1
        assert await rig.repo.list_memory_records(snapshot.character_id, snapshot.user_scope,
                                                  include_inactive=True) == []
        assert await rig.repo.list_sources(snapshot.character_id, snapshot.user_scope) == []
        assert (await rig.repo.get_relationship_record(snapshot.character_id,
            snapshot.user_scope))["interaction_count"] == 1
        assert not any(item["receipt_key"] == key for item in rig.database.integration_receipt(
            "memory_pending", now=float("inf"), limit=100))

        # Wake the actual loop for another complete DB polling round. A disabled
        # terminal receipt must not replay the turn or increment interaction.
        worker.wakeup.set()
        await asyncio.wait_for(pauses.get(), timeout=1)
        assert rig.database.integration_receipt("get", key=key) == row
        assert len(rig.outcomes) == 1
        assert (await rig.repo.get_relationship_record(snapshot.character_id,
            snapshot.user_scope))["interaction_count"] == 1
        assert_no_models_or_rule_fallback(rig)
    finally:
        parked.release()
        await worker.shutdown(timeout=2)
