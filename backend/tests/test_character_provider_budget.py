"""Character services use the provider resolved for their actual turn.

SQLite, profile loading, default/service factories, recall, selection parsing,
context compilation, queue-entry orchestration and completion writes are real.
Only model output and queue timing are controlled. No live model, external
HTTP, real credentials or paid inference are used.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from html import unescape
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request
from provider_fixtures import cloud_connection
from services import character_context
from services.character_context import TurnInput

from api import generate
from app.runtime import RuntimeContainer
from character import profile_registry
from character.models import MemoryItem, UserScope
from db import adapter
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference import model_manager
from inference.provider_context import get_provider_context_budget
from inference.request_context import get_request_context, resolve_request_context, use_request_context
from inference.review_client import OpenAICompatibleReviewClient
from infra.circuit_breaker import CircuitBreakerRegistry

ROLE = "fiction-provider-budget"
OWNER = "fiction-budget-owner"
BODY = "虚构排练记录：林澈说本周四14:00至15:00排练，尚未报名；仅收到书面确认且当天不下雨才参加，预约不豁免雨天限制。"
QUERY = "核对我保存的虚构排练记录的时间、否定和例外；保留其引述属性，不视为我的个人经历。"


async def arrange_provider_budget(tmp_path, monkeypatch):
    """Return a real default-factory rig; do not assume any cache repair."""
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "fixture.json").write_text(json.dumps({
        "character_id": ROLE, "display_name": "虚构预算核对角色",
        "identity": "保留引述主体、时间、否定与例外的虚构资料核对角色",
        "values": ["完整记录优先，不补写经历"],
    }, ensure_ascii=False), encoding="utf-8")
    registry = profile_registry.CharacterProfileRegistry(profiles)
    registry.load_profiles()
    monkeypatch.setattr(profile_registry, "_default_registry", registry)
    monkeypatch.setattr(character_context, "_default_service", None)
    for name, value in {
        "OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "65536", "VLLM_MAX_MODEL_LEN": "8192",
        "CONTEXTUAL_MEMORY_SELECTION_ENABLED": "true", "CAHM_SEMANTIC_MEMORY_ENABLED": "false",
        "CONTEXTUAL_DECISION_POLICY_ENABLED": "false", "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED": "false",
        "MEMORY_SOURCE_RECALL_ENABLED": "false", "CHARACTER_INDEPENDENT_TASKS_ENABLED": "false",
    }.items():
        monkeypatch.setenv(name, value)
    default_db = SQLiteDB(tmp_path / "default-budget.sqlite")
    custom_db = SQLiteDB(tmp_path / "custom-budget.sqlite")
    circuit_registry = CircuitBreakerRegistry()
    scope = UserScope("web", "web-character", OWNER, OWNER, "private")
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    for database in (default_db, custom_db):
        database.update_config(dict(useKnowledgeBase=False, temperature=.2, maxTokens=256, topP=1.0))
        repository = DatabaseCharacterMemoryRepository(database)
        assert await repository.capture_source(
            ROLE, scope, source_message_id="fiction-rehearsal-source", body=BODY, observed_at=stamp,
        ) == "recorded"
        await repository.append_claim(
            ROLE, scope, MemoryItem("", "user_fact", "虚构排练引文：" + BODY, .9),
            memory_key="fiction_rehearsal_quote", confidence=.99, evidence=(BODY,),
            source_message_id="fiction-rehearsal-source", observed_at=stamp.isoformat(),
            metadata={"content_semantics": "quoted_source", "speaker_role": "user",
                      "described_subject": "not_resolved"},
        )
    rig = SimpleNamespace(
        database=default_db, custom_db=custom_db, scope=scope,
        reviews=[], answers=[], review_hold=None, registry=circuit_registry,
    )

    async def completion(messages, **params):
        try:
            payload = json.loads(messages[-1]["content"])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "required_ids" in payload:
            assert payload["query"] == QUERY
            assert len(payload["candidates"]) == len(payload["required_ids"]) == 1
            assert payload["candidates"][0]["evidence"] == [BODY]
            assert params["temperature"] == 0.0 and params["max_tokens"] == 2048
            kind = get_request_context().provider_name
            rig.reviews.append({"provider": kind, "payload": payload})
            if rig.review_hold is not None and rig.review_hold[0] == kind:
                _kind, entered, release = rig.review_hold
                entered.set()
                await release.wait()
            return json.dumps({"decisions": [
                {"id": identity, "label": "use"} for identity in payload["required_ids"]
            ]}), 0.0
        wire = unescape("\n".join(message["content"] for message in messages))
        assert QUERY in wire and BODY in wire
        rig.answers.append({"wire": wire, "params": params, "context": get_request_context()})
        return BODY, 0.0

    connection = cloud_connection("fiction-budget-model", completion=completion)
    manager = SimpleNamespace(
        _current_provider=model_manager.ModelProvider.VLLM,
        get_current_provider=lambda: connection,
    )
    rig.manager = manager
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    # Every selected review transport has a complete, explicit unit client.
    monkeypatch.setattr("inference.review_client.get_context_review_client",
                        AsyncMock(return_value=OpenAICompatibleReviewClient(connection)))
    monkeypatch.setattr(adapter, "db", default_db)
    monkeypatch.setattr(generate, "db", default_db)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", circuit_registry)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: asyncio.Semaphore(4))
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "只核对所给完整虚构记录，保留引述属性。")
    # Explicitly disable the independent post-reply semantic writer; its
    # interpreter is outside this budget-factory test. Actual counter writes run.
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", lambda: SimpleNamespace(enabled=False))
    return rig


@pytest.fixture
async def budget_rig(tmp_path, monkeypatch):
    rig = await arrange_provider_budget(tmp_path, monkeypatch)
    try:
        yield rig
    finally:
        await rig.registry.clear()
        rig.database.close_connection()
        rig.custom_db.close_connection()


async def select_context(rig, provider):
    rig.manager._current_provider = model_manager.ModelProvider(provider)
    return await resolve_request_context(rig.manager)


def turn_input():
    return TurnInput(QUERY, "web", "web-character", OWNER, "fiction-budget-session", "private")


def reference_packet(prepared):
    rows = [json.loads(line[2:]) for line in prepared.compiled.reference_context.splitlines() if line.startswith("- {")]
    assert len(rows) == 1 and BODY in rows[0]["evidence"]
    return rows[0]


async def observe_adjacent_provider_budgets(rig):
    """Preflight helper: return actual observations without asserting a repair."""
    observations = []
    for provider in ("vllm", "openai_compat"):
        context = await select_context(rig, provider)
        with use_request_context(context):
            current = get_provider_context_budget()
            service = character_context.get_default_character_context_service()
            prepared = await service.prepare_turn(turn_input(), ROLE)
            observations.append(dict(
                provider=provider, current_window=current.window_tokens,
                current_history_chars=current.history_max_chars, service_history_chars=service._history_max_chars,
                current_reference_cap=current.reference_max_chars or 6000,
                actual_reference_cap=prepared.memory_budget["budget_chars"],
                source_observation=reference_packet(prepared)["source_observation"],
            ))
    return observations


@pytest.mark.asyncio
async def test_default_factory_reuses_same_budget_and_refreshes_adjacent_provider_turn(budget_rig):
    rig = budget_rig
    local = await select_context(rig, "vllm")
    with use_request_context(local):
        first = character_context.get_default_character_context_service()
        assert character_context.get_default_character_context_service() is first
        prepared_local = await first.prepare_turn(turn_input(), ROLE)
        assert prepared_local.memory_budget["budget_chars"] == 6000
        assert BODY in prepared_local.compiled.reference_context
        original_history_cap = first._history_max_chars
    cloud = await select_context(rig, "openai_compat")
    with use_request_context(cloud):
        second = character_context.get_default_character_context_service()
        assert character_context.get_default_character_context_service() is second
        prepared_cloud = await second.prepare_turn(turn_input(), ROLE)
        assert prepared_cloud.memory_budget["budget_chars"] == get_provider_context_budget().reference_max_chars == 32768
        assert second._history_max_chars == get_provider_context_budget().history_max_chars == 65536
        assert reference_packet(prepared_cloud)["source_observation"] is True
        assert reference_packet(prepared_cloud)["subject_scope"] == "not_resolved"
    assert second is not first
    assert first._history_max_chars == original_history_cap == 16000
    assert prepared_local.memory_budget["budget_chars"] == 6000
    assert not rig.answers and [row["provider"] for row in rig.reviews] == ["vllm", "openai_compat"]


@pytest.mark.asyncio
async def test_invalid_current_budget_errors_instead_of_reusing_old_service(budget_rig, monkeypatch):
    rig = budget_rig
    cloud = await select_context(rig, "openai_compat")
    with use_request_context(cloud):
        previous = character_context.get_default_character_context_service()
        monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "4096")
        with pytest.raises(ValueError, match="Cloud working context window"):
            character_context.get_default_character_context_service()
        assert previous._history_max_chars == 65536
        monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "65536")
        assert character_context.get_default_character_context_service() is previous
    assert not rig.reviews and not rig.answers


@pytest.mark.asyncio
async def test_captured_local_preparation_is_not_mutated_by_overlapping_cloud_factory(budget_rig):
    rig = budget_rig
    local = await select_context(rig, "vllm")
    entered, release = asyncio.Event(), asyncio.Event()
    rig.review_hold = ("vllm", entered, release)

    async def prepare_local():
        with use_request_context(local):
            service = character_context.get_default_character_context_service()
            prepared = await service.prepare_turn(turn_input(), ROLE)
            assert get_request_context() is local
            return service, prepared

    pending = asyncio.create_task(prepare_local())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        cloud = await select_context(rig, "openai_compat")
        with use_request_context(cloud):
            cloud_service = character_context.get_default_character_context_service()
            prepared_cloud = await cloud_service.prepare_turn(turn_input(), ROLE)
            assert get_request_context() is cloud
        release.set()
        local_service, prepared_local = await asyncio.wait_for(pending, 1)
        assert prepared_local.memory_budget["budget_chars"] == 6000
        assert prepared_cloud.memory_budget["budget_chars"] == 32768
        assert local_service._history_max_chars == 16000 and cloud_service._history_max_chars == 65536
        assert BODY in prepared_local.compiled.reference_context and BODY in prepared_cloud.compiled.reference_context
        assert reference_packet(prepared_cloud)["source_observation"] is True
        assert get_request_context() is None
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("database_kind", ["default", "custom"])
async def test_queued_http_turn_uses_execution_provider_and_one_owning_service(budget_rig, monkeypatch, database_kind):
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    from app.main import create_app

    rig = budget_rig
    database = rig.database if database_kind == "default" else rig.custom_db
    other_database = rig.custom_db if database_kind == "default" else rig.database
    # This ownership test needs an enabled, terminal synthetic no-change writer.
    # Source capture and the completion/archive/counter writes remain real.
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", lambda: SimpleNamespace(
        enabled=True, schedule=lambda **kwargs: False, status=SimpleNamespace(last_outcome="no_change"),
    ))
    queued, release = asyncio.Event(), asyncio.Event()
    preparations, completions = [], []
    actual_prepare = character_context.CharacterContextService.prepare_turn
    actual_complete = character_context.CharacterContextService.complete_turn

    async def prepare(service, *args, **kwargs):
        prepared = await actual_prepare(service, *args, **kwargs)
        preparations.append((service, prepared, get_request_context()))
        return prepared

    async def complete(service, prepared, *args, **kwargs):
        outcome = await actual_complete(service, prepared, *args, **kwargs)
        completions.append((service, prepared, outcome, get_request_context()))
        return outcome

    monkeypatch.setattr(character_context.CharacterContextService, "prepare_turn", prepare)
    monkeypatch.setattr(character_context.CharacterContextService, "complete_turn", complete)

    class QueueTiming:
        def priority_for(self, platform, session_type):
            return 5

        async def check_rate_limits(self, platform, session_id, identity):
            assert identity == OWNER

        async def submit(self, job, *, session_id, priority):
            assert session_id == "fiction-budget-session" and priority == 5
            queued.set()
            await release.wait()
            return await job()

    local = await select_context(rig, "vllm")
    with use_request_context(local):
        old_default = character_context.get_default_character_context_service()
        assert old_default._history_max_chars == 16000
    application = create_app(RuntimeContainer(
        db=database, is_pg_mode=lambda: False, inference_runtime=QueueTiming(), startup_env={},
    ))
    # Resolve the actual HTTP dependency while local is selected. Its owning
    # database must survive queueing, while its provider budget resolves later.
    service = generate.get_request_chat_generation_service(Request({"type": "http", "app": application}))
    request = MessageRequest(
        message=QUERY, characterId=ROLE, loraName="default", platform="web", adapter="web-character",
        sessionId="fiction-budget-session", sessionType="private", conversationType="private",
        senderId=OWNER, userId=OWNER, sourceMessageId=f"fiction-queued-{database_kind}",
        traceId=f"fiction-budget-{database_kind}",
    )
    pending = asyncio.create_task(generate.generate_reply(request, {"id": OWNER, "user_id": OWNER}, service))
    try:
        await asyncio.wait_for(queued.wait(), 1)
        assert not preparations and not completions and not rig.answers
        rig.manager._current_provider = model_manager.ModelProvider.OPENAI_COMPAT
        release.set()
        result = await asyncio.wait_for(pending, 5)
        assert result.reply == BODY and result.modelInvoked is True
        assert len(preparations) == len(completions) == len(rig.answers) == 1
        prepared_service, prepared, context = preparations[0]
        completed_service, completed_prepared, outcome, completed_context = completions[0]
        assert prepared_service is completed_service and completed_prepared is prepared
        assert prepared_service._memory_repo._database is database
        assert prepared_service._message_repo._database is database
        assert prepared_service._history_max_chars == 65536
        assert prepared.memory_budget["budget_chars"] == 32768
        assert reference_packet(prepared)["source_observation"] is True
        assert context is completed_context is rig.answers[0]["context"]
        assert context.provider_name == "openai_compat"
        assert outcome.interaction_count == 1 and not outcome.failed_writes
        assert old_default._history_max_chars == 16000
        assert [row["provider"] for row in rig.reviews] == ["openai_compat"]
        saved = database.execute_sql("SELECT * FROM messages ORDER BY id")
        invocations = database.execute_sql("SELECT * FROM model_invocations ORDER BY id")
        assert len(saved) == len(invocations) == 1
        assert saved[0]["message"] == QUERY and saved[0]["reply"] == BODY
        assert saved[0]["traceId"] == request.traceId
        assert invocations[0]["errorType"] == "" and invocations[0]["completionTokens"] > 0
        relationship = await DatabaseCharacterMemoryRepository(database).get_relationship_record(ROLE, rig.scope)
        assert relationship["interaction_count"] == 1
        assert other_database.execute_sql("SELECT * FROM messages") == []
        assert other_database.execute_sql("SELECT * FROM model_invocations") == []
        assert await DatabaseCharacterMemoryRepository(other_database).get_relationship_record(ROLE, rig.scope) is None
        assert get_request_context() is None
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
