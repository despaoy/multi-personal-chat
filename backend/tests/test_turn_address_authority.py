"""Only turn-derived address authority and actual owner-lock interleaving."""

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from character.models import build_user_scope
from character.models import RelationshipState
from db import memory_source
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService, TurnInput


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        db = SQLiteDB(tmp_path_factory.mktemp("turn-address") / "records.sqlite")
    else:
        url = os.environ["STAGE35_ADDRESS_PG_URL"]
        assert "/stage3_stage35_address_guards?" in url and "port=25433" in url
        with pytest.MonkeyPatch.context() as env:
            env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        db = SyncPgAdapter(PgDatabase(url))
    yield db
    (db.close if request.param == "pg" else db.close_connection)()


def source(database):
    user = "address-" + uuid.uuid4().hex
    fields = ("tsukiyashiro_kisaki", "qq", "unit-address", user, "private", user)
    source_id = "source-" + uuid.uuid4().hex
    stamp = datetime.now(timezone.utc)
    database.upsert_character_relationship(*fields, "familiar", "知禾", "手动关系摘要", interaction_count=11)
    assert (
        database.capture_memory_source(
            *fields,
            source_message_id=source_id,
            body="请叫我柚汀。这个称呼只用于对话，不是更改我的姓名。",
            observed_at=stamp,
        )
        == "recorded"
    )
    return fields, source_id, stamp


def test_current_turn_changes_only_address_and_preserves_manual_fields(database):
    fields, sid, stamp = source(database)
    before = database.get_character_relationship(*fields)
    result = database.set_character_address_from_turn(*fields, source_message_id=sid, observed_at=stamp, address="柚汀")
    assert result["status"] == "written"
    after = database.get_character_relationship(*fields)
    assert after["preferred_address"] == "柚汀"
    for field in ["relationship_stage", "summary", "interaction_count", "created_at"]:
        assert after[field] == before[field]


@pytest.mark.parametrize("denied", ["stale", "revoked", "conflict"])
def test_turn_address_rejects_erased_identity_or_replaced_clock(database, denied):
    fields, sid, stamp = source(database)
    if denied in {"stale", "revoked"}:
        database.clear_character_memories(*fields)
    if denied in {"revoked", "conflict"}:
        stamp = datetime.now(timezone.utc) + timedelta(seconds=1)
    before = database.get_character_relationship(*fields)
    result = database.set_character_address_from_turn(*fields, source_message_id=sid, observed_at=stamp, address="柚汀")
    assert result["status"] == denied and database.get_character_relationship(*fields) == before


@pytest.mark.parametrize("invalid", ["naive_clock", "empty_source", "expanded_scope", "narrative", "empty_address"])
def test_turn_address_requires_explicit_valid_receipt_scope(database, invalid):
    fields, sid, stamp = source(database)
    fields = list(fields)
    original = tuple(fields)
    kwargs = dict(source_message_id=sid, observed_at=stamp, address="柚汀")
    if invalid == "naive_clock":
        kwargs["observed_at"] = stamp.replace(tzinfo=None)
    if invalid == "empty_source":
        kwargs["source_message_id"] = ""
    if invalid == "expanded_scope":
        fields[0] = "*"
    if invalid == "narrative":
        fields[2] = "narrative"
    if invalid == "empty_address":
        kwargs["address"] = ""
    before = database.get_character_relationship(*original)
    with pytest.raises(ValueError):
        database.set_character_address_from_turn(*fields, **kwargs)
    assert database.get_character_relationship(*original) == before


@pytest.mark.asyncio
async def test_complete_turn_does_not_write_rejected_source_address(database, monkeypatch):
    from character import memory_llm

    fields, sid, stamp = source(database)
    database.clear_character_memories(*fields)
    repo = DatabaseCharacterMemoryRepository(database)
    scope = build_user_scope(fields[1], fields[2], fields[3], fields[5], fields[4])
    service = CharacterContextService.__new__(CharacterContextService)
    service._memory_repo = repo
    scheduler = SimpleNamespace(
        enabled=True,
        status=SimpleNamespace(last_outcome="write_gate"),
        schedule=lambda **_: pytest.fail("Stale source must not enqueue semantic work"),
    )
    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: scheduler)
    prepared = SimpleNamespace(
        character_id=fields[0],
        user_scope=scope,
        received_at=stamp,
        memory_operation_receipt=None,
        compiled=SimpleNamespace(used_memory_ids=()),
        history=(),
        relationship=RelationshipState(stage="familiar", preferred_address="知禾"),
    )
    turn = TurnInput(
        message="请叫我柚汀。这个称呼只用于对话，不是更改我的姓名。",
        platform=scope.platform,
        adapter=scope.adapter,
        sender_id=scope.sender_id,
        conversation_id=scope.conversation_id,
        conversation_type="private",
        received_at=stamp,
    )
    result = await service.complete_turn(prepared, turn, source_message_id=sid)
    assert result.source_capture == "stale" and not result.memory_enrichment_scheduled
    assert database.get_character_relationship(*fields)["preferred_address"] == "知禾"


@pytest.mark.asyncio
async def test_pg_real_owner_lock_erasure_wins_before_pending_address_sql(database):
    if not hasattr(database, "_pg"):
        pytest.skip("Real PostgreSQL row-lock case only")
    import asyncpg
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    fields, sid, stamp = source(database)
    before = database.get_character_relationship(*fields)
    url = os.environ["STAGE35_ADDRESS_PG_URL"]
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine)
    # Observe locks in the same explicitly configured, isolated test database.
    assert "/stage3_stage35_address_guards?" in url and "port=25433" in url
    connect_args, connect_kwargs = engine.dialect.create_connect_args(engine.url)
    c = await asyncpg.connect(*connect_args, **connect_kwargs)
    task = None
    try:
        async with sessions() as session:
            await memory_source.run_postgres(
                session, memory_source.lock_owner(memory_source.source_scope(*fields), postgres=True)
            )
            task = asyncio.create_task(
                asyncio.to_thread(
                    database.set_character_address_from_turn,
                    *fields,
                    source_message_id=sid,
                    observed_at=stamp,
                    address="柚汀",
                ),
                name="actual-pg-address-waiter",
            )
            for _ in range(80):
                waiting = await c.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname='stage3_stage35_address_guards' AND wait_event_type='Lock' AND query LIKE '%memory_source_fences%'"
                )
                if waiting:
                    break
                await asyncio.sleep(0.025)
            assert waiting and not task.done(), "Actual new address SQL must be waiting on real owner lock"
            await memory_source.run_postgres(
                session, memory_source.revoke_plan(memory_source.source_scope(*fields), [], clear=True)
            )
            await session.commit()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result["status"] == "stale" and database.get_character_relationship(*fields) == before
        assert database.list_memory_sources(*fields, source_message_ids=(sid,), limit=1) == []
    finally:
        if task is not None and not task.done():
            await asyncio.wait_for(asyncio.shield(task), 5)
        await c.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_final_address_rejection_overrides_earlier_semantic_success(database, monkeypatch):
    import json

    from db.schemas import MessageRequest
    from services import delivery_memory as dm
    from services.character_context import _TurnOutcome
    from services.turn_completion import TurnCompletionRuntime

    fields, sid, stamp = source(database)
    scope = build_user_scope(fields[1], fields[2], fields[3], fields[5], fields[4])
    prepared = SimpleNamespace(
        character_id=fields[0],
        user_scope=scope,
        received_at=stamp,
        memory_operation_receipt=None,
        compiled=SimpleNamespace(used_memory_ids=()),
        history=(),
        relationship=RelationshipState(),
    )
    msg = MessageRequest(
        message="请叫我柚汀。",
        platform=scope.platform,
        adapter=scope.adapter,
        senderId=scope.sender_id,
        userId=scope.sender_id,
        conversationId=scope.conversation_id,
        sessionId=scope.conversation_id,
        sourceMessageId=sid,
    )
    stored = dict(
        reply=dict(replyText="原始回复"),
        context=dict(
            request=msg.model_dump(),
            message_saved=True,
            character_id=fields[0],
            completion_snapshot=dm.freeze_completion(prepared),
        ),
    )
    key, owner = uuid.uuid4().hex * 2, uuid.uuid4().hex
    assert database.integration_receipt("claim", key=key, owner=owner, now=0, expires_at=0)
    assert database.integration_receipt("finish", key=key, owner=owner, status="generated", response=json.dumps(stored))
    initial = database.integration_receipt("get", key=key)
    pending, _ = dm.delivery_envelope(stored)
    assert database.integration_receipt(
        "delivery_memory", key=key, owner=owner, expected=initial["response"], response=pending, expires_at=0
    )
    runtime = TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(dm, "get_turn_completion_runtime", lambda: runtime)
    monkeypatch.setattr(dm, "POLL_SECONDS", 0.01)

    async def complete(*_, memory_receipt, **kwargs):
        memory_receipt.set_result(dict(status="saved", source_capture="recorded", persisted=1))
        return _TurnOutcome(source_capture="stale", memory_enrichment_scheduled=True)

    worker = dm.ensure_delivery_memory_worker(database, SimpleNamespace(complete_turn=complete))
    try:
        await worker.wait_idle(timeout=3)
        row = database.integration_receipt("get", key=key)
        assert row["status"] == "delivered" and json.loads(row["response"])["memory_completion_state"] == "blocked"
    finally:
        await dm.shutdown_delivery_memory()
        await runtime.shutdown(timeout=1)



@pytest.mark.asyncio
@pytest.mark.parametrize("operation_status", ["erased", "failed"])
async def test_explicit_completion_never_reinterprets_erasure_as_address(database, monkeypatch, operation_status):
    import json
    from unittest.mock import AsyncMock

    from api import generate
    from character.memory_extractor import extract_preferred_address
    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, is_memory_erasure_request
    from db.schemas import MessageRequest
    from repositories.character_memory import relationship_from_record

    fields, old_source, _ = source(database)
    role, platform, adapter, sender, kind, conversation = fields
    scope = build_user_scope(platform, adapter, sender, conversation, kind)
    repository = DatabaseCharacterMemoryRepository(database)
    message = "请从长期记忆中删除我让你叫我柚汀的那条原话记录。"
    assert is_memory_erasure_request(message) and extract_preferred_address(message)
    received = datetime.now(timezone.utc)
    current_source = "operation-" + uuid.uuid4().hex

    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            payload = json.loads(messages[1]["content"])
            assert old_source in {row["source_id"] for row in payload["source_erasure_candidates"]}
            return (
                json.dumps({"memories": [], "erase_source_ids": [old_source]})
                if operation_status == "erased"
                else "{invalid"
            )

        async def close(self):
            pass

    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unit-only", "unit-only"), completion=completion)
    try:
        receipt = await scheduler.schedule_and_wait(
            repository=repository,
            character_id=role,
            user_scope=scope,
            message=message,
            rule_hints=(),
            source_message_id=current_source,
            observed_at=received,
        )
        assert receipt["status"] == operation_status
        relationship = relationship_from_record(await repository.get_relationship_record(role, scope))
        before = database.get_character_relationship(*fields)
        prepared = SimpleNamespace(
            character_id=role,
            user_scope=scope,
            received_at=received,
            memory_operation_receipt=receipt,
            relationship=relationship,
            history=(),
            compiled=SimpleNamespace(used_memory_ids=()),
        )
        service = CharacterContextService(object(), repository, object(), source_recall_enabled=False)
        writes = AsyncMock(wraps=repository.set_address_from_turn)
        monkeypatch.setattr(repository, "set_address_from_turn", writes)
        monkeypatch.setattr(
            "character.memory_llm.get_memory_enrichment_scheduler",
            lambda: pytest.fail("An executed operation must not initialize another writer"),
        )
        outcomes = []
        complete_turn = service.complete_turn

        async def complete(*args, **kwargs):
            outcome = await complete_turn(*args, **kwargs)
            outcomes.append(outcome)
            return outcome

        monkeypatch.setattr(service, "complete_turn", complete)
        request = MessageRequest(
            message=message,
            characterId=role,
            platform=platform,
            adapter=adapter,
            senderId=sender,
            conversationId=conversation,
            sessionId=conversation,
            conversationType=kind,
            sourceMessageId=current_source,
        )
        warning = await generate._complete_character_turn(prepared, request, character_service=service)
        after = database.get_character_relationship(*fields)
        assert after["preferred_address"] == before["preferred_address"] == "知禾"
        assert after["relationship_stage"] == before["relationship_stage"] == "familiar"
        assert after["summary"] == before["summary"] and after["interaction_count"] == before["interaction_count"] + 1
        assert len(outcomes) == 1 and outcomes[0].memory_enrichment_status == receipt["status"]
        assert outcomes[0].source_capture == receipt.get("source_capture", "")
        assert (
            outcomes[0].stage == relationship.stage and outcomes[0].preferred_address == relationship.preferred_address
        )
        assert not outcomes[0].failed_writes and not outcomes[0].memory_enrichment_scheduled
        writes.assert_not_awaited()
        assert (warning is None) if operation_status == "erased" else ("失败" in warning and "称呼" not in warning)
        rows = await repository.list_sources(role, scope)
        assert [row["source_message_id"] for row in rows] == ([] if operation_status == "erased" else [old_source])
        assert completion.calls == 1
    finally:
        await scheduler.shutdown()
