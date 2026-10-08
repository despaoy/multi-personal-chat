"""Only delivered memory durability, exact job ownership and receipt fences."""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from character.context_builder import build_user_scope
from character.models import RelationshipState
from db.database import SQLiteDB
from db.schemas import MessageRequest
from services import delivery_memory as dm
from services.character_context import _TurnOutcome
from services.turn_completion import TurnCompletionRuntime


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        database = SQLiteDB(tmp_path_factory.mktemp("delivery-memory") / "records.sqlite")
    else:
        url = os.environ["STAGE34_DELIVERY_PG_URL"]
        assert "/stage3_stage34_delivery_guards?" in url and "port=25433" in url
        with pytest.MonkeyPatch.context() as env:
            env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        database = SyncPgAdapter(PgDatabase(url))
    yield database
    (database.close if request.param == "pg" else database.close_connection)()


def generated(database):
    key, owner = uuid.uuid4().hex * 2, uuid.uuid4().hex
    request = MessageRequest(
        message="我本人长期喜欢深蓝色油墨，这是明确的个人偏好。",
        platform="qq",
        adapter="unit",
        userId=owner,
        senderId=owner,
        sessionId=owner,
        conversationId=owner,
        sourceMessageId="source-" + key,
        traceId=owner,
    )
    scope = build_user_scope("qq", "unit", owner, owner, "private")
    prepared = SimpleNamespace(
        character_id="tsukiyashiro_kisaki",
        user_scope=scope,
        history=({"role": "user", "content": "生成时的原始历史"},),
        relationship=RelationshipState(preferred_address="杉岚"),
        received_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        compiled=SimpleNamespace(used_memory_ids=("selected-before-delivery",)),
        memory_operation_receipt=None,
    )
    stored = dict(
        reply=dict(replyText="原始模型回复"),
        context=dict(
            request=request.model_dump(),
            message_saved=True,
            character_id=prepared.character_id,
            completion_snapshot=dm.freeze_completion(prepared),
        ),
    )
    assert database.integration_receipt("claim", key=key, owner=owner, now=0, expires_at=0)
    assert database.integration_receipt(
        "finish", key=key, owner=owner, status="generated", response=json.dumps(stored, ensure_ascii=False)
    )
    return key, owner, stored


def deliver(database, key, owner, stored):
    record = database.integration_receipt("get", key=key)
    response, state = dm.delivery_envelope(stored)
    assert state == "pending"
    assert database.integration_receipt(
        "delivery_memory", key=key, owner=owner, expected=record["response"], response=response, expires_at=0
    )


def test_delivery_and_pending_are_atomic_and_duplicates_cannot_reset(database):
    key, owner, stored = generated(database)
    deliver(database, key, owner, stored)
    actual = database.integration_receipt("get", key=key)
    assert actual["status"] == "delivered" and json.loads(actual["response"])["memory_completion_state"] == "pending"
    assert not database.integration_receipt(
        "delivery_memory", key=key, owner=owner, expected=actual["response"], response="reset", expires_at=0
    )
    assert not database.integration_receipt("delivery", key=key, owner=owner, status="delivery_failed")
    # Remove this fixture from worker selection without changing physical truth.
    assert database.integration_receipt(
        "memory_update",
        key=key,
        owner=owner,
        expected=actual["response"],
        response=dm.encode(stored, "completed"),
        expires_at=0,
    )


@pytest.mark.parametrize("field", ["key", "owner", "expected"])
def test_new_memory_cas_retains_key_owner_and_response_authority(database, field):
    key, owner, stored = generated(database)
    deliver(database, key, owner, stored)
    row = database.integration_receipt("get", key=key)
    params = dict(
        key=key, owner=owner, expected=row["response"], response=dm.encode(stored, "running"), expires_at=20, now=0
    )
    params[field] += "wrong"
    assert not database.integration_receipt("memory_claim", **params)
    assert database.integration_receipt("get", key=key) == row
    assert database.integration_receipt(
        "memory_update",
        key=key,
        owner=owner,
        expected=row["response"],
        response=dm.encode(stored, "completed"),
        expires_at=0,
    )


def test_live_renewed_lease_cannot_be_claimed_from_stale_candidate(database):
    key, owner, stored = generated(database)
    deliver(database, key, owner, stored)
    row = database.integration_receipt("get", key=key)
    running = dm.encode(stored, "running")
    assert database.integration_receipt(
        "memory_claim", key=key, owner=owner, expected=row["response"], response=running, expires_at=5, now=0
    )
    assert database.integration_receipt("memory_renew", key=key, owner=owner, expected=running, expires_at=30)
    assert not database.integration_receipt(
        "memory_claim", key=key, owner=owner, expected=running, response="bad", expires_at=40, now=6
    )
    assert database.integration_receipt(
        "memory_claim", key=key, owner=owner, expected=running, response=running, expires_at=60, now=31
    )
    assert database.integration_receipt(
        "memory_update", key=key, owner=owner, expected=running, response=dm.encode(stored, "completed"), expires_at=0
    )


@pytest.mark.parametrize("invalid", ["scope", "character", "clock", "legacy"])
def test_snapshot_preserves_authority_and_never_invents_legacy_clock(database, invalid):
    key, owner, stored = generated(database)
    context = stored["context"]
    if invalid == "scope":
        context["completion_snapshot"]["user_scope"]["sender_id"] = "other"
    if invalid == "character":
        context["completion_snapshot"]["character_id"] = "other"
    if invalid == "clock":
        context["completion_snapshot"]["received_at"] = "2026-10-01T00:00:00"
    if invalid == "legacy":
        context.pop("completion_snapshot")
    _, state = dm.delivery_envelope(stored)
    assert state == ("legacy_unknown" if invalid == "legacy" else "blocked")
    assert not any(
        r["receipt_key"] == key for r in database.integration_receipt("memory_pending", now=time.time(), limit=100)
    )


@pytest.mark.asyncio
async def test_busy_ack_persists_then_automatic_worker_waits_for_exact_semantic_receipt(database, monkeypatch):
    from api import integrations

    runtime = TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(dm, "get_turn_completion_runtime", lambda: runtime)
    monkeypatch.setattr(dm, "POLL_SECONDS", 0.01)
    key, owner, stored = generated(database)
    exact = asyncio.Event()
    calls = []

    async def complete(prepared, turn, reply, **kwargs):
        calls.append((prepared, turn, reply, kwargs))
        exact.set()
        return _TurnOutcome(
            source_capture="recorded", memory_enrichment_scheduled=True, memory_enrichment_status="queued_hot"
        )

    service = SimpleNamespace(complete_turn=complete)
    monkeypatch.setattr(integrations, "_request_container_db", lambda _: database)
    monkeypatch.setattr(integrations, "_request_character_service", lambda _: service)
    monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKEN", "synthetic-token")
    monkeypatch.setenv("ASTRBOT_INTEGRATION_TOKENS", "")
    monkeypatch.setenv("INTEGRATION_SIGNATURE_REQUIRED", "false")
    request = SimpleNamespace(headers={"X-Integration-Token": "synthetic-token"})
    payload = integrations.DeliveryAcknowledgement(receiptId=key, deliveryToken=owner, status="delivered")
    occupied = runtime.reserve()
    try:
        ack = await integrations.acknowledge_delivery(payload, request)
        assert ack == dict(acknowledged=True, changed=True, memoryStatus="pending")
        assert not calls
        occupied.release()
        await asyncio.wait_for(exact.wait(), 2)
        prepared, turn, reply, kw = calls[0]
        assert prepared.received_at == turn.received_at == datetime(2026, 10, 1, tzinfo=timezone.utc)
        assert prepared.history == ({"role": "user", "content": "生成时的原始历史"},)
        assert prepared.compiled.used_memory_ids == ("selected-before-delivery",) and reply == "原始模型回复"
        worker = dm._workers[id(database)]
        assert not kw["memory_receipt"].done()
        duplicate = await integrations.acknowledge_delivery(payload, request)
        assert duplicate["memoryStatus"] == "pending" and not duplicate["changed"] and len(calls) == 1
        kw["memory_receipt"].set_result(dict(status="saved", source_capture="recorded", persisted=1))
        await worker.wait_idle(timeout=3)
        final = await integrations.acknowledge_delivery(payload, request)
        assert final["memoryStatus"] == "completed" and not final["changed"] and len(calls) == 1
    finally:
        occupied.release()
        await dm.shutdown_delivery_memory()
        await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_expired_restart_receipt_resumes_without_answer_regeneration(database, monkeypatch):
    runtime = TurnCompletionRuntime(capacity=1)
    monkeypatch.setattr(dm, "get_turn_completion_runtime", lambda: runtime)
    monkeypatch.setattr(dm, "POLL_SECONDS", 0.01)
    key, owner, stored = generated(database)
    deliver(database, key, owner, stored)
    row = database.integration_receipt("get", key=key)
    stored["context"]["memory_completion"].update(state="running", turn_finished=True, attempt_id="abandoned")
    assert database.integration_receipt(
        "memory_claim",
        key=key,
        owner=owner,
        expected=row["response"],
        response=dm.encode(stored, "running"),
        now=0,
        expires_at=0,
    )
    calls = []

    async def complete(prepared, turn, reply, **kwargs):
        calls.append(kwargs)
        return _TurnOutcome(source_capture="recorded", memory_enrichment_status="no_change")

    worker = dm.ensure_delivery_memory_worker(database, SimpleNamespace(complete_turn=complete))
    try:
        await worker.wait_idle(timeout=3)
        assert len(calls) == 1 and calls[0]["memory_retry_only"]
        final = database.integration_receipt("get", key=key)
        assert (
            final["status"] == "delivered" and json.loads(final["response"])["memory_completion_state"] == "completed"
        )
        assert json.loads(final["response"])["reply"]["replyText"] == "原始模型回复"
    finally:
        await dm.shutdown_delivery_memory()
        await runtime.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_completion_write_failure_is_blocked_with_preserved_semantic_receipt(database, monkeypatch):
    runtime = TurnCompletionRuntime(2)
    monkeypatch.setattr(dm, "get_turn_completion_runtime", lambda: runtime)
    monkeypatch.setattr(dm, "POLL_SECONDS", 0.01)
    key, owner, stored = generated(database)
    deliver(database, key, owner, stored)
    calls = []

    async def complete(prepared, turn, reply, **kwargs):
        calls.append(kwargs)
        kwargs['memory_receipt'].set_result({'status': 'saved', 'source_capture': 'recorded'})
        return _TurnOutcome(failed_writes=('preferred_address',), source_capture='recorded',
                            memory_enrichment_scheduled=True, memory_enrichment_status='queued_hot')

    worker = dm.ensure_delivery_memory_worker(database, SimpleNamespace(complete_turn=complete))
    try:
        await worker.wait_idle(timeout=3)
        row = database.integration_receipt('get', key=key)
        response = json.loads(row['response'])
        marker = response['context']['memory_completion']
        assert row['status'] == 'delivered' and response['memory_completion_state'] == 'blocked'
        assert marker['reason'] == 'completion_write_failed'
        assert marker['completion_outcome']['failed_writes'] == ['preferred_address']
        assert marker['semantic_receipt']['status'] == 'saved'
        assert response['reply']['replyText'] == '原始模型回复'
        await worker.wait_idle(timeout=3)
        assert len(calls) == 1
    finally:
        await dm.shutdown_delivery_memory()
        await runtime.shutdown(timeout=1)
