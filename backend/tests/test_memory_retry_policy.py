"""Bounded delivered-memory recovery with explicit, isolated provider controls."""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services import delivery_memory as dm
from services.character_context import CharacterContextService
from services.turn_completion import TurnCompletionRuntime
from test_delivery_memory import deliver, generated

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, OpenAICompatibleMemoryCompletion
from character.models import UserScope
from db.database import SQLiteDB
from db.memory_claim_guard import MemoryClaimConflict


def provider_error(kind):
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    if kind == "timeout":
        return httpx.ReadTimeout("synthetic provider timeout", request=request)
    if kind == "connect":
        return httpx.ConnectError("synthetic connection failure", request=request)
    status = int(kind)
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("synthetic provider rejection", request=request, response=response)


def proposal(body, value="深蓝色油墨"):
    return json.dumps({"memories": [dict(
        attributed_to="user", operation="ADD", kind="like", value=value,
        evidence=body, confidence=.99,
    )]}, ensure_ascii=False)


@pytest.fixture
async def delivered_memory(tmp_path, monkeypatch):
    """Use the real chain while parking only the automatic polling loop.

    Each step invokes _process with a real runtime reservation. The database's
    existing pending backoff is advanced explicitly, without sleeping or
    bypassing its receipt compare-and-swap authority.
    """
    rig = SimpleNamespace(path=tmp_path / "delivery-retry.sqlite")
    rig.database = SQLiteDB(rig.path)
    rig.key, rig.owner, stored = generated(rig.database)
    deliver(rig.database, rig.key, rig.owner, stored)
    rig.snapshot, _ = dm.thaw_completion(stored["context"])
    rig.body = stored["context"]["request"]["message"]
    rig.source_id = stored["context"]["request"]["sourceMessageId"]
    rig.original_reply = stored["reply"]
    rig.completion = SimpleNamespace(
        complete=AsyncMock(return_value=proposal(rig.body)), close=AsyncMock(),
    )
    rig.runtime = TurnCompletionRuntime(1)
    monkeypatch.setattr(dm, "get_turn_completion_runtime", lambda: rig.runtime)
    monkeypatch.setattr(dm, "POLL_SECONDS", 3600)
    monkeypatch.setattr("character.memory_llm.get_memory_enrichment_scheduler", lambda: rig.scheduler)

    async def open_worker():
        rig.repo = DatabaseCharacterMemoryRepository(rig.database)
        rig.scheduler = MemoryEnrichmentScheduler(
            config=MemoryLlmConfig(True, "unused", "fixture", queue_size=2, idle_seconds=0),
            completion=rig.completion,
            embedding_provider=SimpleNamespace(embed_texts=Mock(
                side_effect=AssertionError("This test must never call an embedding model"))),
        )
        parked = rig.runtime.reserve()
        try:
            rig.worker = dm.DeliveryMemoryWorker(
                rig.database, CharacterContextService(object(), rig.repo, object()))
            # The loop observes the occupied runtime and enters its normal pause.
            await asyncio.sleep(0)
        finally:
            parked.release()

    def stored_response():
        return json.loads(rig.database.integration_receipt("get", key=rig.key)["response"])

    def write_response(response, state):
        row = rig.database.integration_receipt("get", key=rig.key)
        assert rig.database.integration_receipt(
            "memory_update", key=rig.key, owner=rig.owner, expected=row["response"],
            response=dm.encode(response, state), expires_at=0,
        )

    async def step():
        row = rig.database.integration_receipt("get", key=rig.key)
        # Expire only the server's existing lease/backoff; keep its response.
        assert rig.database.integration_receipt(
            "memory_update", key=rig.key, owner=rig.owner, expected=row["response"],
            response=row["response"], expires_at=0,
        )
        row = rig.database.integration_receipt("get", key=rig.key)
        slot = rig.runtime.reserve()
        try:
            await rig.worker._process(row, rig.runtime, slot)
        finally:
            slot.release()
        return stored_response()

    async def rebuild():
        await rig.worker.shutdown(timeout=1)
        if not rig.scheduler.status.closed:
            await rig.scheduler.shutdown(timeout=1)
        rig.database.close_connection()
        rig.database = SQLiteDB(rig.path)
        await open_worker()

    def selectable():
        return any(row["receipt_key"] == rig.key for row in rig.database.integration_receipt(
            "memory_pending", now=float("inf"), limit=100))

    rig.step = step
    rig.rebuild = rebuild
    rig.stored_response = stored_response
    rig.write_response = write_response
    rig.selectable = selectable
    await open_worker()
    try:
        yield rig
    finally:
        await rig.worker.shutdown(timeout=1)
        await rig.runtime.shutdown(timeout=1)
        if not rig.scheduler.status.closed:
            await rig.scheduler.shutdown(timeout=1)
        rig.database.close_connection()


async def assert_saved_once(rig, response, attempts):
    marker = response["context"]["memory_completion"]
    assert response["memory_completion_state"] == "completed"
    assert marker["attempts"] == attempts
    assert marker["semantic_receipt"]["status"] == "saved"
    assert marker["semantic_receipt"]["persisted"] == 1
    records = await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert len(records) == 1 and "深蓝色油墨" in records[0]["content"]
    assert records[0]["source_message_id"] == rig.source_id
    assert records[0]["source_message_ids"] == [rig.source_id]
    assert datetime.fromisoformat(records[0]["observed_at"]) == rig.snapshot.received_at
    assert await rig.repo.list_sources(rig.snapshot.character_id, rig.snapshot.user_scope) == [dict(
        source_message_id=rig.source_id, body=rig.body,
        observed_at=rig.snapshot.received_at.isoformat(timespec="microseconds"),
    )]
    relationship = await rig.repo.get_relationship_record(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert relationship["interaction_count"] == 1
    assert response["reply"] == rig.original_reply
    assert not rig.selectable()


@pytest.mark.parametrize("failure", ["non_json", "401", "501"])
async def test_permanent_provider_failure_is_terminal_after_one_call(delivered_memory, failure):
    rig = delivered_memory
    if failure == "non_json":
        rig.completion.complete.return_value = "not valid JSON"
    else:
        rig.completion.complete.side_effect = provider_error(failure)
    response = await rig.step()
    marker = response["context"]["memory_completion"]
    receipt = marker["semantic_receipt"]
    assert response["memory_completion_state"] == "blocked"
    assert marker["reason"] == "semantic_failure" and marker["attempts"] == 1
    assert receipt["status"] == "failed" and receipt["persisted"] == 0
    assert receipt["stage"] == ("proposal_validation" if failure == "non_json" else "proposal_generation")
    assert receipt.get("retryable") is not True
    assert rig.completion.complete.await_count == 1 and not rig.selectable()
    assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
    await rig.rebuild()
    assert not rig.selectable() and rig.completion.complete.await_count == 1


@pytest.mark.parametrize("failure", ["503", "timeout"])
async def test_transient_failures_have_a_persistent_three_attempt_budget(delivered_memory, failure):
    rig = delivered_memory
    assert dm.MAX_COMPLETION_ATTEMPTS == 3
    rig.completion.complete.side_effect = provider_error(failure)
    for attempt in range(1, 4):
        response = await rig.step()
        marker = response["context"]["memory_completion"]
        receipt = marker["semantic_receipt"]
        assert marker["attempts"] == attempt
        assert receipt["status"] == "failed" and receipt["stage"] == "proposal_generation"
        assert receipt["retryable"] is True and receipt["persisted"] == 0
        assert rig.completion.complete.await_count == attempt
        if attempt < 3:
            assert response["memory_completion_state"] == "pending" and rig.selectable()
            # Reopen the actual database and worker; process-local state is gone.
            await rig.rebuild()
            assert rig.stored_response()["context"]["memory_completion"]["attempts"] == attempt
        else:
            assert response["memory_completion_state"] == "blocked"
            assert marker["reason"] == "retry_limit_reached" and not rig.selectable()
    relationship = await rig.repo.get_relationship_record(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert relationship["interaction_count"] == 1
    assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)


@pytest.mark.parametrize("failure", ["503", "timeout", "connect", "429"])
async def test_transient_failure_can_recover_without_changing_source_or_clock(delivered_memory, failure):
    rig = delivered_memory
    rig.completion.complete.side_effect = [provider_error(failure), proposal(rig.body)]
    first = await rig.step()
    assert first["memory_completion_state"] == "pending"
    assert first["context"]["memory_completion"]["attempts"] == 1
    await rig.rebuild()
    final = await rig.step()
    await assert_saved_once(rig, final, attempts=2)
    assert rig.completion.complete.await_count == 2


async def test_real_partial_write_is_preserved_and_never_replayed(delivered_memory, monkeypatch):
    rig = delivered_memory
    rig.body = "我本人喜欢深蓝色油墨；我本人喜欢厚棉纸。"
    response = rig.stored_response()
    response["context"]["request"]["message"] = rig.body
    rig.write_response(response, "pending")
    rig.completion.complete.return_value = json.dumps({"memories": [
        dict(attributed_to="user", operation="ADD", kind="like", value=value,
             evidence=evidence, confidence=.99)
        for value, evidence in (("深蓝色油墨", "我本人喜欢深蓝色油墨"), ("厚棉纸", "我本人喜欢厚棉纸"))
    ]}, ensure_ascii=False)
    actual_append = rig.repo.append_claim

    async def fail_second(*args, **kwargs):
        if "厚棉纸" in args[2].content:
            raise OSError("synthetic second-write outage")
        return await actual_append(*args, **kwargs)

    monkeypatch.setattr(rig.repo, "append_claim", fail_second)
    final = await rig.step()
    marker = final["context"]["memory_completion"]
    receipt = marker["semantic_receipt"]
    assert final["memory_completion_state"] == "blocked"
    assert marker["reason"] == "partial_write" and marker["attempts"] == 1
    assert receipt["status"] == "partial" and receipt["accepted"] == 2 and receipt["persisted"] == 1
    assert receipt["operation_outcomes"] == ["saved", "failed"]
    records = await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
    assert len(records) == 1 and "深蓝色油墨" in records[0]["content"]
    assert rig.completion.complete.await_count == 1 and not rig.selectable()
    # Before this fix, the same real partial receipt was persisted as pending.
    # Emulate that upgrade state, preserving the already committed claim.
    marker.pop("attempts")
    marker.pop("execution_started")
    rig.write_response(final, "pending")
    await rig.rebuild()
    final = await rig.step()
    marker = final["context"]["memory_completion"]
    assert final["memory_completion_state"] == "blocked" and marker["reason"] == "partial_write"
    assert marker["semantic_receipt"] == receipt
    assert not rig.selectable() and rig.completion.complete.await_count == 1
    assert len(await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)) == 1


@pytest.mark.parametrize("started", ["missing", True])
async def test_unknown_running_execution_cannot_reuse_an_older_retryable_receipt(delivered_memory, started):
    rig = delivered_memory
    response = rig.stored_response()
    marker = response["context"]["memory_completion"]
    marker.update(state="running", attempts=1, turn_finished=True, semantic_receipt=dict(
        status="failed", stage="proposal_generation", retryable=True, persisted=0,
    ))
    if started is True:
        marker["execution_started"] = True
    rig.write_response(response, "running")
    final = await rig.step()
    marker = final["context"]["memory_completion"]
    assert final["memory_completion_state"] == "blocked"
    assert marker["reason"] == "execution_outcome_unknown" and marker["attempts"] == 1
    rig.completion.complete.assert_not_awaited()
    assert not rig.selectable()


async def test_explicitly_unstarted_running_execution_can_resume(delivered_memory):
    rig = delivered_memory
    response = rig.stored_response()
    response["context"]["memory_completion"].update(
        state="running", attempts=0, execution_started=False,
    )
    rig.write_response(response, "running")
    final = await rig.step()
    await assert_saved_once(rig, final, attempts=1)
    rig.completion.complete.assert_awaited_once()


async def test_never_started_queued_cancellation_refunds_and_can_resume(delivered_memory):
    rig = delivered_memory
    occupied = asyncio.Event()
    release = asyncio.Event()

    async def hold_provider(_messages):
        occupied.set()
        await release.wait()
        return proposal(rig.body)

    rig.completion.complete.side_effect = hold_provider
    assert rig.scheduler.schedule(
        repository=rig.repo, character_id=rig.snapshot.character_id,
        user_scope=UserScope("web", "test", "other", "other", "private"),
        message=rig.body, rule_hints=[], source_message_id="occupying-source", immediate=True,
    )
    await asyncio.wait_for(occupied.wait(), 1)
    target = asyncio.create_task(rig.step())
    try:
        async with asyncio.timeout(1):
            while rig.scheduler.status.queued != 1:
                await asyncio.sleep(0)
        await rig.scheduler.shutdown(timeout=.01)
        response = await asyncio.wait_for(target, 1)
        marker = response["context"]["memory_completion"]
        receipt = marker["semantic_receipt"]
        assert response["memory_completion_state"] == "pending"
        assert marker["attempts"] == 0 and marker["turn_finished"]
        assert receipt["status"] == "cancelled" and receipt["stage"] == "queue"
        assert receipt["accepted"] == receipt["persisted"] == 0
        assert rig.completion.complete.await_count == 1 and rig.selectable()
        assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
        rig.completion.complete.side_effect = None
        rig.completion.complete.return_value = proposal(rig.body)
        await rig.rebuild()
        final = await rig.step()
        await assert_saved_once(rig, final, attempts=1)
        assert rig.completion.complete.await_count == 2
    finally:
        release.set()
        if not target.done():
            target.cancel()
        await asyncio.gather(target, return_exceptions=True)


async def test_active_provider_cancellation_keeps_attempt_and_blocks(delivered_memory):
    rig = delivered_memory
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold_provider(_messages):
        started.set()
        await release.wait()
        return proposal(rig.body)

    rig.completion.complete.side_effect = hold_provider
    target = asyncio.create_task(rig.step())
    try:
        await asyncio.wait_for(started.wait(), 1)
        await rig.scheduler.shutdown(timeout=.01)
        response = await asyncio.wait_for(target, 1)
        marker = response["context"]["memory_completion"]
        assert response["memory_completion_state"] == "blocked" and marker["attempts"] == 1
        assert marker["semantic_receipt"]["status"] == "cancelled"
        assert marker["semantic_receipt"]["stage"] == "proposal_generation"
        assert rig.completion.complete.await_count == 1 and not rig.selectable()
    finally:
        release.set()
        if not target.done():
            target.cancel()
        await asyncio.gather(target, return_exceptions=True)


async def test_claim_conflict_is_blocked_without_recalling_provider(delivered_memory, monkeypatch):
    rig = delivered_memory
    monkeypatch.setattr(rig.repo, "append_claim", AsyncMock(side_effect=MemoryClaimConflict("changed")))
    response = await rig.step()
    marker = response["context"]["memory_completion"]
    assert response["memory_completion_state"] == "blocked" and marker["attempts"] == 1
    assert marker["semantic_receipt"]["status"] == "conflict"
    assert marker["semantic_receipt"]["persisted"] == 0
    assert rig.completion.complete.await_count == 1 and not rig.selectable()


async def test_same_http_error_during_persistence_is_not_a_provider_retry(delivered_memory, monkeypatch):
    rig = delivered_memory
    monkeypatch.setattr(rig.repo, "append_claim", AsyncMock(side_effect=provider_error("503")))
    response = await rig.step()
    marker = response["context"]["memory_completion"]
    receipt = marker["semantic_receipt"]
    assert response["memory_completion_state"] == "blocked"
    assert marker["reason"] == "semantic_failure" and marker["attempts"] == 1
    assert receipt["status"] == "failed" and receipt["stage"] == "persistence"
    assert receipt.get("retryable") is not True
    assert rig.completion.complete.await_count == 1 and not rig.selectable()


async def test_existing_exhausted_budget_is_not_reserved_again(delivered_memory):
    rig = delivered_memory
    response = rig.stored_response()
    marker = response["context"]["memory_completion"]
    marker.update(attempts=3, execution_started=False, semantic_receipt=dict(
        status="failed", stage="proposal_generation", retryable=True, persisted=0,
    ))
    rig.write_response(response, "pending")
    final = await rig.step()
    marker = final["context"]["memory_completion"]
    assert final["memory_completion_state"] == "blocked"
    assert marker["reason"] == "retry_limit_reached" and marker["attempts"] == 3
    rig.completion.complete.assert_not_awaited()
    assert not rig.selectable()


@pytest.mark.parametrize("attempts", [True, -1, "2"])
async def test_invalid_persisted_budget_cannot_authorize_execution(delivered_memory, attempts):
    rig = delivered_memory
    response = rig.stored_response()
    response["context"]["memory_completion"]["attempts"] = attempts
    rig.write_response(response, "pending")
    final = await rig.step()
    assert final["memory_completion_state"] == "blocked"
    assert final["context"]["memory_completion"]["reason"] == "invalid_retry_budget"
    rig.completion.complete.assert_not_awaited()
    assert not rig.selectable()


async def test_runtime_shutdown_before_task_creation_refunds_reserved_execution(delivered_memory, monkeypatch):
    rig = delivered_memory
    operation = rig.worker.operation

    async def close_after_reservation(name, **params):
        accepted = await operation(name, **params)
        if name == "memory_update" and json.loads(params["response"])["context"]["memory_completion"].get("execution_started") is True:
            await rig.runtime.shutdown(timeout=0)
        return accepted

    monkeypatch.setattr(rig.worker, "operation", close_after_reservation)
    final = await rig.step()
    marker = final["context"]["memory_completion"]
    assert final["memory_completion_state"] == "pending" and marker["attempts"] == 0
    assert marker["execution_started"] is False and marker["reason"] == "completion_unavailable"
    rig.completion.complete.assert_not_awaited()
    assert rig.selectable()


async def test_legitimate_empty_proposals_complete_without_claims_or_retry(delivered_memory):
    rig = delivered_memory
    rig.completion.complete.return_value = '{"memories":[]}'
    final = await rig.step()
    marker = final["context"]["memory_completion"]
    assert final["memory_completion_state"] == "completed" and marker["attempts"] == 1
    assert marker["semantic_receipt"]["status"] == "no_change"
    assert marker["semantic_receipt"]["persisted"] == 0
    assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
    rig.completion.complete.assert_awaited_once()
    assert not rig.selectable()


@pytest.mark.parametrize("receipt", [
    dict(status="failed", stage="proposal_validation", error="ValueError", persisted=0),
    dict(status="failed", stage="proposal_generation", error="HTTPStatusError", persisted=0),
    dict(status="cancelled", stage="proposal_generation", accepted=0, persisted=0),
    dict(status="unknown", persisted=0),
    dict(status="capacity", persisted=0),
])
async def test_legacy_pending_without_safe_retry_proof_is_not_replayed(delivered_memory, receipt):
    rig = delivered_memory
    response = rig.stored_response()
    response["context"]["memory_completion"].update(turn_finished=True, semantic_receipt=receipt)
    rig.write_response(response, "pending")
    await rig.rebuild()
    final = await rig.step()
    marker = final["context"]["memory_completion"]
    assert final["memory_completion_state"] == "blocked" and marker["reason"] == "semantic_failure"
    assert marker["semantic_receipt"] == receipt
    rig.completion.complete.assert_not_awaited()
    assert not rig.selectable()
    assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)


class _HeartbeatClock:
    """Delegate all APIs; replace only this worker's heartbeat sleep."""

    def __init__(self, update_written, heartbeat_awake):
        self.update_written = update_written
        self.heartbeat_awake = heartbeat_awake
        self.park = asyncio.Event()
        self.first_heartbeat = True

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay):
        if delay != 10:
            return await asyncio.sleep(delay)
        if not self.first_heartbeat:
            await self.park.wait()
            return
        self.first_heartbeat = False
        await self.update_written.wait()
        self.heartbeat_awake.set()


@pytest.mark.parametrize("lease_case", ["own", "external", "database_failure", "primary_cancel"])
async def test_heartbeat_distinguishes_own_publish_from_external_cas(
    delivered_memory, monkeypatch, lease_case,
):
    rig = delivered_memory
    external_replacement = lease_case == "external"
    update_written = asyncio.Event()
    release_update = asyncio.Event()
    heartbeat_awake = asyncio.Event()
    renewal_finished = asyncio.Event()
    release_provider = asyncio.Event()
    provider_started = asyncio.Event()
    renewal_results = []
    target_responses = []
    external_responses = []
    operation = rig.worker.operation

    # Replace the module reference, never the shared asyncio.sleep function.
    monkeypatch.setattr(dm, "asyncio", _HeartbeatClock(update_written, heartbeat_awake))

    async def observe_operation(name, **params):
        if name == "memory_renew":
            if lease_case in {"database_failure", "primary_cancel"}:
                renewal_finished.set()
                raise OSError("synthetic lease database outage")
            # Synchronous real CAS fixes the old implementation's exact
            # A->B update / renew(A) interleaving without executor timing.
            accepted = rig.database.integration_receipt(name, **params)
            renewal_results.append((params["expected"], accepted))
            renewal_finished.set()
            return accepted
        if name == "memory_update" and not target_responses:
            proposed = json.loads(params["response"])
            marker = proposed["context"]["memory_completion"]
            if proposed["memory_completion_state"] == "running" and marker.get("turn_finished") is True:
                accepted = rig.database.integration_receipt(name, **params)
                assert accepted
                target_responses.append(params["response"])
                assert rig.database.integration_receipt("get", key=rig.key)["response"] == params["response"]
                if external_replacement:
                    external = json.loads(params["response"])
                    external["context"]["memory_completion"].update(
                        state="blocked", attempt_id="synthetic-external-attempt",
                        reason="synthetic_external_cas",
                    )
                    replacement = dm.encode(external, "blocked")
                    assert rig.database.integration_receipt(
                        "memory_update", key=rig.key, owner=rig.owner,
                        expected=params["response"], response=replacement,
                        expires_at=params["expires_at"],
                    )
                    external_responses.append(replacement)
                # The database has B (or external C), but persist has not yet
                # published B into its closure. A shared lock must keep the
                # heartbeat outside this complete update-and-publish interval.
                update_written.set()
                await release_update.wait()
                return accepted
        return await operation(name, **params)

    monkeypatch.setattr(rig.worker, "operation", observe_operation)

    async def finish_after_renewal(_messages):
        provider_started.set()
        # Do not let a fast provider receipt outrun the observed renewal.
        await renewal_finished.wait()
        if lease_case == "primary_cancel":
            await release_provider.wait()
        return '{"memories":[]}' if external_replacement else proposal(rig.body)

    rig.completion.complete.side_effect = finish_after_renewal
    target = asyncio.create_task(rig.step())
    try:
        await asyncio.wait_for(heartbeat_awake.wait(), 2)
        # Under the old code the synchronous renew(A) has failed; with the
        # fix the heartbeat is waiting on the shared lock. Never wait for its
        # renewal before releasing persist: that would deadlock the fixed code.
        await asyncio.sleep(0)
        release_update.set()
        if lease_case == "primary_cancel":
            await asyncio.wait_for(renewal_finished.wait(), 2)
            await asyncio.wait_for(provider_started.wait(), 2)
            target.cancel()
            with pytest.raises(asyncio.CancelledError):
                await target
            response = rig.stored_response()
            assert response["memory_completion_state"] == "running"
            assert response["context"]["memory_completion"]["turn_finished"] is True
        elif lease_case == "database_failure":
            with pytest.raises(OSError, match="synthetic lease database outage"):
                await asyncio.wait_for(target, 3)
            response = rig.stored_response()
            assert response["memory_completion_state"] == "running"
            assert response["context"]["memory_completion"]["turn_finished"] is True
            # The claim may already be committed; failure never invents rollback.
            assert len(await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)) == 1
            await rig.rebuild()
            final = await rig.step()
            assert final["memory_completion_state"] == "blocked"
            assert final["context"]["memory_completion"]["reason"] == "execution_outcome_unknown"
            assert len(await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)) == 1
        elif external_replacement:
            with pytest.raises(RuntimeError, match="Completion lease superseded"):
                await asyncio.wait_for(target, 3)
            assert renewal_results == [(target_responses[0], False)]
            row = rig.database.integration_receipt("get", key=rig.key)
            assert row["status"] == "delivered" and row["response"] == external_responses[0]
            assert not rig.selectable()
            assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
        else:
            response = await asyncio.wait_for(target, 3)
            assert renewal_results == [(target_responses[0], True)]
            await assert_saved_once(rig, response, attempts=1)
        assert rig.completion.complete.await_count == 1
    finally:
        # Release only test-controlled gates; production cancellation/receipt
        # ownership remains exercised by the fixture's normal cleanup.
        release_update.set()
        renewal_finished.set()
        release_provider.set()
        if not target.done():
            target.cancel()
        await asyncio.gather(target, return_exceptions=True)


@pytest.mark.parametrize(
    ("statuses", "final_state", "final_reason"),
    [
        pytest.param((429, 200), "completed", None, id="http-429-recovers-and-saves"),
        pytest.param((503, 503, 503), "blocked", "retry_limit_reached", id="http-503-stops-at-three"),
        pytest.param((401,), "blocked", "semantic_failure", id="http-401-stops-at-one"),
    ],
)
async def test_default_memory_http_transport_obeys_delivery_retry_policy(
    delivered_memory, statuses, final_state, final_reason,
):
    rig = delivered_memory
    requests = []

    def respond(request):
        assert request.method == "POST"
        assert str(request.url) == "https://api.deepseek.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer synthetic-memory-key"
        wire = json.loads(request.content)
        assert wire["model"] == "deepseek-flash"
        assert wire["temperature"] == 0.0 and wire["max_tokens"] == 768
        assert wire["stream"] is False and wire["thinking"] == {"type": "disabled"}
        assert json.loads(wire["messages"][-1]["content"])["current_user_message"] == rig.body
        assert len(requests) < len(statuses), "An exhausted or permanent failure made another HTTP request"
        status = statuses[len(requests)]
        requests.append(wire)
        if status == 200:
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": proposal(rig.body)},
            }]})
        return httpx.Response(status, json={"error": {"message": "synthetic provider rejection"}})

    completion = OpenAICompatibleMemoryCompletion(MemoryLlmConfig(
        enabled=True,
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        api_key="synthetic-memory-key",
        timeout_seconds=1,
    ))
    # Keep the actual .complete(), request encoding, status check and decoding.
    # Setting this instance's client bypasses only its lazy network-client creation.
    completion._client = httpx.AsyncClient(transport=httpx.MockTransport(respond), timeout=1)
    rig.completion = completion
    rig.scheduler._completion = completion
    try:
        for attempt, status in enumerate(statuses, 1):
            response = await rig.step()
            marker = response["context"]["memory_completion"]
            receipt = marker["semantic_receipt"]
            assert len(requests) == attempt and marker["attempts"] == attempt
            assert response["reply"] == rig.original_reply
            if status != 200:
                assert receipt["status"] == "failed"
                assert receipt["stage"] == "proposal_generation"
                assert receipt["http_status"] == status and receipt["persisted"] == 0
                assert receipt["retryable"] is (status in {429, 503})
            if attempt < len(statuses):
                assert response["memory_completion_state"] == "pending" and rig.selectable()
                assert marker["reason"] == "transient_model_failure"

        assert response["memory_completion_state"] == final_state
        assert len(requests) == len(statuses) and not rig.selectable()
        rig.scheduler._embedding_provider.embed_texts.assert_not_called()
        if final_state == "completed":
            await assert_saved_once(rig, response, attempts=len(statuses))
        else:
            assert marker["reason"] == final_reason
            assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
            relationship = await rig.repo.get_relationship_record(rig.snapshot.character_id, rig.snapshot.user_scope)
            assert relationship["interaction_count"] == 1
    finally:
        await completion.close()

@pytest.mark.parametrize("statuses", [(503, 200), (503, 503, 503), (401,)])
async def test_scope_review_uses_the_same_bounded_delivery_retry_contract(delivered_memory, statuses):
    rig = delivered_memory
    rig.body += "请让所有角色都记住我喜欢深蓝色油墨。"
    stored = rig.stored_response()
    stored["context"]["request"]["message"] = rig.body
    rig.write_response(stored, "pending")
    reviews, proposals = [], []

    async def complete(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["current_user_message"] == rig.body
        if "candidate_scopes" in payload:
            assert payload["candidate_scopes"] == ["user_global"]
            assert len(reviews) < len(statuses)
            status = statuses[len(reviews)]
            reviews.append(status)
            if status != 200:
                raise provider_error(str(status))
            return json.dumps({"decisions": [dict(scope_level="user_global", authorized=True, evidence=rig.body)]})
        proposals.append(payload)
        return proposal(rig.body)

    rig.completion.complete.side_effect = complete
    for attempt, status in enumerate(statuses, 1):
        response = await rig.step()
        marker = response["context"]["memory_completion"]
        assert marker["attempts"] == attempt and len(reviews) == attempt
        if status != 200:
            receipt = marker["semantic_receipt"]
            assert receipt["stage"] == "scope_review" and receipt["status"] == "failed"
            assert receipt["retryable"] is (status == 503) and receipt["persisted"] == 0
            assert not proposals
        if attempt < len(statuses):
            assert response["memory_completion_state"] == "pending" and rig.selectable()
            assert marker["reason"] == "transient_model_failure"
            await rig.rebuild()
    if statuses[-1] == 200:
        assert len(proposals) == 1
        await assert_saved_once(rig, response, attempts=len(statuses))
    else:
        assert response["memory_completion_state"] == "blocked" and not rig.selectable()
        assert marker["reason"] == ("retry_limit_reached" if statuses[-1] == 503 else "semantic_failure")
        assert not await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope)
        relationship = await rig.repo.get_relationship_record(rig.snapshot.character_id, rig.snapshot.user_scope)
        assert relationship["interaction_count"] == 1
    assert response["reply"] == rig.original_reply
