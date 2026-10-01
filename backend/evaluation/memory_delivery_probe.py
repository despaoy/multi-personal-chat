"""Real owned completion occupancy and a new admitted-turn persistence boundary."""

import asyncio
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import asdict

import asyncpg

from character.memory_extractor import extract_preferred_address
from character.memory_llm import get_memory_enrichment_scheduler
from db.memory_source import source_identity, source_scope
from inference.lora_registry import get_lora_character_id
from infra.security_utils import integration_signature
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services import turn_completion
from services.character_context import CharacterContextService


async def run_delivery_memory_probe(
    client,
    args,
    proof,
    fixture,
    cloud_calls,
    *,
    database_name,
    source_database,
    root,
    output,
    identity,
    database,
    runtime,
    capture_storage_proof,
):
    fields = ("tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity)
    holder, target = fixture["cases"][-2:]
    holder_id = args.run_label + "-holder"
    target_id = args.run_label + "-target"
    assert len(holder["message"]) > 150 and len(target["message"]) > 150
    assert extract_preferred_address(holder["message"]) == "鹭枫"
    assert database.memory_source_admission(*fields, source_message_id=holder_id, body=holder["message"]) == "new"
    target_sender = args.run_label + "-qq-owner"
    target_fields = ("tsukiyashiro_kisaki", "qq", "stage34-gateway", target_sender, "private", target_sender)
    proof["integration_target_fields"] = list(target_fields)
    assert (
        database.memory_source_admission(*target_fields, source_message_id=target_id, body=target["message"]) == "new"
    )
    old_runtime = turn_completion.get_turn_completion_runtime()
    assert old_runtime.active == 0 and old_runtime.capacity == 128
    owned = turn_completion.TurnCompletionRuntime(capacity=1)
    turn_completion._runtime = owned
    proof["controlled_capacity"] = dict(
        default_capacity=old_runtime.capacity,
        isolated_capacity=owned.capacity,
        actual_runtime_instance=True,
        production_unchanged=True,
    )
    old_timeout = database._operation_timeout
    assert old_timeout == 30
    database._operation_timeout = 180
    proof["isolated_operation_timeout"] = dict(
        before=old_timeout,
        during=180,
        production_unchanged=True,
        reason="observe actual held relationship SQL while independent target inference completes",
    )
    scheduler = get_memory_enrichment_scheduler()
    original_upsert = DatabaseCharacterMemoryRepository.upsert_relationship
    original_complete = CharacterContextService.complete_turn
    original_sync = database.upsert_character_relationship
    started, finished = threading.Event(), threading.Event()
    outcomes = []
    handles = []
    attempts = []
    locked = False
    locker = None
    transaction = None
    proof["native_payloads"] = []

    async def connect(name=database_name):
        c = await asyncpg.connect(user="boot", database=name, host=str(root / "socket"), port=25433)
        assert await c.fetchval("SHOW data_directory") == str(root / "data")
        return c

    async def source(source_id):
        c = await connect()
        key = source_identity(source_scope(*(fields if source_id == holder_id else target_fields)), source_id)[
            "source_key"
        ]
        try:
            row = await c.fetchrow(
                "SELECT state,body,body_digest,observed_at FROM memory_sources WHERE source_key=$1", key
            )
            if row is None:
                return None
            result = dict(row)
            result["terms"] = await c.fetchval("SELECT count(*) FROM memory_source_terms WHERE source_key=$1", key)
            return result
        finally:
            await c.close()

    def observed_sync(*pargs, **kwargs):
        if len(pargs) < 8 or pargs[7] != "鹭枫":
            return original_sync(*pargs, **kwargs)
        row = dict(preferred_address=pargs[7])
        attempts.append(row)
        started.set()
        try:
            result = original_sync(*pargs, **kwargs)
            row["completed"] = True
            return result
        except BaseException as e:
            row["exception_type"] = type(e).__name__
            raise
        finally:
            finished.set()

    async def held_upsert(repo, character_id, user_scope, state):
        nonlocal locked, locker, transaction
        if state.preferred_address == "鹭枫" and not locked:
            proof["holder_source_before_lock"] = await source(holder_id)
            assert (
                proof["holder_source_before_lock"]["state"] == "recorded"
                and proof["holder_source_before_lock"]["body"] == holder["message"]
            )
            locker = await connect()
            transaction = locker.transaction()
            await transaction.start()
            row = await locker.fetchrow(
                "SELECT character_id FROM character_relationships WHERE character_id=$1 AND platform=$2 AND adapter=$3 AND sender_id=$4 AND conversation_type=$5 AND conversation_id=$6 FOR UPDATE",
                *fields,
            )
            assert row is not None
            locked = True
            proof["controlled_lock"] = dict(
                kind="actual PG relationship row lock after full speech capture",
                full_scope=list(fields),
                locker_pid=locker.get_server_pid(),
                production_unchanged=True,
            )
        return await original_upsert(repo, character_id, user_scope, state)

    async def observed_complete(service, prepared, turn, reply, **kwargs):
        if kwargs.get("source_message_id") == holder_id:
            handles.append(asyncio.current_task())
        row = dict(
            source_message_id=kwargs.get("source_message_id"),
            message=turn.message,
            received_at=prepared.received_at.isoformat(),
        )
        try:
            result = await original_complete(service, prepared, turn, reply, **kwargs)
            row["outcome"] = asdict(result)
            return result
        except asyncio.CancelledError:
            row["cancelled"] = True
            raise
        finally:
            outcomes.append(row)

    database.upsert_character_relationship = observed_sync
    DatabaseCharacterMemoryRepository.upsert_relationship = held_upsert
    CharacterContextService.complete_turn = observed_complete

    old_prepare = CharacterContextService.prepare_turn
    proof["delivery_completion_feedback"] = []
    proof["target_preparation_times"] = []

    async def observed_prepare(service, turn, character_id):
        prepared = await old_prepare(service, turn, character_id)
        if turn.message == target["message"]:
            proof["target_preparation_times"].append(
                dict(
                    received_at=prepared.received_at.isoformat(),
                    input_received_at=turn.received_at.isoformat() if turn.received_at else None,
                    scope=asdict(prepared.user_scope),
                )
            )
        return prepared

    CharacterContextService.prepare_turn = observed_prepare
    private_token = secrets.token_urlsafe(48)
    os.environ.update(
        ASTRBOT_INTEGRATION_TOKEN=private_token,
        ASTRBOT_INTEGRATION_TOKENS="",
        INTEGRATION_SIGNATURE_REQUIRED="true",
        ASTRBOT_ENABLED="true",
        ASTRBOT_QQ_ENABLED="true",
    )
    loras = database.loras
    configured = next((r for r in loras if r["name"] == "kisaki"), None)
    if configured is None:
        database.add_lora(dict(id=args.run_label + "-persona", name="kisaki", status="inactive"))
        configured = next(r for r in database.loras if r["name"] == "kisaki")
    database.update_lora_status(configured["id"], "active")
    active = [r for r in database.loras if r["status"] == "active"]
    assert len(active) == 1 and get_lora_character_id(active[0]["name"]) == "tsukiyashiro_kisaki"
    proof["isolated_integration_role"] = dict(
        active_name=active[0]["name"],
        actual_character_id=get_lora_character_id(active[0]["name"]),
        production_unchanged=True,
    )
    proof["signed_request_auth_statuses"] = []
    response_private = None

    async def signed_post(path, payload):
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        stamp = str(int(time.time()))
        nonce = uuid.uuid4().hex
        return await client.post(
            path,
            content=encoded,
            headers={
                "Content-Type": "application/json",
                "X-Integration-Token": private_token,
                "X-Integration-Timestamp": stamp,
                "X-Integration-Nonce": nonce,
                "X-Integration-Signature": integration_signature(private_token, stamp, nonce, encoded),
            },
        )

    async def receipt():
        record = database.integration_receipt("get", key=response_private["receiptId"])
        stored = json.loads(record["response"])
        return dict(
            status=record["status"],
            owner_matches_actual_delivery_token=record["owner"] == response_private["deliveryToken"],
            context=stored.get("context", {}),
            stored_reply_equals_actual=stored["reply"]["replyText"] == response_private["replyText"],
        )

    async def send(case, source_id, suffix):
        nonlocal response_private
        before = len(cloud_calls)
        if source_id == holder_id:
            payload = dict(
                message=case["message"],
                characterId=fields[0],
                sessionId=args.run_label + "-" + suffix,
                sessionType="private",
                sourceMessageId=source_id,
                history=[],
            )
            proof["native_payloads"].append(payload)
            response = await client.post("/api/generate", json=payload)
            body = response.json()
        else:
            payload = dict(
                platform="qq",
                adapter="stage34-gateway",
                messageId=source_id,
                conversationId=target_sender,
                conversationType="private",
                senderId=target_sender,
                senderName="隔离链路用户",
                text=case["message"],
                requestBudgetSeconds=180,
            )
            proof["integration_payload"] = payload
            bad = await client.post("/api/integrations/astrbot/messages", json=payload)
            assert bad.status_code == 401
            proof["signed_request_auth_statuses"].append(bad.status_code)
            response = await signed_post("/api/integrations/astrbot/messages", payload)
            response_private = response.json()
            assert (
                response.status_code == 200
                and response_private["shouldReply"]
                and not response_private["retryable"]
                and response_private["receiptId"]
                and response_private["deliveryToken"]
            )
            proof["signed_request_auth_statuses"].append(response.status_code)
            body = {k: v for k, v in response_private.items() if k not in {"deliveryToken", "traceId"}}
            proof["receipt_before_delivery"] = await receipt()
            assert (
                proof["receipt_before_delivery"]["context"]["character_id"] == fields[0]
                and proof["receipt_before_delivery"]["context"]["message_saved"]
            )
            proof["target_before_delivery"] = await source(source_id)
        proof["generation"].append(
            dict(
                id=case["id"],
                source_message_id=source_id,
                message=case["message"],
                http_status=response.status_code,
                response=body,
                cloud_call_range=[before, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))

    async def acknowledge(suffix):
        before = len(cloud_calls)
        payload = dict(
            receiptId=response_private["receiptId"], deliveryToken=response_private["deliveryToken"], status="delivered"
        )
        response = await signed_post("/api/integrations/astrbot/delivery", payload)
        proof.setdefault("delivery_acknowledgements", []).append(
            dict(
                kind=suffix,
                http_status=response.status_code,
                response=response.json(),
                cloud_call_range=[before, len(cloud_calls)],
            )
        )
        assert response.status_code == 200
        proof["signed_request_auth_statuses"].append(response.status_code)
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))

    holder_request = asyncio.create_task(send(holder, holder_id, "holder"), name="delivery-original-holder-request")
    try:
        assert await asyncio.to_thread(started.wait, 120), "Actual held relationship write not started"
        await holder_request
        assert await scheduler.flush_memory(timeout=90)
        proof["actual_holder_task_live"] = not handles[0].done()
        proof["actual_holder_thread_live"] = not finished.is_set()
        proof["actual_owned_active_before_target"] = owned.active
        c = await connect()
        try:
            rows = await c.fetch(
                "SELECT pid,wait_event_type,wait_event,query FROM pg_stat_activity WHERE datname=$1 AND wait_event_type='Lock'",
                database_name,
            )
            proof["actual_lock_waiters"] = [
                dict(
                    pid=r["pid"],
                    wait_event_type=r["wait_event_type"],
                    wait_event=r["wait_event"],
                    relationship_write="character_relationships" in r["query"] and "INSERT INTO" in r["query"],
                )
                for r in rows
            ]
        finally:
            await c.close()
        assert proof["actual_holder_task_live"] and proof["actual_holder_thread_live"] and owned.active == 1
        await send(target, target_id, "target")
        await acknowledge("while-full")
        proof["receipt_at_busy_return"] = await receipt()
        proof["target_at_busy_return"] = await source(target_id)
        proof["owned_active_at_busy_return"] = owned.active
        proof["holder_thread_live_at_busy_return"] = not finished.is_set()
        proof["semantic_status_at_busy_return"] = asdict(scheduler.status)
        proof["completion_outcomes_at_busy_return"] = list(outcomes)
        proof["prepared_at_busy_return"] = list(proof["prepared_diagnostics"])
        (output / "busy-observation.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    finally:
        if transaction is not None:
            await transaction.rollback()
            proof["relationship_lock_released"] = True
        if locker is not None:
            await locker.close()
        if not holder_request.done():
            await holder_request
    assert await asyncio.to_thread(finished.wait, 40)
    if not handles[0].done():
        await asyncio.wait_for(asyncio.shield(handles[0]), 45)
    await asyncio.sleep(0)
    proof["holder_completion_terminal"] = dict(done=handles[0].done(), cancelled=handles[0].cancelled())
    from services.delivery_memory import ensure_delivery_memory_worker, shutdown_delivery_memory

    worker = ensure_delivery_memory_worker(database)
    await worker.wait_idle(timeout=120)
    proof["automatic_completion_without_duplicate_ack"] = await receipt()
    proof["target_after_holder_release"] = await source(target_id)
    await acknowledge("after-capacity-release")
    proof["receipt_after_duplicate_ack"] = await receipt()
    assert await scheduler.flush_memory(timeout=90)
    await shutdown_delivery_memory()
    proof["delivery_worker_terminal"] = worker.task.done() and not worker.task.cancelled() and worker.active_key is None
    await owned.shutdown(timeout=35)
    proof["target_final_source"] = await source(target_id)
    proof["target_final_visible_sources"] = database.list_memory_sources(
        *target_fields, source_message_ids=(target_id,), limit=1
    )
    proof["target_final_claims"] = [
        r
        for r in database.list_character_memory_claims(*target_fields, limit=None, include_inactive=True)
        if r["source_message_id"] == target_id
    ]
    proof["holder_relationship_write_attempts"] = attempts
    proof["completion_outcomes"] = outcomes
    proof["memory_status"] = asdict(scheduler.status)
    proof["completion_runtime_terminal"] = dict(
        active=owned.active,
        completed=owned.completed,
        failed=owned.failed,
        cancelled=owned.cancelled,
        closed=owned.closed,
        reserved=getattr(owned, "reserved", 0),
    )
    proof["sync_pending_final"] = len(database._pending)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
    proof["scope_owner_sources_after"] = database.list_memory_sources(*fields, limit=100)
    proof["scope_owner_claims_after"] = database.list_character_memory_claims(
        *fields, limit=None, include_inactive=True
    )
    c = await connect()
    try:
        rows = await c.fetch(
            'SELECT * FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id',
            "web",
            "web-character",
            identity,
            fields[0],
        )
        old = [dict(r) for r in rows if r["id"] <= proof["scope_sql"]["owner_message_max_id_before"]]
        proof["scope_sql"]["owner_original_messages_sha256_after"] = hashlib.sha256(
            json.dumps(old, sort_keys=True, default=str).encode()
        ).hexdigest()
        proof["scope_sql"]["new_messages"] = [
            dict(r) for r in rows if r["id"] > proof["scope_sql"]["owner_message_max_id_before"]
        ]
        proof["lock_waiters_remaining"] = await c.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE datname=$1 AND wait_event_type='Lock'", database_name
        )
    finally:
        await c.close()
    c = await connect(source_database)
    try:
        snapshots = {}
        for table in ["memory_sources", "memory_source_terms", "character_memories", "messages"]:
            rows = await c.fetch("SELECT * FROM " + table)
            snapshots[table] = hashlib.sha256(
                json.dumps(
                    sorted([dict(r) for r in rows], key=lambda r: json.dumps(r, sort_keys=True, default=str)),
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
        proof["scope_sql"]["source_database_snapshot_after"] = snapshots
    finally:
        await c.close()
    database._operation_timeout = old_timeout
    proof["isolated_operation_timeout"]["restored"] = database._operation_timeout == 30
    capture_storage_proof(proof, output)
    proof["guard_checks"] = audit_delivery_memory_guards(proof, fixture, cloud_calls)
    proof["checks"] = {**proof["guard_checks"], **audit_delivery_memory(proof, fixture, cloud_calls)}
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            dict(native_requests=len(proof["generation"]), cloud_calls=len(cloud_calls), checks=proof["checks"]),
            ensure_ascii=False,
        )
    )
    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]


def audit_delivery_memory_guards(proof, fixture, cloud_calls):
    holder, target = fixture["cases"][-2:]
    return dict(
        normal_owner_auth=proof["auth_statuses"] == [200, 200],
        signed_integration_auth=proof["signed_request_auth_statuses"] == [401, 200, 200, 200],
        full_new_input=len(holder["message"]) > 150 and len(target["message"]) > 150,
        actual_role_present=proof["isolated_integration_role"]["actual_character_id"] == "tsukiyashiro_kisaki"
        and proof["receipt_before_delivery"]["context"]["character_id"] == "tsukiyashiro_kisaki",
        actual_current_pro=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        full_target_in_actual_model=any(
            target["message"] in json.dumps(c["request"], ensure_ascii=False) for c in cloud_calls
        ),
        actual_full_capacity=proof["actual_owned_active_before_target"] == proof["owned_active_at_busy_return"] == 1
        and proof["controlled_capacity"]["isolated_capacity"] == 1,
        real_live_task_and_thread=proof["actual_holder_task_live"]
        and proof["actual_holder_thread_live"]
        and proof["holder_thread_live_at_busy_return"]
        and any(r["relationship_write"] for r in proof["actual_lock_waiters"]),
        speech_not_captured_before_delivery=proof["target_before_delivery"] is None
        or proof["target_before_delivery"]["body"] is None,
        actual_physical_receipt_delivered=proof["receipt_at_busy_return"]["status"]
        == proof["receipt_after_duplicate_ack"]["status"]
        == "delivered"
        and proof["receipt_at_busy_return"]["owner_matches_actual_delivery_token"],
        original_reply_preserved=proof["receipt_after_duplicate_ack"]["stored_reply_equals_actual"],
        duplicate_did_not_regenerate_answer=proof["delivery_acknowledgements"][1]["cloud_call_range"][0]
        == proof["delivery_acknowledgements"][1]["cloud_call_range"][1]
        or all(
            c["response"].get("choices", [{}])[0].get("message", {}).get("content")
            != proof["generation"][1]["response"]["replyText"]
            for c in cloud_calls[proof["delivery_acknowledgements"][1]["cloud_call_range"][0] :]
        ),
        all_actual_work_terminal=proof["jobs_terminal"]
        and proof["sync_pending_final"] == 0
        and proof["completion_runtime_terminal"]["active"] == proof["completion_runtime_terminal"]["reserved"] == 0
        and proof["holder_completion_terminal"]["done"]
        and not proof["holder_completion_terminal"]["cancelled"],
        real_lock_released=proof["relationship_lock_released"] and proof["lock_waiters_remaining"] == 0,
        original_sources_unchanged=all(
            r in proof["scope_owner_sources_after"] for r in proof["scope_owner_sources_before"]
        ),
        original_claims_unchanged=all(
            r in proof["scope_owner_claims_after"] for r in proof["scope_owner_claims_before"]
        ),
        original_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        parent_fixture_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        isolated_timeout_restored=proof["isolated_operation_timeout"]["restored"],
        old_calls_not_replayed=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
    )


def audit_delivery_memory(proof, fixture, cloud_calls):
    target = fixture["cases"][-1]
    source = proof["target_final_source"]
    ack = proof["delivery_acknowledgements"][0]["response"]
    return dict(
        pending_completion_persisted_while_full=bool(
            proof["receipt_at_busy_return"]["context"].get("memory_completion")
        ),
        first_ack_exposes_pending_memory=ack.get("memoryStatus") == "pending",
        original_target_source_complete=source is not None
        and source["state"] == "recorded"
        and source["body"] == target["message"]
        and source["terms"] > 0,
        source_readable=any(r["body"] == target["message"] for r in proof["target_final_visible_sources"]),
        actual_writer_executed=any(
            r.get("source_message_id") == proof["generation"][1]["source_message_id"]
            and r.get("source_capture") == "recorded"
            for r in proof["memory_status"]["recent_results"]
        ),
        explicit_preference_grounded=any(
            "深蓝色油墨" in r["content"]
            and json.loads(r.get("metadata_json") or "{}").get("attributed_to") == "user"
            and json.loads(r.get("metadata_json") or "{}").get("content_semantics") != "quoted_source"
            for r in proof["target_final_claims"]
        ),
        first_server_generation_receipt_preserved=source is not None
        and source["observed_at"] == proof["target_preparation_times"][0]["received_at"],
        completion_receipt_terminal=proof["receipt_after_duplicate_ack"]["context"]
        .get("memory_completion", {})
        .get("state")
        == "completed",
        **(
            dict(
                automatic_completion_before_duplicate=proof["automatic_completion_without_duplicate_ack"]["context"][
                    "memory_completion"
                ]["state"]
                == "completed",
                exact_semantic_receipt_terminal=proof["receipt_after_duplicate_ack"]["context"]["memory_completion"]
                .get("semantic_receipt", {})
                .get("status")
                == "saved"
                and proof["receipt_after_duplicate_ack"]["context"]["memory_completion"]["semantic_receipt"].get(
                    "persisted", 0
                )
                > 0,
                generation_snapshot_retained=proof["receipt_before_delivery"]["context"]["completion_snapshot"][
                    "received_at"
                ]
                == proof["target_preparation_times"][0]["received_at"]
                and len(proof["target_preparation_times"]) == 1,
                delivery_worker_terminal=proof["delivery_worker_terminal"],
            )
            if "automatic_completion_without_duplicate_ack" in proof
            else {}
        ),
    )
