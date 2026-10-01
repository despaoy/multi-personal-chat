"""Actual signed late delivery after an authorized semantic memory erasure."""

import asyncio
import hashlib
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict

import asyncpg

from character.memory_extractor import extract_preferred_address
from character.memory_llm import get_memory_enrichment_scheduler, is_memory_erasure_request
from db.memory_source import source_identity, source_scope
from inference.lora_registry import get_lora_character_id
from infra.security_utils import integration_signature
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService
from services.delivery_memory import ensure_delivery_memory_worker, shutdown_delivery_memory
from services.turn_completion import get_turn_completion_runtime


async def run_delivery_erasure_probe(
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
    erase, target = fixture["cases"][-2:]
    sender = "stage34-delivery-memory-fixed-qq-owner"
    fields = ("tsukiyashiro_kisaki", "qq", "stage34-gateway", sender, "private", sender)
    target_id = args.run_label + "-old-generated"
    erase_id = args.run_label + "-erase"
    proof["integration_target_fields"] = list(fields)
    assert len(target["message"]) > 150 and len(erase["message"]) > 200
    assert extract_preferred_address(target["message"]) == "柚汀" and is_memory_erasure_request(erase["message"])
    assert not is_memory_erasure_request(target["message"])
    assert database.memory_source_admission(*fields, source_message_id=target_id, body=target["message"]) == "new"
    active = [r for r in database.loras if r["status"] == "active"]
    assert len(active) == 1 and get_lora_character_id(active[0]["name"]) == fields[0]
    proof["actual_inherited_role"] = dict(active_name=active[0]["name"], character_id=fields[0], not_invented=True)
    original = get_turn_completion_runtime()
    assert original.capacity == 128 and original.active == original.reserved == 0
    scheduler = get_memory_enrichment_scheduler()
    proof["capacity"] = dict(capacity=original.capacity, changed=False)
    token = secrets.token_urlsafe(48)
    os.environ.update(
        ASTRBOT_INTEGRATION_TOKEN=token,
        ASTRBOT_INTEGRATION_TOKENS="",
        INTEGRATION_SIGNATURE_REQUIRED="true",
        ASTRBOT_ENABLED="true",
        ASTRBOT_QQ_ENABLED="true",
    )
    proof["signed_auth_statuses"] = []
    proof["completion_outcomes"] = []
    proof["relationship_write_attempts"] = []
    completions = []
    old_complete = CharacterContextService.complete_turn
    old_upsert = DatabaseCharacterMemoryRepository.upsert_relationship

    async def observed_complete(service, prepared, turn, reply, **kwargs):
        if prepared.user_scope.sender_id == sender:
            task = asyncio.current_task()
            completions.append(task)
            record = dict(
                source_message_id=kwargs.get("source_message_id"),
                received_at=prepared.received_at.isoformat(),
                task_name=task.get_name(),
            )
            try:
                result = await old_complete(service, prepared, turn, reply, **kwargs)
                record["outcome"] = asdict(result)
                return result
            finally:
                proof["completion_outcomes"].append(record)
        return await old_complete(service, prepared, turn, reply, **kwargs)

    async def observed_upsert(repo, character_id, user_scope, state, **kwargs):
        if user_scope.sender_id == sender:
            record = dict(
                preferred_address=state.preferred_address, character_id=character_id, scope=asdict(user_scope)
            )
            proof["relationship_write_attempts"].append(record)
            result = await old_upsert(repo, character_id, user_scope, state, **kwargs)
            record["committed"] = True
            return result
        return await old_upsert(repo, character_id, user_scope, state, **kwargs)

    CharacterContextService.complete_turn = observed_complete
    DatabaseCharacterMemoryRepository.upsert_relationship = observed_upsert

    async def connection(name=database_name):
        c = await asyncpg.connect(user="boot", database=name, host=str(root / "socket"), port=25433)
        assert await c.fetchval("SHOW data_directory") == str(root / "data")
        return c

    def receipt(response):
        row = database.integration_receipt("get", key=response["receiptId"])
        stored = json.loads(row["response"])
        return dict(
            status=row["status"],
            owner_matches_actual_token=row["owner"] == response["deliveryToken"],
            context=stored["context"],
            original_reply_preserved=stored["reply"]["replyText"] == response["replyText"],
            completion_state=stored.get("memory_completion_state"),
        )

    async def snapshot():
        c = await connection()
        try:
            scope = source_scope(*fields)
            key = source_identity(scope, target_id)["source_key"]
            raw = await c.fetchrow("SELECT state,body,observed_at FROM memory_sources WHERE source_key=$1", key)
            relationship = database.get_character_relationship(*fields)
            return dict(
                source=dict(raw) if raw else None,
                source_terms=await c.fetchval("SELECT count(*) FROM memory_source_terms WHERE source_key=$1", key),
                owner_fence=await c.fetchval(
                    "SELECT revoked_before FROM memory_source_fences WHERE owner_key=$1", scope["owner_key"]
                ),
                relationship=relationship,
                visible_sources=database.list_memory_sources(*fields, limit=100),
                claims=database.list_character_memory_claims(*fields, limit=None, include_inactive=True),
            )
        finally:
            await c.close()

    async def signed(path, payload):
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))
        nonce = uuid.uuid4().hex
        response = await client.post(
            path,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Integration-Token": token,
                "X-Integration-Timestamp": timestamp,
                "X-Integration-Nonce": nonce,
                "X-Integration-Signature": integration_signature(token, timestamp, nonce, body),
            },
        )
        proof["signed_auth_statuses"].append(response.status_code)
        return response

    async def generate(case, source_id):
        payload = dict(
            platform="qq",
            adapter=fields[2],
            messageId=source_id,
            conversationId=sender,
            conversationType="private",
            senderId=sender,
            senderName="隔离链路用户",
            text=case["message"],
            requestBudgetSeconds=180,
        )
        proof.setdefault("integration_payloads", []).append(payload)
        if not proof["generation"]:
            bad = await client.post("/api/integrations/astrbot/messages", json=payload)
            assert bad.status_code == 401
            proof["unauthorized_status"] = bad.status_code
        start = len(cloud_calls)
        response = await signed("/api/integrations/astrbot/messages", payload)
        private = response.json()
        proof["generation"].append(
            dict(
                id=case["id"],
                source_message_id=source_id,
                message=case["message"],
                http_status=response.status_code,
                response={k: v for k, v in private.items() if k not in {"deliveryToken", "traceId"}},
                cloud_call_range=[start, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
        assert (
            response.status_code == 200
            and private["shouldReply"]
            and not private["retryable"]
            and private["receiptId"]
            and private["deliveryToken"]
        )
        return private

    async def acknowledge(private, kind):
        start = len(cloud_calls)
        response = await signed(
            "/api/integrations/astrbot/delivery",
            dict(receiptId=private["receiptId"], deliveryToken=private["deliveryToken"], status="delivered"),
        )
        proof.setdefault("acknowledgements", []).append(
            dict(
                kind=kind,
                http_status=response.status_code,
                response=response.json(),
                cloud_call_range=[start, len(cloud_calls)],
            )
        )
        assert response.status_code == 200
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))

    proof["owner_before"] = await snapshot()
    prior = "stage34-delivery-memory-fixed-target"
    assert any(
        r["source_message_id"] == prior and "MB-764-C" in r["body"] for r in proof["owner_before"]["visible_sources"]
    )
    assert len(proof["owner_before"]["claims"]) == 2 and not proof["owner_before"]["relationship"].get(
        "preferred_address"
    )
    target_reply = await generate(target, target_id)
    proof["target_generated_receipt"] = receipt(target_reply)
    proof["owner_before_erasure"] = await snapshot()
    assert (
        proof["target_generated_receipt"]["status"] == "generated" and proof["owner_before_erasure"]["source"] is None
    )
    frozen = proof["target_generated_receipt"]["context"]["completion_snapshot"]
    assert frozen["character_id"] == fields[0] and frozen["user_scope"]["sender_id"] == sender
    erase_reply = await generate(erase, erase_id)
    await acknowledge(erase_reply, "actual-erasure")
    worker = ensure_delivery_memory_worker(database)
    await worker.wait_idle(timeout=120)
    proof["erasure_receipt"] = receipt(erase_reply)
    proof["owner_after_erasure"] = await snapshot()
    erasure = proof["erasure_receipt"]["context"]["memory_completion"]["semantic_receipt"]
    proof["actual_erasure_verified"] = (
        erasure["status"] == "erased"
        and erasure["persisted"] == 2
        and not proof["owner_after_erasure"]["claims"]
        and not proof["owner_after_erasure"]["visible_sources"]
    )
    (output / "before-late-ack.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    assert proof["actual_erasure_verified"], (
        "Erasure/fence prerequisite not established; do not infer delayed-write defect"
    )
    assert (
        proof["owner_after_erasure"]["owner_fence"]
        and frozen["received_at"] <= proof["owner_after_erasure"]["owner_fence"]
    )
    proof["calls_before_late_ack"] = len(cloud_calls)
    await acknowledge(target_reply, "old-generated-after-erasure")
    await worker.wait_idle(timeout=90)
    proof["target_terminal_receipt"] = receipt(target_reply)
    proof["owner_after_late_ack"] = await snapshot()
    await acknowledge(target_reply, "duplicate-old-generated")
    proof["target_duplicate_receipt"] = receipt(target_reply)
    proof["owner_after_duplicate"] = await snapshot()
    proof["calls_after_late_ack"] = len(cloud_calls)
    await shutdown_delivery_memory()
    proof["delivery_worker_terminal"] = worker.task.done() and not worker.task.cancelled() and worker.active_key is None
    await original.shutdown(timeout=35)
    assert await scheduler.flush_memory(timeout=90)
    proof["completion_tasks_terminal"] = all(t.done() and not t.cancelled() for t in completions)
    proof["completion_runtime_terminal"] = dict(
        active=original.active, reserved=original.reserved, failed=original.failed, cancelled=original.cancelled
    )
    proof["memory_status"] = asdict(scheduler.status)
    proof["sync_pending_final"] = len(database._pending)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
    c = await connection()
    try:
        proof["lock_waiters_remaining"] = await c.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE datname=$1 AND wait_event_type='Lock'", database_name
        )
        rows = await c.fetch(
            'SELECT * FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id',
            "web",
            "web-character",
            identity,
            fields[0],
        )
        proof["scope_sql"]["owner_original_messages_sha256_after"] = hashlib.sha256(
            json.dumps([dict(r) for r in rows], sort_keys=True, default=str).encode()
        ).hexdigest()
    finally:
        await c.close()
    c = await connection(source_database)
    try:
        snapshot_after = {}
        for table in ["memory_sources", "memory_source_terms", "character_memories", "messages"]:
            rows = await c.fetch("SELECT * FROM " + table)
            snapshot_after[table] = hashlib.sha256(
                json.dumps(
                    sorted([dict(r) for r in rows], key=lambda r: json.dumps(r, sort_keys=True, default=str)),
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
        proof["scope_sql"]["source_database_snapshot_after"] = snapshot_after
    finally:
        await c.close()
    capture_storage_proof(proof, output)
    proof["guard_checks"] = audit_delivery_erasure_guards(proof, fixture, cloud_calls)
    proof["checks"] = {**proof["guard_checks"], **audit_delivery_erasure(proof, fixture, cloud_calls)}
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


def audit_delivery_erasure_guards(proof, fixture, cloud_calls):
    erase, target = fixture["cases"][-2:]
    old = proof["target_generated_receipt"]
    done = proof["target_terminal_receipt"]
    return dict(
        normal_owner_auth=proof["auth_statuses"] == [200, 200],
        actual_signed_integration_auth=proof["unauthorized_status"] == 401
        and proof["signed_auth_statuses"] == [200] * 5,
        complete_new_input=len(target["message"]) > 150 and len(erase["message"]) > 200,
        actual_inherited_role=proof["actual_inherited_role"]["character_id"] == "tsukiyashiro_kisaki",
        actual_current_pro=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        both_full_statements_in_model=all(
            any(case["message"] in json.dumps(c["request"], ensure_ascii=False) for c in cloud_calls)
            for case in [target, erase]
        ),
        real_existing_owner_evidence=len(proof["owner_before"]["claims"]) == 2
        and any("MB-764-C" in r["body"] for r in proof["owner_before"]["visible_sources"]),
        late_source_not_stored_before_delivery=proof["owner_before_erasure"]["source"] is None,
        actual_erasure_terminal=proof["actual_erasure_verified"]
        and proof["erasure_receipt"]["context"]["memory_completion"]["state"] == "completed",
        real_fence_newer_than_original_clock=bool(proof["owner_after_erasure"]["owner_fence"])
        and old["context"]["completion_snapshot"]["received_at"] <= proof["owner_after_erasure"]["owner_fence"],
        physical_delivery_truth=done["status"] == proof["target_duplicate_receipt"]["status"] == "delivered"
        and done["owner_matches_actual_token"],
        original_model_reply_preserved=old["original_reply_preserved"] and done["original_reply_preserved"],
        no_late_model_or_answer_regeneration=proof["calls_before_late_ack"] == proof["calls_after_late_ack"],
        original_snapshot_preserved=old["context"]["completion_snapshot"] == done["context"]["completion_snapshot"],
        unchanged_capacity=proof["capacity"] == dict(capacity=128, changed=False),
        all_actual_handles_terminal=proof["delivery_worker_terminal"]
        and proof["completion_tasks_terminal"]
        and proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["lock_waiters_remaining"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == 0,
        original_web_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        source_parent_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        old_calls_not_replayed=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
    )


def audit_delivery_erasure(proof, fixture, cloud_calls):
    state = proof["owner_after_late_ack"]
    before = proof["owner_after_erasure"]
    receipt = proof["target_terminal_receipt"]["context"]["memory_completion"]
    return dict(
        old_raw_speech_not_restored=state["source"] is None
        and state["source_terms"] == 0
        and not state["visible_sources"],
        erased_owner_facts_not_restored=state["claims"] == [],
        exact_completion_reports_blocked=receipt["state"] == "blocked"
        and receipt["semantic_receipt"]["source_capture"] in {"stale", "revoked"},
        old_preferred_address_not_restored=state["relationship"].get("preferred_address")
        == before["relationship"].get("preferred_address")
        and state["relationship"].get("preferred_address") != "柚汀",
        no_stale_address_write=not any(
            r["preferred_address"] == "柚汀" and r.get("committed") for r in proof["relationship_write_attempts"]
        ),
        duplicate_does_not_change_owner_state=proof["owner_after_duplicate"] == state
        and proof["acknowledgements"][-1]["response"]["changed"] is False,
    )
