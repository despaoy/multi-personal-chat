"""One real signed full erasure request: generation state versus actual completion."""

import hashlib
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict
from html import unescape

import asyncpg

from character.memory_llm import get_memory_enrichment_scheduler, is_memory_erasure_request
from inference.generation_request import DEFERRED_MEMORY_OPERATION_POLICY
from inference.lora_registry import get_lora_character_id
from infra.security_utils import integration_signature
from services.delivery_memory import ensure_delivery_memory_worker, shutdown_delivery_memory
from services.turn_completion import get_turn_completion_runtime


async def run_pending_operation_probe(
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
    case = fixture["cases"][-1]
    sender = "stage34-delivery-memory-fixed-qq-owner"
    fields = ("tsukiyashiro_kisaki", "qq", "stage34-gateway", sender, "private", sender)
    assert len(case["message"]) > 200 and is_memory_erasure_request(case["message"])
    proof["actual_fields"] = list(fields)
    active = [r for r in database.loras if r["status"] == "active"]
    assert len(active) == 1 and get_lora_character_id(active[0]["name"]) == fields[0]
    proof["actual_inherited_role"] = dict(active_name=active[0]["name"], character_id=fields[0])
    original = get_turn_completion_runtime()
    proof["capacity"] = dict(capacity=original.capacity, active=original.active, reserved=original.reserved)
    scheduler = get_memory_enrichment_scheduler()
    token = secrets.token_urlsafe(48)
    os.environ.update(
        ASTRBOT_INTEGRATION_TOKEN=token,
        ASTRBOT_INTEGRATION_TOKENS="",
        INTEGRATION_SIGNATURE_REQUIRED="true",
        ASTRBOT_ENABLED="true",
        ASTRBOT_QQ_ENABLED="true",
    )
    proof["signed_auth_statuses"] = []

    async def connection(name=database_name):
        c = await asyncpg.connect(user="boot", database=name, host=str(root / "socket"), port=25433)
        assert await c.fetchval("SHOW data_directory") == str(root / "data")
        return c

    async def snapshot():
        c = await connection()
        try:
            rows = await c.fetch(
                'SELECT * FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id',
                fields[1],
                fields[2],
                sender,
                fields[0],
            )
            return dict(
                claims=database.list_character_memory_claims(*fields, limit=None, include_inactive=True),
                sources=database.list_memory_sources(*fields, limit=100),
                relationship=database.get_character_relationship(*fields),
                original_archive_sha256=hashlib.sha256(
                    json.dumps(
                        [dict(r) for r in rows if r["sourceMessageId"] == "stage34-delivery-memory-fixed-target"],
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest(),
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

    def receipt(private):
        row = database.integration_receipt("get", key=private["receiptId"])
        stored = json.loads(row["response"])
        return dict(
            status=row["status"],
            owner_matches_actual_token=row["owner"] == private["deliveryToken"],
            context=stored["context"],
            original_reply_preserved=stored["reply"]["replyText"] == private["replyText"],
            completion_state=stored.get("memory_completion_state"),
        )

    proof["owner_before"] = await snapshot()
    assert len(proof["owner_before"]["claims"]) == 2
    assert any("MB-764-C" in r["body"] for r in proof["owner_before"]["sources"])
    payload = dict(
        platform="qq",
        adapter=fields[2],
        messageId=args.run_label + "-erase",
        conversationId=sender,
        conversationType="private",
        senderId=sender,
        senderName="隔离链路用户",
        text=case["message"],
        requestBudgetSeconds=180,
    )
    bad = await client.post("/api/integrations/astrbot/messages", json=payload)
    proof["unauthorized_status"] = bad.status_code
    assert bad.status_code == 401
    start = len(cloud_calls)
    response = await signed("/api/integrations/astrbot/messages", payload)
    private = response.json()
    proof["generation"].append(
        dict(
            id=case["id"],
            message=case["message"],
            http_status=response.status_code,
            response={k: v for k, v in private.items() if k not in {"deliveryToken", "traceId"}},
            cloud_call_range=[start, len(cloud_calls)],
        )
    )
    (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    assert (
        response.status_code == 200 and private["shouldReply"] and not private["retryable"] and private["deliveryToken"]
    )
    proof["generated_receipt"] = receipt(private)
    proof["owner_before_ack"] = await snapshot()
    start = len(cloud_calls)
    acknowledged = await signed(
        "/api/integrations/astrbot/delivery",
        dict(receiptId=private["receiptId"], deliveryToken=private["deliveryToken"], status="delivered"),
    )
    proof["acknowledgements"] = [
        dict(
            kind="actual-erasure",
            http_status=acknowledged.status_code,
            response=acknowledged.json(),
            cloud_call_range=[start, len(cloud_calls)],
        )
    ]
    assert acknowledged.status_code == 200
    worker = ensure_delivery_memory_worker(database)
    await worker.wait_idle(timeout=120)
    proof["terminal_receipt"] = receipt(private)
    proof["owner_after_ack"] = await snapshot()
    start = len(cloud_calls)
    duplicate = await signed(
        "/api/integrations/astrbot/delivery",
        dict(receiptId=private["receiptId"], deliveryToken=private["deliveryToken"], status="delivered"),
    )
    proof["acknowledgements"].append(
        dict(
            kind="duplicate",
            http_status=duplicate.status_code,
            response=duplicate.json(),
            cloud_call_range=[start, len(cloud_calls)],
        )
    )
    proof["owner_after_duplicate"] = await snapshot()
    proof["duplicate_receipt"] = receipt(private)
    proof["calls_after_duplicate"] = len(cloud_calls)
    await shutdown_delivery_memory()
    proof["delivery_worker_terminal"] = worker.task.done() and not worker.task.cancelled() and worker.active_key is None
    await original.shutdown(timeout=35)
    assert await scheduler.flush_memory(timeout=90)
    proof["completion_runtime_terminal"] = dict(
        active=original.active, reserved=original.reserved, failed=original.failed, cancelled=original.cancelled
    )
    proof["memory_status"] = asdict(scheduler.status)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
    proof["sync_pending_final"] = len(database._pending)
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
    proof["checks"] = audit_pending_operation(proof, fixture, cloud_calls)
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


def audit_pending_operation(proof, fixture, cloud_calls):
    case = fixture["cases"][-1]
    generation = proof["generation"][0]
    selected = cloud_calls[slice(*generation["cloud_call_range"])]
    answers = [c for c in selected if c["request"].get("max_tokens") == 1024]
    reply = generation["response"]["replyText"]
    before = proof["owner_before"]
    generated = proof["generated_receipt"]
    done = proof["terminal_receipt"]
    actual = done["context"]["memory_completion"]["semantic_receipt"]
    return dict(
        normal_owner_auth=proof["auth_statuses"] == [200, 200],
        actual_signed_integration_auth=proof["unauthorized_status"] == 401
        and proof["signed_auth_statuses"] == [200, 200, 200],
        complete_new_input=len(case["message"]) > 200 and is_memory_erasure_request(case["message"]),
        actual_inherited_role=proof["actual_inherited_role"]["character_id"] == "tsukiyashiro_kisaki",
        current_pro_all_success=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        real_owner_evidence=len(before["claims"]) == 2 and any("MB-764-C" in r["body"] for r in before["sources"]),
        real_final_model_called=len(answers) == 1,
        full_current_message_on_final_wire=len(answers) == 1
        and case["message"] in unescape(answers[0]["request"]["messages"][-1]["content"]),
        deferred_server_state_on_wire=len(answers) == 1
        and DEFERRED_MEMORY_OPERATION_POLICY in answers[0]["request"]["messages"][0]["content"],
        no_fabricated_execution_receipt=generated["context"]["completion_snapshot"]["memory_operation_receipt"] is None
        and "memory_operation_deferred" not in generated["context"]["completion_snapshot"],
        not_erased_before_ack=proof["owner_before_ack"]["claims"] == before["claims"]
        and proof["owner_before_ack"]["sources"] == before["sources"],
        generated_not_delivered=generated["status"] == "generated",
        reply_reports_delivery_then_processing="交付" in reply
        and "确认后" in reply
        and any(s in reply for s in ["处理", "核对", "尝试删除"]),
        reply_does_not_deny_capability=not any(
            s in reply
            for s in [
                "无法执行删除",
                "无法在普通对话",
                "不支持删除",
                "没有可用的记忆管理",
                "没有记忆管理",
                "不能执行删除",
            ]
        ),
        reply_does_not_claim_completed=not any(
            s in reply for s in ["已删除", "已经删除", "已经删掉", "已彻底删除", "已经忘掉"]
        ),
        remaining_question_answered="预约" in reply
        and any(s in reply for s in ["不会", "不等于", "不影响", "仍然有效"]),
        exact_semantic_erasure=actual["status"] == "erased"
        and actual["persisted"] == 2
        and done["context"]["memory_completion"]["state"] == "completed",
        actual_owner_memory_absent=not proof["owner_after_ack"]["claims"] and not proof["owner_after_ack"]["sources"],
        physical_delivery_truth=done["status"] == proof["duplicate_receipt"]["status"] == "delivered"
        and done["owner_matches_actual_token"],
        original_reply_and_clock_preserved=generated["original_reply_preserved"]
        and done["original_reply_preserved"]
        and generated["context"]["completion_snapshot"] == done["context"]["completion_snapshot"],
        duplicate_idempotent=proof["owner_after_duplicate"] == proof["owner_after_ack"]
        and proof["acknowledgements"][-1]["response"]["changed"] is False,
        duplicate_no_models=proof["acknowledgements"][-1]["cloud_call_range"][0]
        == proof["acknowledgements"][-1]["cloud_call_range"][1],
        archive_not_deleted=before["original_archive_sha256"] == proof["owner_after_ack"]["original_archive_sha256"],
        unchanged_capacity=proof["capacity"] == dict(capacity=128, active=0, reserved=0),
        all_handles_terminal=proof["delivery_worker_terminal"]
        and proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["lock_waiters_remaining"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == 0,
        parent_and_web_archive_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"]
        and proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        no_old_calls_replayed=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
    )
