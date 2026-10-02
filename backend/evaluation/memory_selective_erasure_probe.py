"""Signed selective erasure and a new retained-preference comprehension case."""

import hashlib
import json
import os
import re
import secrets
import time
import uuid
from dataclasses import asdict
from html import unescape

import asyncpg

from character.memory_llm import get_memory_enrichment_scheduler, is_memory_erasure_request
from evaluation.memory_operation_pending_probe import _claims_completed_erasure
from inference.generation_request import DEFERRED_MEMORY_OPERATION_POLICY
from inference.lora_registry import get_lora_character_id
from infra.security_utils import integration_signature
from services.delivery_memory import ensure_delivery_memory_worker, shutdown_delivery_memory
from services.turn_completion import get_turn_completion_runtime


async def run_selective_erasure_probe(
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
    sender = fixture.get("native_owner", "stage34-delivery-memory-fixed-qq-owner")
    fields = ("tsukiyashiro_kisaki", "qq", "stage34-gateway", sender, "private", sender)
    assert len(case["message"]) > 200
    proof["actual_intent_gate"] = is_memory_erasure_request(case["message"])
    assert proof["actual_intent_gate"] is (not fixture.get("source_capture_only", False))
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
                source_links=[dict(r) for r in await c.fetch(
                    "SELECT s.source_message_id,l.memory_id FROM memory_sources s JOIN memory_source_links l ON l.source_key=s.source_key "
                    "WHERE s.owner_key=$1 ORDER BY s.source_message_id,l.memory_id",
                    __import__('db.memory_source',fromlist=['source_scope']).source_scope(*fields)['owner_key'])],
                claims=database.list_character_memory_claims(*fields, limit=None, include_inactive=True),
                sources=database.list_memory_sources(*fields, limit=100),
                relationship=database.get_character_relationship(*fields),
                original_archive_sha256=hashlib.sha256(
                    json.dumps(
                        [dict(r) for r in rows if r["sourceMessageId"] in fixture.get("all_prior_source_ids", [fixture.get("initial_native_source_message_id", fixture.get("native_source_message_id", "stage34-delivery-memory-fixed-target"))])],
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
    assert any(fixture.get("native_receipt", "MB-764-C") in r["body"] for r in proof["owner_before"]["sources"])
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
    proof["primary_source_message_id"] = payload["messageId"]
    proof["primary_erasure_source_message_id"] = payload["messageId"]
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
    # A new comprehension case, only after actual selective deletion succeeded.
    before = proof["owner_before"]
    after = proof["owner_after_ack"]
    course = fixture.get("erased_memory_key", fixture["course_memory_key"])
    semantic = proof["terminal_receipt"]["context"]["memory_completion"]["semantic_receipt"]
    deletion_succeeded = (
        not any(r["memory_key"] == course for r in after["claims"])
        and _retained_claim_preserved(before, after, fixture)
        and semantic["status"] == "erased" and semantic["persisted"] == 1
    )
    if fixture.get("source_capture_only"):
        deletion_succeeded = (
            any(r['source_message_id'] == proof['primary_source_message_id'] and r['body'] == fixture['new_native_source'] for r in after['sources'])
            and semantic.get('status') == 'source_only'
            and after['claims'] == before['claims']
        )
    proof["retained_read_executed"] = deletion_succeeded
    if deletion_succeeded:
        read_message = fixture["retained_read_message"]
        read_payload = dict(payload, messageId=args.run_label + "-read", text=read_message)
        start = len(cloud_calls)
        read_response = await signed("/api/integrations/astrbot/messages", read_payload)
        read_private = read_response.json()
        proof["generation"].append(
            dict(
                id=fixture.get("retained_read_case_id", "read_retained_preference"),
                message=read_message,
                http_status=read_response.status_code,
                response={k: v for k, v in read_private.items() if k not in {"deliveryToken", "traceId"}},
                cloud_call_range=[start, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
        assert read_response.status_code == 200 and read_private["shouldReply"] and read_private["deliveryToken"]
        start = len(cloud_calls)
        read_ack = await signed(
            "/api/integrations/astrbot/delivery",
            dict(receiptId=read_private["receiptId"], deliveryToken=read_private["deliveryToken"], status="delivered"),
        )
        proof["read_ack"] = dict(
            http_status=read_ack.status_code, response=read_ack.json(), cloud_call_range=[start, len(cloud_calls)]
        )
        assert read_ack.status_code == 200
        await worker.wait_idle(timeout=120)
        proof["read_terminal_receipt"] = receipt(read_private)
        proof["owner_after_read"] = await snapshot()
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
    if fixture.get("source_capture_only"):
        proof["original_qq_archive_preserved"] = proof["owner_before"]["original_archive_sha256"] == proof["owner_after_ack"]["original_archive_sha256"]
    proof["scope_owner_sources_after"] = database.list_memory_sources(fields[0], "web", "web-character", identity, "private", identity, limit=100)
    capture_storage_proof(proof, output)
    proof["checks"] = audit_selective_erasure(proof, fixture, cloud_calls)
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


def audit_selective_erasure(proof, fixture, cloud_calls):
    if fixture.get("source_capture_only"):
        from evaluation.memory_quoted_source_audit import audit_quoted_source

        return audit_quoted_source(proof, fixture, cloud_calls)
    generation = proof["generation"][0]
    selected = cloud_calls[slice(*generation["cloud_call_range"])]
    answers = [c for c in selected if c["request"].get("max_tokens") == 1024]
    before = proof["owner_before"]
    after = proof["owner_after_ack"]
    done = proof["terminal_receipt"]
    receipt = done["context"]["memory_completion"]["semantic_receipt"]
    course = fixture.get("erased_memory_key", fixture["course_memory_key"])
    retained = fixture["retained_memory_key"]
    guards = dict(
        normal_owner_auth=proof["auth_statuses"] == [200, 200],
        actual_signed_auth=proof["unauthorized_status"] == 401
        and proof["signed_auth_statuses"] == [200] * (5 if proof.get("retained_read_executed") else 3),
        complete_input=len(fixture["cases"][-1]["message"]) > 200,
        actual_inherited_role=proof["actual_inherited_role"]["character_id"] == "tsukiyashiro_kisaki",
        real_current_pro=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        actual_final_model=len(answers) == 1
        and fixture["cases"][-1]["message"] in unescape(answers[0]["request"]["messages"][-1]["content"]),
        real_existing_two_claims={r["memory_key"] for r in before["claims"]} == {course, retained},
        not_erased_before_ack=proof["owner_before_ack"]["claims"] == before["claims"]
        and proof["owner_before_ack"]["sources"] == before["sources"],
        physical_delivery_and_duplicate=done["status"] == proof["duplicate_receipt"]["status"] == "delivered"
        and done["owner_matches_actual_token"]
        and proof["acknowledgements"][-1]["response"]["changed"] is False,
        original_reply_and_clock=done["original_reply_preserved"]
        and proof["generated_receipt"]["context"]["completion_snapshot"] == done["context"]["completion_snapshot"],
        no_duplicate_models=proof["acknowledgements"][-1]["cloud_call_range"][0]
        == proof["acknowledgements"][-1]["cloud_call_range"][1],
        original_archive=before["original_archive_sha256"] == after["original_archive_sha256"],
        parent_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        web_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        no_old_calls=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
        all_handles_terminal=proof["delivery_worker_terminal"]
        and proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["lock_waiters_remaining"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == 0,
    )
    new_retained = [r for r in after["claims"] if r["memory_key"] == retained]
    business = dict(
        course_claim_deleted=not any(r["memory_key"] == course for r in after["claims"]),
        retained_claim_unchanged=_retained_claim_preserved(before, after, fixture),
        shared_raw_source_unreadable=not any(
            r["source_message_id"] == fixture.get("native_source_message_id", "stage34-delivery-memory-fixed-target") for r in after["sources"]
        ),
        exact_semantic_erasure_one=receipt["status"] == "erased"
        and receipt["persisted"] == 1
        and done["context"]["memory_completion"]["state"] == "completed",
    )
    answer = answers[0] if len(answers) == 1 else {}
    systems = [m["content"] for m in answer.get("request", {}).get("messages", []) if m.get("role") == "system"]
    writers = [c for c in cloud_calls if c["request"].get("max_tokens") == 768]
    constraints = []
    for call in writers:
        for m in call["request"]["messages"]:
            if m.get("role") == "user":
                try:
                    payload = json.loads(m["content"])
                except (TypeError, ValueError):
                    continue
                if payload.get("current_user_message") == fixture["cases"][-1]["message"]:
                    constraints.append(payload.get("partial_erasure_authorization", {}))
    guards.update(
        intent_accepted=proof["actual_intent_gate"] is True,
        deferred_policy_in_actual_request=any(DEFERRED_MEMORY_OPERATION_POLICY in s for s in systems),
        no_premature_completion=not _claims_completed_erasure(generation["response"]["replyText"]),
        no_execution_before_ack=proof["generated_receipt"]["context"].get("memory_completion") is None,
        actual_partial_writer_constraint=len(constraints) == 1
        and constraints[0].get("allowed_erase_memory_ids") == fixture.get("allowed_erase_memory_ids", ["4"])
        and constraints[0].get("protected_memory_ids") == fixture.get("protected_memory_ids", ["5"])
        and constraints[0].get("unresolved_protection") is False,
        erasure_instruction_not_recaptured=not any(r["source_message_id"] == proof.get("primary_erasure_source_message_id", proof["reused_native_fixture"].get("run_label", "")+"-erase") for r in after["sources"]),
        bounded_source_erasure=receipt.get("source_erasure_policy") == "claim_targets_only_for_partial_retention",
    )
    read_checks = dict(retained_read_executed=proof.get("retained_read_executed") is True)
    if proof.get("retained_read_executed"):
        read = proof["generation"][1]
        actual = [c for c in cloud_calls[slice(*read["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        wire = unescape(actual[0]["request"]["messages"][-1]["content"]) if len(actual) == 1 else ""
        block = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
        packets = [json.loads(line[2:]) for line in block[1].splitlines() if line.startswith("- {")] if block else []
        current = proof["owner_after_read"]
        if fixture.get("retained_read_kind") == "course_after_ink_erasure":
            packet_text=json.dumps(packets,ensure_ascii=False)
            stored_retained=[r for r in current['claims'] if r['memory_key']==retained]
            read_checks.update(
                actual_retained_read_model=len(actual)==1 and fixture['retained_read_message'] in wire,
                no_current_preference_hint='深蓝色' not in fixture['retained_read_message'],
                retained_course_in_memory_packet=fixture.get("native_receipt", "MB-764-C") in packet_text and '海庭鹤林' in packet_text,
                deleted_preference_absent_from_memory_packet='深蓝色' not in packet_text,
                deleted_preference_absent_from_retained_claim='深蓝色' not in json.dumps(new_retained,ensure_ascii=False),
                retained_course_facts_understood=all(x in read['response'].get('replyText','') for x in [fixture.get("native_receipt", "MB-764-C"),'海庭鹤林']) and any(day in read['response'].get('replyText','') for day in ('周六','星期六','礼拜六')),
                retained_read_delivery_completed=proof['read_terminal_receipt']['context']['memory_completion']['state']=='completed',
                retained_claim_still_unchanged=new_retained==stored_retained,
                preference_not_recreated=not any(r['memory_key']==course for r in current['claims']),
                original_source_still_unreadable=not any(r['source_message_id']==fixture.get("native_source_message_id", "stage34-delivery-memory-fixed-target") for r in current['sources']))
        else:
            read_checks.update(
                actual_retained_read_model=len(actual) == 1 and fixture["retained_read_message"] in wire,
                no_current_preference_hint="深蓝色" not in fixture["retained_read_message"],
                retained_record_in_memory_packet=any(
                    p.get("memory_key") == retained or "深蓝色油墨进行纸版压印" in p.get("content", "") for p in packets
                ),
                deleted_course_absent_from_memory_packet=all(
                    fixture.get("native_receipt", "MB-764-C") not in json.dumps(p, ensure_ascii=False) for p in packets
                ),
                retained_preference_understood="深蓝色" in read["response"].get("replyText", ""),
                retained_read_delivery_completed=proof["read_terminal_receipt"]["context"]["memory_completion"]["state"]
                == "completed",
                retained_claim_still_unchanged=new_retained
                == [r for r in current["claims"] if r["memory_key"] == retained],
                course_not_recreated=not any(r["memory_key"] == course for r in current["claims"]),
                original_source_still_unreadable=not any(
                    r["source_message_id"] == fixture.get("native_source_message_id", "stage34-delivery-memory-fixed-target") for r in current["sources"]
                ),
            )
    proof["read_checks"] = read_checks
    proof["authority_guards"] = guards
    proof["business_checks"] = business
    checks = {**guards, **business, **read_checks}
    if fixture.get("cross_source_erasure"):
        from evaluation.memory_cross_source_erasure_audit import augment_cross_source_erasure

        return augment_cross_source_erasure(proof, fixture, cloud_calls, checks)
    return checks


def _retained_claim_preserved(before, after, fixture):
    key = fixture["retained_memory_key"]
    old = [r for r in before["claims"] if r["memory_key"] == key]
    new = [r for r in after["claims"] if r["memory_key"] == key]
    if len(old) != 1 or len(new) != 1:
        return False
    if not fixture.get("erasure_evidence_projection_expected"):
        return old == new
    previous, current = old[0], new[0]
    if any(previous.get(k) != current.get(k) for k in previous
           if k not in {"content", "evidence_json", "metadata_json", "metadata"}):
        return False
    metadata = json.loads(current["metadata_json"])
    projection = metadata.pop("erasure_evidence_projection", {})
    if metadata != json.loads(previous["metadata_json"]):
        return False
    sources = [s for s in before["sources"]
               if s["source_message_id"] == fixture["qq_owner_prerequisite"]["source_message_id"]]
    if len(sources) != 1:
        return False
    source = sources[0]
    body = source["body"]
    deleted = fixture["erased_original_statement"]
    if body.count(deleted) != 1 or json.loads(previous["evidence_json"]) != [body]:
        return False
    start = body.index(deleted)
    spans = [[0, start], [start + len(deleted), len(body)]]
    expected_projection = dict(version=1, kind="original_source_fragments",
        sources=[dict(source_message_id=source["source_message_id"], spans=spans)],
        complete_original_source=False)
    return (projection == expected_projection
            and json.loads(current["evidence_json"]) == body.split(deleted)
            and current["content"] == "用户原话保留片段（仅保留内容见证据）")
