"""Observe real owner-lock capture after API timeout, then authoritative terminal state."""

import asyncio
import hashlib
import json
import os
import threading
from dataclasses import asdict

import asyncpg

from character.memory_extractor import extract_memories
from character.memory_llm import get_memory_enrichment_scheduler
from db.memory_source import source_identity, source_scope
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService


async def run_late_capture_probe(
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
    case, cold = fixture["cases"][-2:]
    source_id = args.run_label + "-target"
    authority = source_identity(source_scope(*fields), source_id)
    assert database.memory_source_admission(*fields, source_message_id=source_id, body=case["message"]) == "new"
    proof["eligible_rule_hints"] = [asdict(item) for item in extract_memories(case["message"])]
    assert len(case["message"]) > 150 and proof["eligible_rule_hints"]
    proof["native_payloads"] = []
    scheduler = get_memory_enrichment_scheduler()
    started = threading.Event()
    finished = threading.Event()
    sync_attempts = []
    outcomes = []
    complete_handles = []
    original_sync = database.capture_memory_source
    original_capture = DatabaseCharacterMemoryRepository.capture_source
    original_complete = CharacterContextService.complete_turn
    original_history = CharacterContextService._load_history
    locked = False
    locker = None
    transaction = None

    async def connect(name=database_name):
        c = await asyncpg.connect(user="boot", database=name, host=str(root / "socket"), port=25433)
        assert await c.fetchval("SHOW data_directory") == str(root / "data")
        return c

    async def metadata():
        c = await connect()
        try:
            row = await c.fetchrow(
                "SELECT state,body,observed_at,body_digest FROM memory_sources WHERE source_key=$1",
                authority["source_key"],
            )
            result = dict(row) if row else None
            if result is not None:
                result["terms"] = await c.fetchval(
                    "SELECT count(*) FROM memory_source_terms WHERE source_key=$1", authority["source_key"]
                )
                result["links"] = await c.fetchval(
                    "SELECT count(*) FROM memory_source_links WHERE source_key=$1", authority["source_key"]
                )
            return result
        finally:
            await c.close()

    def observed_sync(*pargs, **kwargs):
        if kwargs.get("source_message_id") != source_id:
            return original_sync(*pargs, **kwargs)
        first = not started.is_set()
        row = dict(source_message_id=source_id, body=kwargs["body"], observed_at=kwargs["observed_at"].isoformat())
        sync_attempts.append(row)
        if first:
            started.set()
        try:
            row["result"] = original_sync(*pargs, **kwargs)
            return row["result"]
        except BaseException as exc:
            row["exception_type"] = type(exc).__name__
            raise
        finally:
            if first:
                finished.set()

    async def held_capture(repo, *pargs, **kwargs):
        nonlocal locked, locker, transaction
        if kwargs.get("source_message_id") == source_id and not locked:
            # Admission and genuine generation have already happened. Do not
            # block source binding or fabricate a to_thread sleep/result.
            proof["source_before_lock"] = await metadata()
            assert proof["source_before_lock"]["state"] == "pending" and proof["source_before_lock"]["body"] is None
            locker = await connect()
            transaction = locker.transaction()
            await transaction.start()
            row = await locker.fetchrow(
                "SELECT owner_key,revoked_before FROM memory_source_fences WHERE owner_key=$1 FOR UPDATE",
                authority["owner_key"],
            )
            assert row is not None
            locked = True
            proof["controlled_lock"] = dict(
                kind="actual PG owner-fence row lock",
                after_valid_admission_and_real_generation=True,
                owner_key=row["owner_key"],
                locker_pid=locker.get_server_pid(),
                revoked_before=row["revoked_before"],
                production_unchanged=True,
            )
        return await original_capture(repo, *pargs, **kwargs)

    async def observed_complete(service, prepared, turn, reply, **kwargs):
        is_target = kwargs.get("source_message_id") == source_id
        if is_target:
            complete_handles.append(asyncio.current_task())
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

    async def cold_history(service, turn, scope, character_id):
        history = await original_history(service, turn, scope, character_id)
        if turn.message == cold["message"]:
            proof["cold_query_history"] = dict(
                original_history=history, returned_history=[], only_case=cold["id"], stored_archive_not_changed=True
            )
            return []
        return history

    database.capture_memory_source = observed_sync
    DatabaseCharacterMemoryRepository.capture_source = held_capture
    CharacterContextService.complete_turn = observed_complete
    if args.late_cold_query:
        CharacterContextService._load_history = cold_history

    async def send(selected, suffix):
        payload = dict(
            message=selected["message"],
            characterId=fields[0],
            sessionId=args.run_label,
            sessionType="private",
            sourceMessageId=args.run_label + "-" + suffix,
            history=[],
        )
        proof["native_payloads"].append(payload)
        before = len(cloud_calls)
        response = await client.post("/api/generate", json=payload)
        proof["generation"].append(
            dict(
                id=selected["id"],
                source_message_id=payload["sourceMessageId"],
                message=selected["message"],
                http_status=response.status_code,
                response=response.json(),
                cloud_call_range=[before, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))

    request_task = asyncio.create_task(send(case, "target"), name="stage32-real-native-request")
    try:
        # This only bounds observation. Release in finally and await the same
        # actual request/task; never launch a replacement on observation expiry.
        observed = await asyncio.to_thread(started.wait, 120)
        assert observed, "Actual capture not yet started"
        await request_task
        proof["source_at_api_return_while_locked"] = await metadata()
        proof["capture_thread_done_at_api_return"] = finished.is_set()
        proof["completion_task_at_api_return"] = dict(
            done=complete_handles[0].done(), cancelled=complete_handles[0].cancelled()
        )
        proof["semantic_status_at_api_return"] = asdict(scheduler.status)
        proof["sync_pending_at_api_return"] = len(database._pending)
        c = await connect()
        try:
            rows = await c.fetch(
                "SELECT pid,state,wait_event_type,wait_event,query FROM pg_stat_activity WHERE datname=$1 AND wait_event_type='Lock'",
                database_name,
            )
            proof["actual_lock_waiters"] = [
                dict(
                    pid=r["pid"],
                    state=r["state"],
                    wait_event_type=r["wait_event_type"],
                    wait_event=r["wait_event"],
                    owner_fence_lock_query="memory_source_fences" in r["query"] and "FOR UPDATE" in r["query"],
                )
                for r in rows
            ]
        finally:
            await c.close()
        (output / "lock-observation.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    finally:
        if transaction is not None:
            await transaction.rollback()
            proof["owner_lock_released"] = True
        if locker is not None:
            await locker.close()
        if not request_task.done():
            await request_task
    assert await asyncio.to_thread(finished.wait, 40), "Actual capture thread still not terminal"
    target_handle = complete_handles[0]
    if not target_handle.done():
        await asyncio.wait_for(asyncio.shield(target_handle), 45)
    proof["completion_task_terminal"] = dict(done=target_handle.done(), cancelled=target_handle.cancelled())
    assert await scheduler.flush_memory(timeout=90)
    proof["source_after_late_thread_terminal"] = await metadata()
    proof["sync_attempts_after_release"] = list(sync_attempts)
    proof["completion_outcomes_after_release"] = list(outcomes)
    proof["semantic_status_after_release"] = asdict(scheduler.status)
    proof["claims_after_release"] = [
        r
        for r in database.list_character_memory_claims(*fields, limit=None, include_inactive=True)
        if r["source_message_id"] == source_id
    ]
    proof["visible_source_after_release"] = database.list_memory_sources(
        *fields, source_message_ids=(source_id,), limit=1
    )
    if args.late_cold_query:
        assert (
            database.memory_source_admission(
                *fields, source_message_id=args.run_label + "-cold-question", body=cold["message"]
            )
            == "new"
        )
        await send(cold, "cold-question")
        assert await scheduler.flush_memory(timeout=90)
    proof["capture_thread_terminal"] = finished.is_set()
    proof["sync_pending_final"] = len(database._pending)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
    proof["completion_outcomes"] = outcomes
    proof["memory_status"] = asdict(scheduler.status)
    proof["scope_owner_sources_after"] = database.list_memory_sources(*fields, limit=100)
    proof["scope_owner_claims_after"] = database.list_character_memory_claims(
        *fields, limit=None, include_inactive=True
    )
    proof["queue_stats"] = runtime.stats()
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
    capture_storage_proof(proof, output)
    proof["configured_model"] = os.environ["OPENAI_COMPAT_MODEL"]
    proof["configured_context_window"] = int(os.environ["OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS"])
    proof["guard_checks"] = audit_late_capture_guards(proof, case, cloud_calls)
    proof["checks"] = {**proof["guard_checks"], **audit_late_capture(proof, case, cold, cloud_calls)}
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


def audit_late_capture(proof, case, cold, cloud_calls):
    first = proof["generation"][0]
    warning = next((w for w in first["response"].get("warnings") or [] if "尚未确认" in w), "")
    source = proof["source_after_late_thread_terminal"]
    outcomes = [
        r for r in proof["completion_outcomes_after_release"] if r["source_message_id"] == first["source_message_id"]
    ]
    checks = dict(
        timeout_truthfully_unknown=bool(warning)
        and warning in first["response"]["reply"]
        and "保存失败" not in warning,
        archive_retains_same_unknown_notice=any(
            r["sourceMessageId"] == first["source_message_id"]
            and r["reply"] == first["response"]["reply"]
            and r["message"] == case["message"]
            for r in proof["scope_sql"]["new_messages"]
        ),
        late_source_committed_complete=source["state"] == "recorded"
        and source["body"] == case["message"]
        and source["body_digest"] is None
        and source["terms"] > 0,
        first_receipt_preserved=source["observed_at"] == proof["source_before_lock"]["observed_at"],
        late_source_readable=any(r["body"] == case["message"] for r in proof["visible_source_after_release"]),
        completion_not_cancelled=proof["completion_task_terminal"]["done"]
        and not proof["completion_task_terminal"]["cancelled"]
        and bool(outcomes)
        and not any(r.get("cancelled") for r in outcomes),
        late_enrichment_not_lost=bool(outcomes)
        and outcomes[0].get("outcome", {}).get("memory_enrichment_scheduled") is True,
        actual_late_writer_executed=any(
            r.get("source_message_id") == first["source_message_id"] and r.get("status") == "saved"
            for r in proof["semantic_status_after_release"]["recent_results"]
        ),
        explicit_owner_preference_grounded=any(
            "青绿色油墨" in r["content"]
            and json.loads(r.get("metadata_json") or "{}").get("attributed_to") == "user"
            and json.loads(r.get("metadata_json") or "{}").get("content_semantics") != "quoted_source"
            for r in proof["claims_after_release"]
        ),
    )
    if "cold_query_history" in proof:
        query = proof["generation"][1]
        prepared = [r for r in proof["prepared_diagnostics"] if r["query"] == cold["message"]]
        reply = query["response"]["reply"]
        checks.update(
            cold_history_empty=bool(prepared)
            and prepared[0]["history"] == []
            and proof["cold_query_history"]["returned_history"] == [],
            actual_full_quote_in_cold_model=any(
                case["message"] in json.dumps(c["request"], ensure_ascii=False)
                for c in cloud_calls[query["cloud_call_range"][0] : query["cloud_call_range"][1]]
            ),
            cold_reply_complete=all(
                word in reply for word in ["南湾杉谷", "木刻版画", "MH-582-Z", "有效", "青绿色油墨"]
            )
            and any(w in reply for w in ["未参加", "尚未参加", "没有参加"])
            and any(w in reply for w in ["未出发", "没有出发", "尚未出发"])
            and ("11月14日" in reply or "11 月 14 日" in reply or "2026-11-14" in reply),
            cold_reply_is_actual_model=any(
                (c["response"].get("choices") or [{}])[0].get("message", {}).get("content") == reply
                for c in cloud_calls[query["cloud_call_range"][0] : query["cloud_call_range"][1]]
            ),
        )
    return checks


def audit_late_capture_guards(proof, case, cloud_calls):
    return dict(
        normal_auth=proof["auth_statuses"] == [200, 200],
        complete_new_input=len(case["message"]) > 150 and bool(proof["eligible_rule_hints"]),
        real_model_only=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        full_original_input_in_model=any(
            case["message"] in json.dumps(c["request"], ensure_ascii=False) for c in cloud_calls
        ),
        admitted_before_real_lock=proof["controlled_lock"]["after_valid_admission_and_real_generation"],
        real_pg_lock_wait=any(r["owner_fence_lock_query"] for r in proof["actual_lock_waiters"]),
        capture_thread_live_at_api_return=not proof["capture_thread_done_at_api_return"]
        and proof["sync_pending_at_api_return"] > 0,
        source_pending_at_api_return=proof["source_at_api_return_while_locked"]["state"] == "pending"
        and proof["source_at_api_return_while_locked"]["body"] is None,
        lock_released=proof["owner_lock_released"] and proof["lock_waiters_remaining"] == 0,
        actual_threads_jobs_terminal=proof["capture_thread_terminal"]
        and proof["jobs_terminal"]
        and proof["sync_pending_final"] == 0,
        old_sources_unchanged=all(r in proof["scope_owner_sources_after"] for r in proof["scope_owner_sources_before"]),
        old_claims_unchanged=all(r in proof["scope_owner_claims_after"] for r in proof["scope_owner_claims_before"]),
        old_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        source_fixture_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        no_old_calls=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
    )
