"""Real owned completion occupancy and a new admitted-turn persistence boundary."""

import asyncio
import hashlib
import json
import threading
from dataclasses import asdict

import asyncpg

from character.memory_extractor import extract_preferred_address
from character.memory_llm import get_memory_enrichment_scheduler
from db.memory_source import source_identity, source_scope
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services import turn_completion
from services.character_context import CharacterContextService


async def run_turn_capacity_probe(
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
    assert extract_preferred_address(holder["message"]) == "杉岚"
    assert database.memory_source_admission(*fields, source_message_id=holder_id, body=holder["message"]) == "new"
    assert database.memory_source_admission(*fields, source_message_id=target_id, body=target["message"]) == "new"
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
        key = source_identity(source_scope(*fields), source_id)["source_key"]
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
        if len(pargs) < 8 or pargs[7] != "杉岚":
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
        if state.preferred_address == "杉岚" and not locked:
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

    async def send(case, source_id, suffix):
        payload = dict(
            message=case["message"],
            characterId=fields[0],
            sessionId=args.run_label + "-" + suffix,
            sessionType="private",
            sourceMessageId=source_id,
            history=[],
        )
        proof["native_payloads"].append(payload)
        before = len(cloud_calls)
        response = await client.post("/api/generate", json=payload)
        proof["generation"].append(
            dict(
                id=case["id"],
                source_message_id=source_id,
                message=case["message"],
                http_status=response.status_code,
                response=response.json(),
                cloud_call_range=[before, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))

    holder_request = asyncio.create_task(send(holder, holder_id, "holder"), name="stage33-original-holder-request")
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
    proof["target_after_holder_release"] = await source(target_id)
    if args.turn_capacity_retry:
        assert proof["generation"][1]["http_status"] == 503
        proof["retry_same_original_body_and_source"] = True
        await send(target, target_id, "retry")
    assert await scheduler.flush_memory(timeout=90)
    await owned.shutdown(timeout=35)
    proof["target_final_source"] = await source(target_id)
    proof["target_final_visible_sources"] = database.list_memory_sources(
        *fields, source_message_ids=(target_id,), limit=1
    )
    proof["target_final_claims"] = [
        r
        for r in database.list_character_memory_claims(*fields, limit=None, include_inactive=True)
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
    proof["guard_checks"] = audit_turn_capacity_guards(proof, fixture, cloud_calls)
    proof["checks"] = {**proof["guard_checks"], **audit_turn_capacity(proof, fixture, cloud_calls)}
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


def audit_turn_capacity_guards(proof, fixture, cloud_calls):
    holder, target = fixture["cases"][-2:]
    return dict(
        normal_auth=proof["auth_statuses"] == [200, 200],
        complete_inputs=len(holder["message"]) > 150 and len(target["message"]) > 150,
        real_current_pro=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        holder_full_input_in_actual_model=any(
            holder["message"] in json.dumps(c["request"], ensure_ascii=False) for c in cloud_calls
        ),
        real_runtime_and_capacity=proof["controlled_capacity"]["actual_runtime_instance"]
        and proof["controlled_capacity"]["isolated_capacity"] == 1
        and proof["actual_owned_active_before_target"] == 1,
        real_held_task_and_thread=proof["actual_holder_task_live"]
        and proof["actual_holder_thread_live"]
        and proof["holder_thread_live_at_busy_return"],
        actual_pg_relationship_wait=any(r["relationship_write"] for r in proof["actual_lock_waiters"]),
        complete_holder_source_before_lock=proof["holder_source_before_lock"]["state"] == "recorded"
        and proof["holder_source_before_lock"]["body"] == holder["message"],
        real_capacity_still_full_at_target_return=proof["owned_active_at_busy_return"] == 1,
        lock_released=proof["relationship_lock_released"] and proof["lock_waiters_remaining"] == 0,
        threads_jobs_runtime_terminal=proof["holder_completion_terminal"]["done"]
        and not proof["holder_completion_terminal"]["cancelled"]
        and proof["jobs_terminal"]
        and proof["sync_pending_final"] == 0
        and proof["completion_runtime_terminal"]["active"] == proof["completion_runtime_terminal"]["reserved"] == 0,
        old_sources_unchanged=all(r in proof["scope_owner_sources_after"] for r in proof["scope_owner_sources_before"]),
        old_claims_unchanged=all(r in proof["scope_owner_claims_after"] for r in proof["scope_owner_claims_before"]),
        old_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        parent_fixture_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        isolated_timeout_restored=proof["isolated_operation_timeout"]["restored"],
        no_history_injected=all(p["history"] == [] for p in proof["native_payloads"]),
        no_old_calls=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
    )


def audit_turn_capacity(proof, fixture, cloud_calls):
    holder, target = fixture["cases"][-2:]
    busy = proof["generation"][1]
    before, after = busy["cloud_call_range"]
    source = proof["target_final_source"]
    target_id = busy["source_message_id"]
    detail = busy["response"].get("detail")
    detail = detail if isinstance(detail, dict) else {}
    checks = dict(
        busy_rejected_before_generation=busy["http_status"] == 503 and detail.get("code") == "turn_completion_busy",
        no_busy_model_calls=before == after,
        no_busy_context_preparation=not any(r["query"] == target["message"] for r in proof["prepared_at_busy_return"]),
        no_busy_target_archive=not any(
            r["sourceMessageId"] == target_id and r["sessionId"] == proof["native_payloads"][1]["sessionId"]
            for r in proof["scope_sql"]["new_messages"]
        ),
        no_target_completion_without_slot=not any(
            r["source_message_id"] == target_id for r in proof["completion_outcomes_at_busy_return"]
        ),
    )
    if len(proof["generation"]) == 3:
        retry = proof["generation"][2]
        previous = proof["target_at_busy_return"]
        checks.update(
            same_body_source_retry=proof.get("retry_same_original_body_and_source") is True
            and retry["message"] == busy["message"] == target["message"]
            and retry["source_message_id"] == target_id,
            retry_generates_once=retry["http_status"] == 200
            and retry["cloud_call_range"][1] > retry["cloud_call_range"][0],
            complete_original_source_saved=source is not None
            and source["state"] == "recorded"
            and source["body"] == target["message"]
            and source["body_digest"] is None
            and source["terms"] > 0,
            first_binding_receipt_preserved=previous is not None
            and source is not None
            and previous["observed_at"] == source["observed_at"],
            full_original_source_readable=any(
                r["body"] == target["message"] for r in proof["target_final_visible_sources"]
            ),
            actual_target_writer_executed=any(
                r.get("source_message_id") == target_id and r.get("source_capture") == "recorded"
                for r in proof["memory_status"]["recent_results"]
            ),
            complete_target_in_actual_pro=any(
                target["message"] in json.dumps(c["request"], ensure_ascii=False)
                for c in cloud_calls[retry["cloud_call_range"][0] : retry["cloud_call_range"][1]]
            ),
            third_party_not_owner_fact=all(
                json.loads(r.get("metadata_json") or "{}").get("content_semantics") == "quoted_source"
                or json.loads(r.get("metadata_json") or "{}").get("attributed_to") != "user"
                for r in proof["target_final_claims"]
            ),
        )
    return checks
