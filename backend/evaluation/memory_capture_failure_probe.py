"""Real isolated PG capture failure after genuine pro generation; no fake reply."""

import hashlib
import json
import os
from dataclasses import asdict

import asyncpg

from character.memory_llm import get_memory_enrichment_scheduler
from db.memory_source import source_identity, source_scope
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import CharacterContextService


async def run_capture_failure_probe(
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
    case = fixture["cases"][-1]
    source_id = args.run_label + "-target"
    key = source_identity(source_scope(*fields), source_id)["source_key"]
    assert database.memory_source_admission(*fields, source_message_id=source_id, body=case["message"]) == "new"
    assert len(case["message"]) > 150
    scheduler = get_memory_enrichment_scheduler()
    outcomes = []
    attempts = []
    original_complete = CharacterContextService.complete_turn
    original_capture = DatabaseCharacterMemoryRepository.capture_source

    async def observed_capture(repo, *pargs, **kwargs):
        if kwargs.get("source_message_id") != source_id:
            return await original_capture(repo, *pargs, **kwargs)
        row = dict(source_message_id=source_id, body=kwargs["body"], observed_at=kwargs["observed_at"].isoformat())
        attempts.append(row)
        try:
            row["result"] = await original_capture(repo, *pargs, **kwargs)
            return row["result"]
        except Exception as exc:
            row.update(
                exception_type=type(exc).__name__,
                controlled_pg_trigger_error="stage30 controlled capture failure" in str(exc),
            )
            raise

    async def observed_complete(service, prepared, turn, reply, **kwargs):
        result = await original_complete(service, prepared, turn, reply, **kwargs)
        outcomes.append(
            dict(
                source_message_id=kwargs.get("source_message_id"),
                message=turn.message,
                received_at=prepared.received_at.isoformat(),
                outcome=asdict(result),
            )
        )
        return result

    CharacterContextService.complete_turn = observed_complete
    DatabaseCharacterMemoryRepository.capture_source = observed_capture

    async def connect(name=database_name):
        connection = await asyncpg.connect(user="boot", database=name, host=str(root / "socket"), port=25433)
        assert await connection.fetchval("SHOW data_directory") == str(root / "data")
        return connection

    async def metadata():
        c = await connect()
        try:
            row = await c.fetchrow(
                "SELECT state,body,observed_at,body_digest FROM memory_sources WHERE source_key=$1", key
            )
            result = dict(row) if row else None
            if result is not None:
                result["terms"] = await c.fetchval("SELECT count(*) FROM memory_source_terms WHERE source_key=$1", key)
                result["links"] = await c.fetchval("SELECT count(*) FROM memory_source_links WHERE source_key=$1", key)
            return result
        finally:
            await c.close()

    async def send(tag):
        before = len(cloud_calls)
        response = await client.post(
            "/api/generate",
            json=dict(
                message=case["message"],
                characterId=fields[0],
                sessionId=args.run_label,
                sessionType="private",
                sourceMessageId=source_id,
            ),
        )
        proof["generation"].append(
            dict(
                id=case["id"] + "-" + tag,
                source_message_id=source_id,
                message=case["message"],
                http_status=response.status_code,
                response=response.json(),
                cloud_call_range=[before, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))

    c = await connect()
    try:
        # Only this new source's pending-to-recorded transition fails. Pending
        # admission, ordinary archive and interaction updates remain real.
        literal = key.replace("'", "''")
        await c.execute(
            "CREATE FUNCTION stage30_capture_fault() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.source_key='"
            + literal
            + "' AND NEW.body IS NOT NULL THEN RAISE EXCEPTION 'stage30 controlled capture failure' USING ERRCODE='55000'; END IF; RETURN NEW; END $$"
        )
        await c.execute(
            "CREATE TRIGGER stage30_capture_fault BEFORE INSERT OR UPDATE ON memory_sources FOR EACH ROW EXECUTE FUNCTION stage30_capture_fault()"
        )
        proof["controlled_fault"] = dict(
            kind="real PG trigger SQLSTATE55000",
            cloned_database_only=True,
            only_new_source_key=key,
            pending_reservation_allowed=True,
            model_not_faked=True,
            production_not_changed=True,
        )
    finally:
        await c.close()
    try:
        await send("failed-save")
        assert await scheduler.flush_memory(timeout=90)
        proof["source_after_failure"] = await metadata()
        proof["visible_after_failure"] = database.list_memory_sources(*fields, source_message_ids=(source_id,), limit=1)
        proof["claims_after_failure"] = [
            r
            for r in database.list_character_memory_claims(*fields, limit=None, include_inactive=True)
            if r["source_message_id"] == source_id
        ]
        proof["scheduler_after_failure"] = asdict(scheduler.status)
        proof["capture_attempts_after_failure"] = list(attempts)
    finally:
        c = await connect()
        try:
            await c.execute("DROP TRIGGER stage30_capture_fault ON memory_sources")
            await c.execute("DROP FUNCTION stage30_capture_fault()")
            proof["fault_removed"] = True
        finally:
            await c.close()
    if args.capture_retry:
        assert database.memory_source_admission(*fields, source_message_id=source_id, body=case["message"]) == "pending"
        await send("retry-after-fault-removal")
        assert await scheduler.flush_memory(timeout=90)
        proof["source_after_retry"] = await metadata()
        proof["visible_after_retry"] = database.list_memory_sources(*fields, source_message_ids=(source_id,), limit=1)
    proof["completion_outcomes"] = outcomes
    proof["capture_attempts"] = attempts
    proof["memory_status"] = asdict(scheduler.status)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
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
        proof["fault_objects_remaining"] = await c.fetchval(
            "SELECT count(*) FROM pg_trigger WHERE tgname='stage30_capture_fault'"
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
    proof["guard_checks"] = dict(
        normal_auth=proof["auth_statuses"] == [200, 200],
        complete_new_input=len(case["message"]) > 150,
        actual_pg_capture_failure=bool(attempts) and attempts[0].get("controlled_pg_trigger_error") is True,
        successful_actual_model_before_failure=proof["generation"][0]["http_status"] == 200
        and bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        full_original_input_in_actual_model=any(
            case["message"] in json.dumps(c["request"], ensure_ascii=False) for c in cloud_calls
        ),
        failed_source_not_readable=proof["source_after_failure"]["state"] == "pending"
        and proof["source_after_failure"]["body"] is None
        and bool(proof["source_after_failure"]["body_digest"])
        and proof["source_after_failure"]["terms"] == 0
        and proof["visible_after_failure"] == [],
        no_failed_source_fact=proof["claims_after_failure"] == [],
        failed_capture_no_semantic_enqueue=outcomes[0]["outcome"]["source_capture"] == "failed"
        and not outcomes[0]["outcome"]["memory_enrichment_scheduled"],
        jobs_terminal=proof["jobs_terminal"],
        fault_removed=proof["fault_removed"] and proof["fault_objects_remaining"] == 0,
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
    proof["feedback_checks"] = audit_capture_feedback(proof, case, cloud_calls)
    proof["checks"] = {**proof["guard_checks"], **proof["feedback_checks"]}
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            dict(
                native_requests=len(proof["generation"]),
                cloud_calls=len(cloud_calls),
                guard_checks=proof["guard_checks"],
                feedback_checks=proof["feedback_checks"],
            ),
            ensure_ascii=False,
        )
    )
    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]


def audit_capture_feedback(proof, case, cloud_calls):
    response = proof["generation"][0]["response"]
    warnings = response.get("warnings") or []
    checks = dict(
        failed_save_has_explicit_warning=any("长期记忆" in w and "保存失败" in w for w in warnings),
        failed_save_notice_in_reply=any(w in response.get("reply", "") for w in warnings if "保存失败" in w),
    )
    checks["actual_generated_content_retained"] = all(
        any(
            (call["response"].get("choices") or [{}])[0].get("message", {}).get("content")
            == row["response"]["reply"].split("\n\n保存提示：", 1)[0]
            for call in cloud_calls[row["cloud_call_range"][0] : row["cloud_call_range"][1]]
        )
        for row in proof["generation"]
    )
    checks["archive_keeps_actual_model_content"] = all(
        any(
            message["reply"] == row["response"]["reply"].split("\n\n保存提示：", 1)[0]
            and message["message"] == case["message"]
            for message in proof["scope_sql"]["new_messages"]
        )
        for row in proof["generation"]
    )
    first = proof["capture_attempts_after_failure"][0]
    source = proof["source_after_failure"]
    checks["actual_capture_failure_matches_complete_source"] = (
        first.get("controlled_pg_trigger_error") is True
        and first["body"] == case["message"]
        and first["observed_at"] == source["observed_at"]
    )
    checks["failed_source_remains_unreadable"] = (
        source["state"] == "pending"
        and source["body"] is None
        and source["terms"] == 0
        and proof["visible_after_failure"] == []
    )
    outcome = proof["completion_outcomes"][0]["outcome"]
    checks["failed_outcome_not_reported_success"] = (
        outcome["source_capture"] == "failed"
        and outcome["memory_enrichment_status"] == "source_capture_failed"
        and not outcome["memory_enrichment_scheduled"]
    )
    if "source_after_retry" in proof:
        source = proof["source_after_retry"]
        checks.update(
            retry_stores_exact_complete_body=source["state"] == "recorded"
            and source["body"] == case["message"]
            and source["body_digest"] is None
            and source["terms"] > 0,
            retry_preserves_first_receipt=source["observed_at"] == proof["source_after_failure"]["observed_at"],
            retry_is_readable=any(r["body"] == case["message"] for r in proof["visible_after_retry"]),
            retry_warning_cleared=not any(
                "保存失败" in w for w in proof["generation"][1]["response"].get("warnings") or []
            ),
            retry_capture_recorded=proof["completion_outcomes"][1]["outcome"]["source_capture"] == "recorded",
            retry_does_not_force_owner_fact=not any(
                r["source_message_id"] == proof["generation"][1]["source_message_id"]
                for r in proof["scope_owner_claims_after"]
            ),
        )
    return checks
