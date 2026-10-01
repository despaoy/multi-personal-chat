"""Real isolated PG capture failure after genuine pro generation; no fake reply."""

import hashlib
import json
import os
from dataclasses import asdict

import asyncpg

from character.context_builder import build_user_scope
from character.memory_llm import get_memory_enrichment_scheduler
from db.memory_source import source_identity, source_scope
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService


async def run_history_feedback_probe(
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
    proof["native_payloads"] = []
    case, question = fixture["cases"][-2:]
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
                controlled_pg_trigger_error="stage31 controlled capture failure" in str(exc),
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

    async def send(tag, selected=case):
        current_source = source_id if selected is case else args.run_label + "-status-question"
        before = len(cloud_calls)
        payload = dict(
            message=selected["message"],
            characterId=fields[0],
            sessionId=args.run_label,
            sessionType="private",
            sourceMessageId=current_source,
            history=[],
        )
        proof["native_payloads"].append(payload)
        response = await client.post("/api/generate", json=payload)
        proof["generation"].append(
            dict(
                id=selected["id"] + "-" + tag,
                source_message_id=current_source,
                message=selected["message"],
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
            "CREATE FUNCTION stage31_capture_fault() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.source_key='"
            + literal
            + "' AND NEW.body IS NOT NULL THEN RAISE EXCEPTION 'stage31 controlled capture failure' USING ERRCODE='55000'; END IF; RETURN NEW; END $$"
        )
        await c.execute(
            "CREATE TRIGGER stage31_capture_fault BEFORE INSERT OR UPDATE ON memory_sources FOR EACH ROW EXECUTE FUNCTION stage31_capture_fault()"
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
            await c.execute("DROP TRIGGER stage31_capture_fault ON memory_sources")
            await c.execute("DROP FUNCTION stage31_capture_fault()")
            proof["fault_removed"] = True
        finally:
            await c.close()
    scope = build_user_scope(fields[1], fields[2], identity, identity, "private")
    proof["history_after_failed_save"] = await DatabaseMessageRepository(database).list_recent_conversation_history(
        scope, character_id=fields[0], limit=8, max_chars=8000
    )
    proof["scoped_turns_after_failed_save"] = asdict(
        await DatabaseMessageRepository(database).list_scoped_turns(scope, fields[0], limit=8)
    )
    assert (
        database.memory_source_admission(
            *fields, source_message_id=args.run_label + "-status-question", body=question["message"]
        )
        == "new"
    )
    await send("history-status", question)
    assert await scheduler.flush_memory(timeout=90)
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
            "SELECT count(*) FROM pg_trigger WHERE tgname='stage31_capture_fault'"
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
    proof["feedback_checks"] = audit_history_feedback(proof, case, question, cloud_calls)
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


def audit_history_feedback(proof, case, question, cloud_calls):
    first, second = proof["generation"]
    warning = next((w for w in first["response"].get("warnings") or [] if "保存失败" in w), "")
    first_reply = first["response"]["reply"]
    stored = [r for r in proof["scope_sql"]["new_messages"] if r["sourceMessageId"] == first["source_message_id"]]
    question_prepare = [r for r in proof["prepared_diagnostics"] if r["query"] == question["message"]]
    query_calls = cloud_calls[second["cloud_call_range"][0] : second["cloud_call_range"][1]]
    raw_answer = first_reply.split("\n\n保存提示：", 1)[0]
    query_answer = second["response"]["reply"]
    negative = any(
        word in query_answer
        for word in ["保存失败", "未能保存", "未成功保存", "没有成功保存", "没有保存", "未保存", "没能保存"]
    )
    checks = dict(
        current_api_truthful_notice=bool(warning) and warning in first_reply,
        actual_model_content_kept=any(
            (c["response"].get("choices") or [{}])[0].get("message", {}).get("content") == raw_answer
            for c in cloud_calls[first["cloud_call_range"][0] : first["cloud_call_range"][1]]
        ),
        exactly_one_target_archive=len(stored) == 1,
        archive_matches_final_response=len(stored) == 1 and stored[0]["reply"] == first_reply,
        full_user_body_unchanged=len(stored) == 1 and stored[0]["message"] == case["message"],
        repository_history_keeps_notice=any(
            r["role"] == "assistant" and r["content"] == first_reply for r in proof["history_after_failed_save"]
        ),
        scoped_turn_keeps_notice=any(
            u["role"] == "assistant" and u["text"] == first_reply
            for t in proof["scoped_turns_after_failed_save"]["turns"]
            for u in t["utterances"]
        ),
        new_question_uses_server_history=bool(question_prepare)
        and any(r["role"] == "assistant" and r["content"] == first_reply for r in question_prepare[0]["history"]),
        notice_in_actual_question_model_input=bool(warning)
        and any(
            m.get("role") == "assistant" and warning in m.get("content", "")
            for c in query_calls
            for m in c["request"]["messages"]
        ),
        actual_question_understands_failed_state=negative and "长期记忆" in query_answer,
        question_answer_not_faked=any(
            (c["response"].get("choices") or [{}])[0].get("message", {}).get("content") == query_answer
            for c in query_calls
        ),
        question_input_complete=second["message"] == question["message"],
        no_client_history_injected=len(proof.get("native_payloads", [])) == 2
        and all(p.get("history") == [] for p in proof["native_payloads"]),
        notice_not_source_or_owner_fact=proof["source_after_failure"]["body"] is None
        and not proof["claims_after_failure"],
    )
    return checks
