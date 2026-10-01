"""Hold one real writer while a second complete native message meets capacity."""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict

import asyncpg

from character.memory_llm import get_memory_enrichment_scheduler
from db.memory_source import source_identity, source_scope
from services.character_context import CharacterContextService


async def run_capacity_probe(
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
    warmup, target, cold = fixture["cases"][-3:]
    scheduler = get_memory_enrichment_scheduler()
    assert scheduler._capacity == 1 and scheduler._idle_seconds == 0
    held = asyncio.Event()
    release = asyncio.Event()
    original_process = scheduler._process_job
    outcomes = []
    original_complete = CharacterContextService.complete_turn
    original_history = CharacterContextService._load_history
    proof["controlled_capacity"] = dict(
        queue_size=1,
        idle_seconds=0,
        production_queue_not_changed=True,
        holding="One real accepted job before its original processing, no fabricated model response or inflight counter",
    )

    async def held_process(job):
        if job.source_message_id == args.run_label + "-warmup":
            proof["held_job"] = dict(
                source_message_id=job.source_message_id, message=job.message, observed_at=job.observed_at.isoformat()
            )
            held.set()
            await asyncio.wait_for(release.wait(), timeout=120)
        return await original_process(job)

    async def observed_complete(service, prepared, turn, reply, **kwargs):
        result = await original_complete(service, prepared, turn, reply, **kwargs)
        outcomes.append(
            dict(
                message=turn.message,
                source_message_id=kwargs.get("source_message_id"),
                received_at=prepared.received_at.isoformat(),
                outcome=asdict(result),
            )
        )
        return result

    async def cold_history(service, turn, scope, character_id):
        history = await original_history(service, turn, scope, character_id)
        if turn.message == cold["message"]:
            proof["cold_query_history"] = dict(
                original_history=history, returned_history=[], only_case=cold["id"], stored_archive_not_changed=True
            )
            return []
        return history

    scheduler._process_job = held_process
    CharacterContextService.complete_turn = observed_complete
    if args.capacity_cold_query:
        CharacterContextService._load_history = cold_history

    async def source_metadata(source_id):
        key = source_identity(source_scope(*fields), source_id)["source_key"]
        connection = await asyncpg.connect(user="boot", database=database_name, host=str(root / "socket"), port=25433)
        try:
            assert await connection.fetchval("SHOW data_directory") == str(root / "data")
            row = await connection.fetchrow(
                "SELECT state,body,observed_at,body_digest FROM memory_sources WHERE source_key=$1", key
            )
            result = dict(row) if row else None
            if result is not None:
                result["terms"] = await connection.fetchval(
                    "SELECT count(*) FROM memory_source_terms WHERE source_key=$1", key
                )
                result["links"] = await connection.fetchval(
                    "SELECT count(*) FROM memory_source_links WHERE source_key=$1", key
                )
            return result
        finally:
            await connection.close()

    async def send(case, source_suffix):
        source_id = args.run_label + "-" + source_suffix
        assert database.memory_source_admission(*fields, source_message_id=source_id, body=case["message"]) == "new"
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
                id=case["id"],
                source_message_id=source_id,
                message=case["message"],
                http_status=response.status_code,
                response=response.json(),
                cloud_call_range=[before, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))

    try:
        await send(warmup, "warmup")
        await asyncio.wait_for(held.wait(), timeout=45)
        proof["held_status_before_target"] = asdict(scheduler.status)
        proof["held_inflight_before_target"] = scheduler._inflight
        assert proof["held_status_before_target"]["processing"] == 1 and scheduler._inflight == 1
        await send(target, "target")
        proof["target_while_writer_held"] = await source_metadata(args.run_label + "-target")
        proof["status_after_capacity_target"] = asdict(scheduler.status)
        proof["target_visible_while_writer_held"] = database.list_memory_sources(
            *fields, source_message_ids=(args.run_label + "-target",), limit=1
        )
    finally:
        release.set()
    assert await scheduler.flush_memory(timeout=90)
    proof["held_job_released_and_terminal"] = scheduler._processing == scheduler._inflight == 0
    proof["target_after_release"] = await source_metadata(args.run_label + "-target")
    proof["target_claims_after_release"] = [
        row
        for row in database.list_character_memory_claims(*fields, limit=None, include_inactive=True)
        if row["source_message_id"] == args.run_label + "-target"
    ]
    proof["warmup_after_release"] = await source_metadata(args.run_label + "-warmup")
    if args.capacity_cold_query:
        await send(cold, "cold-query")
        assert await scheduler.flush_memory(timeout=90)
    proof["completion_outcomes"] = outcomes
    proof["memory_status"] = asdict(scheduler.status)
    proof["scope_owner_sources_after"] = database.list_memory_sources(*fields, limit=100)
    proof["scope_owner_claims_after"] = database.list_character_memory_claims(
        *fields, limit=None, include_inactive=True
    )
    proof["queue_stats"] = runtime.stats()
    connection = await asyncpg.connect(user="boot", database=database_name, host=str(root / "socket"), port=25433)
    try:
        rows = await connection.fetch(
            'SELECT * FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id',
            "web",
            "web-character",
            identity,
            fields[0],
        )
        old_rows = [dict(row) for row in rows if row["id"] <= proof["scope_sql"]["owner_message_max_id_before"]]
        proof["scope_sql"]["owner_original_messages_sha256_after"] = hashlib.sha256(
            json.dumps(old_rows, sort_keys=True, default=str).encode()
        ).hexdigest()
        proof["scope_sql"]["new_messages"] = [
            dict(row) for row in rows if row["id"] > proof["scope_sql"]["owner_message_max_id_before"]
        ]
    finally:
        await connection.close()
    connection = await asyncpg.connect(user="boot", database=source_database, host=str(root / "socket"), port=25433)
    try:
        snapshots = {}
        for table in ["memory_sources", "memory_source_terms", "character_memories", "messages"]:
            rows = await connection.fetch("SELECT * FROM " + table)
            snapshots[table] = hashlib.sha256(
                json.dumps(
                    sorted([dict(row) for row in rows], key=lambda row: json.dumps(row, sort_keys=True, default=str)),
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
        proof["scope_sql"]["source_database_snapshot_after"] = snapshots
    finally:
        await connection.close()
    capture_storage_proof(proof, output)
    proof["configured_model"] = os.environ["OPENAI_COMPAT_MODEL"]
    proof["configured_context_window"] = int(os.environ["OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS"])
    proof["guard_checks"] = dict(
        normal_auth=proof["auth_statuses"] == [200, 200],
        two_new_complete_messages=all(len(case["message"]) > 150 for case in [warmup, target]),
        actual_occupied_capacity=proof["held_status_before_target"]["processing"]
        == proof["held_inflight_before_target"]
        == 1,
        real_capacity_skip=any(
            row.get("reason") == "capacity" for row in proof["status_after_capacity_target"]["recent_results"]
        ),
        capacity_still_bounded=scheduler._capacity == 1,
        actual_large_model_only=bool(cloud_calls)
        and all(call["response"].get("model") == "deepseek-v4-pro" for call in cloud_calls),
        old_sources_unchanged=all(
            row in proof["scope_owner_sources_after"] for row in proof["scope_owner_sources_before"]
        ),
        old_claims_unchanged=all(
            row in proof["scope_owner_claims_after"] for row in proof["scope_owner_claims_before"]
        ),
        old_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        source_fixture_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        original_writer_released_and_terminal=proof["held_job_released_and_terminal"],
        no_old_writes_answers_imports_search=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
        current_source_recall_enabled=proof["controlled_ablation"]["raw_source_recall_enabled"] is True,
    )
    proof["checks"] = {**proof["guard_checks"], **evaluate_capacity_retention(proof, cloud_calls, fixture)}
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            dict(
                result=str(output / "result.json"),
                native_requests=len(proof["generation"]),
                cloud_calls=len(cloud_calls),
                guard_checks=proof["guard_checks"],
            ),
            ensure_ascii=False,
        )
    )

    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]


def evaluate_capacity_retention(proof, calls, fixture):
    """Full raw speech survives a real capacity skip, independently of facts."""
    import re
    from datetime import datetime
    from html import unescape

    target = fixture["cases"][-2]
    generations = [row for row in proof["generation"] if row["id"] == target["id"]]
    outcomes = [row for row in proof["completion_outcomes"] if row["message"] == target["message"]]
    if len(generations) != 1 or len(outcomes) != 1:
        return {"one_target_request_and_completion": False}
    generated, completed = generations[0], outcomes[0]
    source = proof["target_while_writer_held"]
    selected = calls[slice(*generated["cloud_call_range"])]
    answer = [row for row in selected if row["request"].get("max_tokens") == 1024]
    checks = dict(
        target_accepted_natively=generated["http_status"] == 200,
        complete_target_in_actual_model=len(answer) == 1
        and target["message"] in unescape(answer[0]["request"]["messages"][-1]["content"]),
        source_durable_before_held_writer_released=source is not None
        and source["state"] == "recorded"
        and source["body"] == target["message"],
        pending_digest_cleared=source is not None and source["body_digest"] is None,
        target_terms_indexed=source is not None and source["terms"] > 0,
        exact_trusted_receipt_used=source is not None
        and datetime.fromisoformat(source["observed_at"]) == datetime.fromisoformat(completed["received_at"]),
        completed_source_receipt_explicit=completed["outcome"].get("source_capture") == "recorded",
        semantic_capacity_skip_still_happens=completed["outcome"]["memory_enrichment_scheduled"] is False,
        whole_quote_visible_before_release=len(proof["target_visible_while_writer_held"]) == 1
        and proof["target_visible_while_writer_held"][0]["body"] == target["message"],
        source_not_dependent_on_other_job=proof["target_after_release"] == source,
        no_unreviewed_target_fact_promoted=proof["target_claims_after_release"] == []
        and source is not None
        and source["links"] == 0,
        no_target_semantic_job_bypassed_capacity=not any(
            row.get("source_message_id") == generated["source_message_id"]
            for row in proof["memory_status"]["recent_results"]
        ),
    )
    cold = [row for row in proof["generation"] if row["id"] == fixture["cases"][-1]["id"]]
    if cold:
        question = cold[0]
        diags = [row for row in proof["prepared_diagnostics"] if row["query"] == fixture["cases"][-1]["message"]]
        answer = [
            row for row in calls[slice(*question["cloud_call_range"])] if row["request"].get("max_tokens") == 1024
        ]
        diag = diags[0] if len(diags) == 1 else {}
        wire = "\n".join(unescape(m["content"]) for m in answer[0]["request"]["messages"]) if len(answer) == 1 else ""
        reply = question["response"].get("reply", "")
        checks.update(
            one_new_cold_query_accepted=len(cold) == 1 and question["http_status"] == 200,
            cold_query_has_no_history=bool(diag)
            and diag["history"] == []
            and proof["cold_query_history"]["returned_history"] == [],
            cold_query_uses_real_authorized_raw_source=bool(diag)
            and diag["raw_source_status"] == "available"
            and generated["source_message_id"] in diag["raw_source_diagnostics"]["selected_ids"],
            full_target_in_episodic_packet=target["message"] in diag.get("episodic_context", ""),
            full_target_reaches_actual_cold_model=target["message"] in wire and fixture["cases"][-1]["message"] in wire,
            raw_quote_not_system_instruction=bool(answer)
            and all(
                target["message"] not in m["content"] for m in answer[0]["request"]["messages"] if m["role"] == "system"
            ),
            actual_recalled_receipt_and_course="TQ-762-M" in reply and "南汀青禾工坊" in reply and "浅浮雕课" in reply,
            actual_recalled_date=bool(re.search(r"2026\s*年\s*10\s*月\s*25\s*日", reply)),
            actual_recalled_validity_and_no_attendance="有效" in reply
            and bool(re.search(r"(?:尚未|还没|没有|未).{0,8}参加", reply))
            and bool(re.search(r"(?:尚未|还没|没有|未).{0,8}出发", reply)),
            actual_friend_not_owner="祁岚" in reply and bool(re.search(r"不是你本人|不是你的|而不是你", reply)),
        )
    return checks
