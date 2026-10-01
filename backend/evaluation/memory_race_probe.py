"""One isolated, normally authenticated, complete new-source concurrency probe."""

import asyncio
import hashlib
import json
import os

import asyncpg

from api import generate as generation_api
from character.memory_llm import get_memory_enrichment_scheduler


async def run_memory_race(
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
    from dataclasses import asdict

    from db.memory_source import source_identity, source_scope

    source_id = "web:stage28-fresh-concurrent-identity"
    scope_fields = ("tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity)
    source_key = source_identity(source_scope(*scope_fields), source_id)["source_key"]
    race_cases = fixture["cases"][-2:]
    assert (
        database.memory_source_admission(*scope_fields, source_message_id=source_id, body=race_cases[0]["message"])
        == "new"
    )
    proof["race_source_id"] = source_id
    proof["source_initially_new"] = True
    proof["race_requests"] = [
        dict(
            case_id=case["id"],
            message=case["message"],
            sourceMessageId=source_id,
            sessionId=args.run_label + "-" + case["id"],
            characterId="tsukiyashiro_kisaki",
            sessionType="private",
        )
        for case in race_cases
    ]
    original_validation = generation_api._validate_web_source_identity
    validations = []
    gate = asyncio.Event()

    async def observed_validation(request, current_user, **kwargs):
        outcome = dict(message=request.message, sessionId=request.sessionId, sourceMessageId=request.sourceMessageId)
        try:
            await original_validation(request, current_user, **kwargs)
            outcome["admitted"] = True
        except Exception as exc:
            outcome["admitted"] = False
            outcome["status"] = getattr(exc, "status_code", None)
            outcome["detail"] = getattr(exc, "detail", None)
            raise
        finally:
            validations.append(outcome)
            if len(validations) == 2:
                gate.set()
            await asyncio.wait_for(gate.wait(), timeout=15)

    generation_api._validate_web_source_identity = observed_validation

    async def send(case, request):
        response = await client.post(
            "/api/generate", json={key: value for key, value in request.items() if key != "case_id"}
        )
        return dict(id=case["id"], http_status=response.status_code, response=response.json())

    proof["generation"] = await asyncio.gather(
        *(send(case, request) for case, request in zip(race_cases, proof["race_requests"]))
    )
    proof["validation_observations"] = validations
    proof["validation_barrier"] = (
        "Observe both actual validation outcomes, then release together; never bypass the implementation; distinct client sessions and same authenticated private source scope."
    )
    assert await get_memory_enrichment_scheduler().flush_memory(timeout=90)
    proof["scope_owner_sources_after"] = database.list_memory_sources(*scope_fields, limit=100)
    proof["scope_owner_claims_after"] = database.list_character_memory_claims(
        *scope_fields, limit=None, include_inactive=True
    )
    proof["memory_status"] = asdict(get_memory_enrichment_scheduler().status)
    proof["queue_stats"] = runtime.stats()
    connection = await asyncpg.connect(user="boot", database=database_name, host=str(root / "socket"), port=25433)
    try:
        assert await connection.fetchval("SHOW data_directory") == str(root / "data")
        rows = await connection.fetch(
            'SELECT * FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id',
            "web",
            "web-character",
            identity,
            "tsukiyashiro_kisaki",
        )
        old_rows = [dict(row) for row in rows if row["id"] <= proof["scope_sql"]["owner_message_max_id_before"]]
        proof["scope_sql"]["owner_original_messages_sha256_after"] = hashlib.sha256(
            json.dumps(old_rows, sort_keys=True, default=str).encode()
        ).hexdigest()
        proof["scope_sql"]["owner_message_count_after"] = len(rows)
        proof["scope_sql"]["new_messages"] = [
            dict(row) for row in rows if row["id"] > proof["scope_sql"]["owner_message_max_id_before"]
        ]
        row = await connection.fetchrow(
            "SELECT state,body,observed_at FROM memory_sources WHERE source_key=$1", source_key
        )
        proof["race_source_after"] = dict(row) if row else None
        proof["race_source_terms"] = await connection.fetchval(
            "SELECT count(*) FROM memory_source_terms WHERE source_key=$1", source_key
        )
        proof["race_source_links"] = await connection.fetchval(
            "SELECT count(*) FROM memory_source_links WHERE source_key=$1", source_key
        )
    finally:
        await connection.close()
    source_connection = await asyncpg.connect(
        user="boot", database=source_database, host=str(root / "socket"), port=25433
    )
    try:
        snapshots = {}
        for table in ["memory_sources", "memory_source_terms", "character_memories", "messages"]:
            rows = await source_connection.fetch("SELECT * FROM " + table)
            snapshots[table] = hashlib.sha256(
                json.dumps(
                    sorted([dict(row) for row in rows], key=lambda row: json.dumps(row, sort_keys=True, default=str)),
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
        proof["scope_sql"]["source_database_snapshot_after"] = snapshots
    finally:
        await source_connection.close()
    capture_storage_proof(proof, output)
    proof["cloud_summary"] = [
        dict(
            http_status=c["http_status"], model=c["response"].get("model"), output_budget=c["request"].get("max_tokens")
        )
        for c in cloud_calls
    ]
    proof["configured_model"] = os.environ["OPENAI_COMPAT_MODEL"]
    proof["configured_context_window"] = int(os.environ["OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS"])
    proof["scope_guard_checks"] = dict(
        normal_auth=proof["auth_statuses"] == [200, 200],
        same_authenticated_owner=proof["scope_owner_identity"] == proof["scope_current_identity"],
        fresh_id_initially_absent=proof["source_initially_new"],
        two_complete_different_statements=race_cases[0]["message"] != race_cases[1]["message"]
        and all(len(case["message"]) > 140 for case in race_cases),
        distinct_queue_sessions=proof["race_requests"][0]["sessionId"] != proof["race_requests"][1]["sessionId"],
        one_scoped_recorded_source=proof["race_source_after"]["state"] == "recorded"
        and proof["race_source_after"]["body"] in [c["message"] for c in race_cases],
        original_sources_unchanged=all(
            row in proof["scope_owner_sources_after"] for row in proof["scope_owner_sources_before"]
        ),
        unrelated_original_claims_unchanged=all(
            row in proof["scope_owner_claims_after"] for row in proof["scope_owner_claims_before"]
        ),
        old_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        source_fixture_db_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        no_old_writing_answers_imports_search=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
        large_model_only=bool(cloud_calls)
        and all(c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        source_recall_kept=proof["controlled_ablation"]["raw_source_recall_enabled"] is True,
        no_history_ablation=proof["cold_history_diagnostics"] == [],
    )
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    proof["checks"] = {**proof["scope_guard_checks"], **evaluate_race_feedback(proof, cloud_calls)}
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            dict(
                result=str(output / "result.json"),
                native_statuses=[g["http_status"] for g in proof["generation"]],
                cloud_calls=len(cloud_calls),
                scope_guard_checks=proof["scope_guard_checks"],
            ),
            ensure_ascii=False,
        )
    )

    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]


def evaluate_race_feedback(proof, calls):
    """A successful verification requires refusal before the losing model call."""
    from html import unescape

    accepted = [g for g in proof["generation"] if g["http_status"] == 200]
    rejected = [g for g in proof["generation"] if g["http_status"] == 409]
    if len(accepted) != 1 or len(rejected) != 1:
        return {"exactly_one_accepted_and_one_rejected": False}
    requests = {r["case_id"]: r for r in proof["race_requests"]}
    winner = requests[accepted[0]["id"]]
    loser = requests[rejected[0]["id"]]
    wire = "\n".join(unescape(m["content"]) for call in calls for m in call["request"]["messages"])
    answers = [call for call in calls if call["request"].get("max_tokens") == 1024]
    receipts = proof["memory_status"]["recent_results"]
    new_messages = proof["scope_sql"]["new_messages"]
    return dict(
        exactly_one_accepted_and_one_rejected=True,
        conflict_explicit=rejected[0]["response"].get("detail", {}).get("code") == "source_identity_conflict",
        only_winner_prepared=len(proof["prepared_diagnostics"]) == len(proof["generation_diagnostics"]) == 1
        and proof["prepared_diagnostics"][0]["query"] == winner["message"],
        only_one_complete_winner_answer=len(answers) == 1
        and winner["message"] in unescape(answers[0]["request"]["messages"][-1]["content"]),
        rejected_statement_never_sent_to_model=loser["message"] not in wire,
        one_committed_source_receipt=len(receipts) == 1
        and receipts[0].get("source_capture") == "recorded"
        and receipts[0].get("source_message_id") == winner["sourceMessageId"],
        stored_source_is_winner=proof["race_source_after"]["body"] == winner["message"],
        only_winner_archived=len(new_messages) == 1
        and new_messages[0]["message"] == winner["message"]
        and new_messages[0]["sourceMessageId"] == winner["sourceMessageId"],
    )
