import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict
from datetime import datetime
from html import unescape
from pathlib import Path

b = Path("/home/boot/lhm/multipersonal-runtime/backups/backend-chain-20261001/stage67")
r = Path("/home/boot/lhm/multipersonal-runtime")
fixture = json.loads((b / "fixture.json").read_text())
old = json.loads((b / "baseline.json").read_text())
native = json.loads((b / "native-pg-fixed/result.json").read_text())
calls = json.loads((b / "native-pg-fixed/cloud-calls.json").read_text())
os.environ.update(
    ENVIRONMENT="production",
    JWT_SECRET=secrets.token_urlsafe(48),
    ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
    DATABASE_PATH=str(b / "retrieval-baseline/database.sqlite"),
    DATABASE_URL="",
    USE_POSTGRESQL="false",
    PYTHONDONTWRITEBYTECODE="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    EMBEDDING_MODEL_PATH=str(r / "models/paraphrase-multilingual-MiniLM-L12-v2"),
    OMP_NUM_THREADS="2",
)

checks = {}


def check(name, condition):
    checks[name] = bool(condition)
    assert checks[name], name


async def main():
    # Establish isolated production environment before importing DB modules.
    import asyncpg

    from character.context_builder import compile_reference_context
    from character.evidence_selector import ContextualEvidenceSelector
    from character.memory_service import CharacterMemoryService
    from character.models import UserScope
    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    db = SQLiteDB(b / "retrieval-baseline/database.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "stage67-memory", "synthetic-owner", "synthetic-owner", "private")

    def digest():
        conn = db._get_connection()
        return {
            table: hashlib.sha256(
                json.dumps(
                    [dict(row) for row in conn.execute("SELECT * FROM " + table)], sort_keys=True, default=str
                ).encode()
            ).hexdigest()
            for table in ["character_memories", "memory_sources", "memory_source_links", "memory_source_terms"]
        }

    before = digest()
    records = await repo.list_memory_records("tsukiyashiro_kisaki", scope, limit=None, include_inactive=True)
    check("original_sqlite_records_unchanged", records == old["stored_records"])
    candidates, count, trace = await CharacterMemoryService(repo).recall_with_diagnostics(
        "tsukiyashiro_kisaki",
        scope,
        fixture["question"],
        for_contextual_selection=True,
        reference_time=datetime.fromisoformat(fixture["reference_time"]),
    )

    async def use(messages):
        return json.dumps({"decisions": [{"id": item.memory_id, "label": "use"} for item in candidates]})

    selected = await ContextualEvidenceSelector(use).select(fixture["question"], candidates)
    stats = {}
    context, ids = compile_reference_context(selected.memories, complete_evidence=True, diagnostics=stats)
    check(
        "same_original_six_sqlite_candidates",
        json.loads(json.dumps([asdict(x) for x in candidates], default=str)) == old["recall"]["items"],
    )
    check("fixed_default_keeps_six", len(selected.memories) == len(ids) == 6)
    check("fixed_default_no_count_or_budget_skip", stats["count_skipped"] == stats["budget_skipped"] == 0)
    check("fixed_reference_has_all_original_source_bodies", all(s["body"] in context for s in fixture["sources"]))
    check("sqlite_reads_leave_tables_unchanged", before == digest())
    stranger = UserScope("web", "stage67-memory", "other-owner", "other-owner", "private")
    other, _, _ = await CharacterMemoryService(repo).recall_with_diagnostics(
        "tsukiyashiro_kisaki",
        stranger,
        fixture["question"],
        for_contextual_selection=True,
        reference_time=datetime.fromisoformat(fixture["reference_time"]),
    )
    check("fixed_cross_owner_rejected", not other)
    (b / "fixed.json").write_text(
        json.dumps(
            {
                "cloud_calls": 0,
                "records": records,
                "items": [asdict(x) for x in candidates],
                "selection": asdict(selected),
                "ids": ids,
                "context": context,
                "stats": stats,
                "storage_unchanged": before == digest(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n"
    )
    check(
        "native_authenticated_owner",
        native["auth_statuses"] == [200, 200]
        and native["chat_auth_statuses"] == [200, 200]
        and native["seed_scope"]["owner"] != "1",
    )
    check("native_postgresql", native["database_mode"] == "PostgreSQL" and native["synthetic_only"])
    check("native_original_sources_and_times", native["seed_sources"] == fixture["sources"])
    check(
        "native_durable_declared_times",
        native["durable_declared_intervals_verified"] and native["durable_seed_verified_before_question"],
    )
    check("native_successful_http_and_one_primary", native["http_status"] == 200 and native["primary_calls"] == 1)
    check(
        "all_four_calls_request_response_pro",
        len(calls) == 4
        and all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
    )
    prepared = native["prepared"][0]
    packets = prepared["memory_packets"]
    check("native_original_question_received", len(native["prepared"]) == 1 and len(packets) == 6)
    check("native_six_requested_fields", {x["memory_key"] for x in packets} == {s["field"] for s in fixture["sources"]})
    check(
        "native_six_source_ids",
        {key for x in packets for key in x["source_message_ids"]} == {s["id"] for s in fixture["sources"]},
    )
    check("native_all_active_nonhistorical", all(x["status"] == "active" and not x["historical"] for x in packets))
    check(
        "native_no_packet_skips",
        prepared["memory_budget"]["count_skipped"] == prepared["memory_budget"]["budget_skipped"] == 0,
    )
    check(
        "native_selector_selected_six",
        prepared["selection_status"] == "selected" and prepared["selection_candidate_count"] == 6,
    )
    primaries = [
        c
        for c in calls
        if c["request"].get("max_tokens") == 2048
        and any("<user_query>" in m["content"] for m in c["request"]["messages"])
    ]
    check("actual_primary_request_unique", len(primaries) == 1)
    wire = unescape(primaries[0]["request"]["messages"][-1]["content"])
    match = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
    check("actual_primary_full_original_question", fixture["question"] in wire)
    check(
        "actual_primary_all_six_memory_values",
        bool(match) and all(s["expected"] in match[1] for s in fixture["sources"]),
    )
    check(
        "no_source_bodies_in_system_prompt",
        all(
            s["body"] not in m["content"]
            for s in fixture["sources"]
            for m in primaries[0]["request"]["messages"]
            if m["role"] == "system"
        ),
    )
    owner = native["seed_scope"]["owner"]
    cluster = r / "evaluations/r148pg.s3"
    conn = await asyncpg.connect(user="boot", database=native["database"], host=str(cluster / "socket"), port=25433)
    try:
        async with conn.transaction(readonly=True):
            check("independent_exact_cluster", await conn.fetchval("SHOW data_directory") == str(cluster / "data"))
            rows = await conn.fetch(
                "SELECT id,status,content,memory_key,valid_from,valid_to FROM character_memories WHERE character_id=$1 AND platform=$2 AND adapter=$3 AND sender_id=$4 AND conversation_id=$5 AND conversation_type=$6",
                "tsukiyashiro_kisaki",
                "web",
                "web-character",
                owner,
                owner,
                "private",
            )
            check("independent_six_durable_claims", len(rows) == 6 and all(row["status"] == "active" for row in rows))
            check(
                "independent_original_values_and_intervals",
                all(
                    any(
                        row["memory_key"] == s["field"]
                        and s["expected"] in row["content"]
                        and datetime.fromisoformat(row["valid_from"]) == datetime.fromisoformat(s["observed_at"])
                        for row in rows
                    )
                    for s in fixture["sources"]
                ),
            )
            check(
                "claims_match_actual_native_packet_ids",
                {str(row["id"]) for row in rows} == {x["memory_id"] for x in packets},
            )
            check("question_did_not_change_seed_records", native["seed_records"] == native["profile_records_after"])
            # Independently read sources through scoped repository on the same PG DB.
            owner_key = json.dumps(("web", "web-character", owner))
            scope_key = json.dumps(("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner))
            source_rows = await conn.fetch(
                "SELECT source_message_id,body,observed_at FROM memory_sources WHERE owner_key=$1 AND scope_key=$2 AND state=$3",
                owner_key,
                scope_key,
                "recorded",
            )
            check(
                "independent_original_source_bodies_and_times",
                all(
                    any(
                        row["source_message_id"] == s["id"]
                        and row["body"] == s["body"]
                        and datetime.fromisoformat(row["observed_at"]) == datetime.fromisoformat(s["observed_at"])
                        for row in source_rows
                    )
                    for s in fixture["sources"]
                ),
            )
            link_rows = await conn.fetch(
                "SELECT l.memory_id,s.source_message_id FROM memory_source_links l JOIN memory_sources s ON s.source_key=l.source_key WHERE s.owner_key=$1 AND s.scope_key=$2 AND s.state=$3",
                owner_key,
                scope_key,
                "recorded",
            )
            expected_links = {(x["memory_id"], source) for x in packets for source in x["source_message_ids"]}
            check(
                "independent_exact_six_claim_source_links",
                {(str(x["memory_id"]), x["source_message_id"]) for x in link_rows} == expected_links
                and len(link_rows) == 6,
            )
            check(
                "independent_question_receipt_one",
                await conn.fetchval(
                    "SELECT count(*) FROM memory_sources WHERE owner_key=$1 AND scope_key=$2 AND state=$3 AND body=$4",
                    owner_key,
                    scope_key,
                    "recorded",
                    fixture["question"],
                )
                == 1,
            )
            check(
                "independent_actual_owner_exists",
                str(await conn.fetchval("SELECT id FROM users WHERE id=$1", int(owner))) == owner and owner != "1",
            )
            (b / "storage-read.json").write_text(
                json.dumps(
                    {
                        "database": native["database"],
                        "owner": owner,
                        "claims": [dict(x) for x in rows],
                        "sources": [dict(x) for x in source_rows],
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n"
            )
    finally:
        await conn.close()
    check("native_pending_writes_zero", native["sync_pending_final"] == 0)
    answer = native["response"]["reply"]
    answer_checks = {s["field"]: s["expected"] in answer for s in fixture["sources"]}
    check("complete_answer_matches_raw_primary", answer == primaries[0]["response"]["choices"][0]["message"]["content"])
    check("original_full_case_preserved", fixture == old["fixture"])
    baseline_native = json.loads((b / "native-baseline-verified.json").read_text())
    check(
        "baseline_model_six_use_but_chain_five",
        baseline_native["genuine_six_model_use_decisions"]
        and baseline_native["prepared_packets"] == 5
        and baseline_native["missing_fields"] == ["user_study_stage"],
    )
    check(
        "baseline_incomplete_primary_blocked",
        baseline_native["primary_calls"] == 0 and baseline_native["cloud_calls"] == 2,
    )
    audit = {
        "checks": checks,
        "chain_passed": sum(checks.values()),
        "chain_total": len(checks),
        "answer_checks": answer_checks,
        "answer_passed": sum(answer_checks.values()),
        "answer_total": len(answer_checks),
        "answer": answer,
        "qualified_cloud_calls": 4,
        "phase_cloud_calls": 6,
        "qualified_answers": 1,
        "phase_primary_calls": 1,
        "targeted_unique_passed": 17,
        "targeted_executions": 18,
        "old_stale_test_failures": 1,
        "all_call_models": "deepseek-v4-pro",
        "no_full_suite": True,
        "no_passing_model_case_replay": True,
        "no_production_chat_writes": True,
    }
    assert all(answer_checks.values())
    (b / "native-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k not in ["checks", "answer_checks", "answer"]}, ensure_ascii=False
        )
    )


asyncio.run(main())
