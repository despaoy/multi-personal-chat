"""Audit saved complete-source requests and independently read the test database."""

import argparse
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

checks = {}


def check(name, condition):
    checks[name] = bool(condition)
    assert checks[name], name


def original_source_answer_fidelity(answer: str, bodies: list[str]) -> dict:
    """Compare exact bodies, allowing only explicit Markdown line separators.

    Two spaces before a newline are Markdown presentation, not source text.
    No stripping of source whitespace, prefixes, values or punctuation occurs.
    """
    separator = r"(?:\n\n|\n| {2}\n)"
    pattern = separator.join(re.escape(body) for body in bodies)
    return {
        "exact_bodies_in_order_no_content_additions": bool(bodies) and re.fullmatch(pattern, answer) is not None,
        "byte_identical_to_blank_line_join": bool(bodies) and answer == "\n\n".join(bodies),
        "markdown_hard_line_breaks": answer.count("  \n"),
        "expected_body_count": len(bodies),
    }


async def main():
    from evaluation.complete_source_role_probe import required_fixture_sources

    r = Path("/home/boot/lhm/multipersonal-runtime")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default=str(r / "backups/backend-chain-20261001/stage68"))
    parser.add_argument("--variant", default="native-pg-fixed")
    parser.add_argument("--test-new-count", type=int, default=10)
    parser.add_argument("--test-affected-count", type=int, default=4)
    parser.add_argument("--test-executions", type=int)
    parser.add_argument(
        "--record-answer-contract-failure",
        action="store_true",
        help="Retain a failed model answer contract; never qualify that answer.",
    )
    args = parser.parse_args()
    unique_tests = args.test_new_count + args.test_affected_count
    test_executions = args.test_executions if args.test_executions is not None else unique_tests
    assert args.test_new_count >= 0 and args.test_affected_count >= 0 and test_executions >= unique_tests
    b = Path(args.phase).resolve()
    assert b.parent == r / "backups/backend-chain-20261001" and re.fullmatch(r"stage[1-9]\d*", b.name)
    assert re.fullmatch(r"native-pg(?:-[a-z]{1,12})*", args.variant)
    fixture = json.loads((b / "fixture.json").read_text())
    required_sources = required_fixture_sources(fixture)
    old = json.loads((b / "baseline.json").read_text())
    native = json.loads((b / args.variant / "result.json").read_text())
    baseline_native = json.loads((b / "native-pg-baseline/result.json").read_text())
    calls = json.loads((b / args.variant / "cloud-calls.json").read_text())
    baseline_calls = json.loads((b / "native-pg-baseline/cloud-calls.json").read_text())
    os.environ.update(
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        DATABASE_PATH=str(b / "retrieval-baseline/database.sqlite"),
        DATABASE_URL="",
        USE_POSTGRESQL="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    import asyncpg

    from character.models import CompiledCharacterContext, UserScope
    from character.source_memory import SourceMemoryService, attach_sources
    from db.database import SQLiteDB
    from inference.generation_request import GenerationRequest, build_generation_request
    from inference.lora_registry import get_lora_system_prompt
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    db = SQLiteDB(b / "retrieval-baseline/database.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", b.name + "-memory", "synthetic-owner", "synthetic-owner", "private")

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
    found = await repo.search_sources("tsukiyashiro_kisaki", scope, query=fixture["question"], limit=None)

    def authority(rows):
        return {(row["source_message_id"], row["body"], row["observed_at"]) for row in rows}

    original_needed = [
        row for row in old["found"] if row["source_message_id"] in {item["id"] for item in required_sources}
    ]
    check("fixed_sqlite_required_original_authority_unchanged", authority(found) == authority(original_needed))
    sources = await SourceMemoryService(repo, max_chars=32768, defer_budget=True).recall(
        "tsukiyashiro_kisaki", scope, fixture["question"]
    )
    packet = json.loads(sources.context)
    check(
        "fixed_sqlite_keeps_all_original_sources",
        len(packet["records"]) == len(required_sources)
        and {(row["source_id"], row["text"]) for row in packet["records"]}
        == {(s["id"], s["body"]) for s in required_sources},
    )
    check(
        "fixed_source_selection_omits_none",
        sources.diagnostics["selection_omitted"] == sources.diagnostics["fresh_recheck_omitted"] == 0,
    )
    check(
        "sources_remain_unresolved_historical_speech",
        packet["speaker_role"] == "user"
        and packet["described_subject"] == packet["current_validity"] == "not_resolved",
    )
    context = attach_sources(CompiledCharacterContext("角色", "", "", memory_status="no_match"), sources)
    plan = build_generation_request(
        GenerationRequest(
            message=fixture["question"],
            persona_prompt=get_lora_system_prompt("kisaki"),
            character_context=context,
            context_window_tokens=65536,
            max_tokens=2048,
        )
    )
    wire = unescape(plan.messages[-1]["content"])
    check("same_original_sqlite_sources_reach_canonical_input", all(s["body"] in wire for s in required_sources))
    check(
        "source_only_creates_no_fact_packets",
        not context.memory_packets
        and not await repo.list_memory_records("tsukiyashiro_kisaki", scope, limit=None, include_inactive=True),
    )
    other = await SourceMemoryService(repo, max_chars=32768, defer_budget=True).recall(
        "tsukiyashiro_kisaki", UserScope("web", b.name + "-memory", "other", "other", "private"), fixture["question"]
    )
    check("fixed_cross_owner_rejected", not other.context and not other.candidate_context)
    check("same_sqlite_storage_four_tables_unchanged", before == digest())
    (b / "fixed.json").write_text(
        json.dumps(
            {
                "cloud_calls": 0,
                "sources": asdict(sources),
                "canonical_messages": plan.messages,
                "stored_claim_count": 0,
                "storage_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    check("original_complete_fixture_preserved", fixture == old["fixture"])
    check(
        "baseline_native_has_original_durable_sources",
        baseline_native["durable_seed_verified_before_question"]
        and baseline_native["durable_declared_source_times_verified"]
        and baseline_native["seed_sources"] == fixture["sources"],
    )
    original_source_trace = baseline_native["prepared"][0]["recall"]["sources"]
    check(
        "baseline_native_reproduces_original_candidate_cut",
        original_source_trace["indexed_read_count"] == old["recall"]["diagnostics"]["indexed_read_count"]
        and len(original_source_trace["selected_ids"]) == len(old["recall"]["diagnostics"]["selected_ids"])
        and original_source_trace["selection_omitted"] == old["recall"]["diagnostics"]["selection_omitted"],
    )
    check(
        "baseline_primary_blocked_not_model_failure",
        baseline_native["primary_calls"] == 0 and baseline_native["http_status"] == 500,
    )
    check(
        "baseline_auxiliary_calls_retained",
        len(baseline_calls) == 2
        and all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in baseline_calls
        ),
    )
    check(
        "native_authentication_and_ordinary_owner",
        native["auth_statuses"] == native["chat_auth_statuses"] == [200, 200] and native["seed_scope"]["owner"] != "1",
    )
    check("native_synthetic_pg", native["database_mode"] == "PostgreSQL" and native["synthetic_only"])
    check("native_original_sources_and_observed_times", native["seed_sources"] == fixture["sources"])
    check(
        "native_durable_sources_before_question",
        native["durable_seed_verified_before_question"] and native["durable_declared_source_times_verified"],
    )
    check("native_successful_primary", native["http_status"] == 200 and native["primary_calls"] == 1)
    check(
        "native_four_completed_pro_calls",
        len(calls) == 4
        and all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
    )
    check(
        "native_no_fact_promotion",
        native["seed_records"] == native["user_fact_records_after"] == []
        and all(not p["memory_packets"] for p in native["prepared"]),
    )
    prepared = native["prepared"][0]
    trace = prepared["recall"]["sources"]
    check("native_all_original_source_ids", set(trace["selected_ids"]) == {s["id"] for s in required_sources})
    check(
        "native_no_source_omission_or_pending",
        trace["selection_omitted"] == trace["fresh_recheck_omitted"] == 0 and not prepared["source_candidate_context"],
    )
    check(
        "native_complete_episodic_packet",
        {(row["source_id"], row["text"]) for row in json.loads(prepared["episodic_context"])["records"]}
        == {(s["id"], s["body"]) for s in required_sources},
    )
    primary = [
        c
        for c in calls
        if c["request"].get("max_tokens") == native["primary_output_tokens"]
        and any("<user_query>" in m["content"] for m in c["request"]["messages"])
    ]
    check("actual_primary_unique", len(primary) == 1)
    wire = unescape(primary[0]["request"]["messages"][-1]["content"])
    match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    check("actual_primary_original_whole_question", fixture["question"] in wire)
    check(
        "actual_primary_all_whole_sources",
        bool(match)
        and {(row["source_id"], row["text"]) for row in json.loads(match[1])["records"]}
        == {(s["id"], s["body"]) for s in required_sources},
    )
    check(
        "actual_sources_not_in_system",
        all(
            s["body"] not in m["content"]
            for s in fixture["sources"]
            for m in primary[0]["request"]["messages"]
            if m["role"] == "system"
        ),
    )
    cluster = r / "evaluations/r148pg.s3"
    owner = native["seed_scope"]["owner"]
    conn = await asyncpg.connect(user="boot", database=native["database"], host=str(cluster / "socket"), port=25433)
    try:
        async with conn.transaction(readonly=True):
            check("independent_exact_cluster", await conn.fetchval("SHOW data_directory") == str(cluster / "data"))
            check("independent_exact_database", await conn.fetchval("SELECT current_database()") == native["database"])
            check(
                "independent_actual_owner",
                str(await conn.fetchval("SELECT id FROM users WHERE id=$1", int(owner))) == owner and owner != "1",
            )
            owner_key = json.dumps(("web", "web-character", owner))
            scope_key = json.dumps(("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner))
            rows = await conn.fetch(
                "SELECT source_message_id,body,observed_at,state FROM memory_sources WHERE owner_key=$1 AND scope_key=$2",
                owner_key,
                scope_key,
            )
            check(
                "independent_seed_sources_match_exact_body_and_time",
                all(
                    any(
                        row["source_message_id"] == s["id"]
                        and row["body"] == s["body"]
                        and row["state"] == "recorded"
                        and datetime.fromisoformat(row["observed_at"]) == datetime.fromisoformat(s["observed_at"])
                        for row in rows
                    )
                    for s in fixture["sources"]
                ),
            )
            check(
                "independent_question_receipt_one",
                sum(row["body"] == fixture["question"] and row["state"] == "recorded" for row in rows) == 1,
            )
            check("independent_only_original_seeds_plus_question", len(rows) == len(fixture["sources"]) + 1)
            check(
                "independent_no_scoped_user_facts",
                await conn.fetchval(
                    "SELECT count(*) FROM character_memories WHERE character_id=$1 AND platform=$2 AND adapter=$3 AND sender_id=$4 AND conversation_id=$5 AND conversation_type=$6",
                    "tsukiyashiro_kisaki",
                    "web",
                    "web-character",
                    owner,
                    owner,
                    "private",
                )
                == 0,
            )
            source_keys = await conn.fetch(
                "SELECT source_key FROM memory_sources WHERE owner_key=$1 AND scope_key=$2", owner_key, scope_key
            )
            check(
                "independent_no_claim_links_for_source_only",
                await conn.fetchval(
                    "SELECT count(*) FROM memory_source_links WHERE source_key=ANY($1::text[])",
                    [row["source_key"] for row in source_keys],
                )
                == 0,
            )
            (b / "storage-read.json").write_text(
                json.dumps(
                    {
                        "database": native["database"],
                        "owner": owner,
                        "sources": [dict(row) for row in rows],
                        "readonly": True,
                        "scoped_claim_count": 0,
                        "seed_source_link_count": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
    finally:
        await conn.close()
    check("pending_writes_zero", native["sync_pending_final"] == 0)
    answer = native["response"]["reply"]
    check("raw_primary_matches_api_answer", answer == primary[0]["response"]["choices"][0]["message"]["content"])
    answer_fidelity = original_source_answer_fidelity(answer, [s["body"] for s in required_sources])
    strict_answer_passed = answer_fidelity["exact_bodies_in_order_no_content_additions"]
    if args.record_answer_contract_failure and not strict_answer_passed:
        checks["answer_exact_original_bodies_in_order_no_content_additions"] = False
    else:
        check("answer_exact_original_bodies_in_order_no_content_additions", strict_answer_passed)
    # Count every completed variant, including failed primary answers.
    phase_calls = []
    for path in sorted(b.glob("native-pg*/cloud-calls.json")):
        phase_calls.extend(json.loads(path.read_text()))
    check(
        "all_phase_calls_request_response_pro",
        all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in phase_calls
        ),
    )
    actual_primary_count = sum(
        c["request"].get("max_tokens") == native["primary_output_tokens"]
        and any("<user_query>" in m["content"] for m in c["request"]["messages"])
        for c in phase_calls
    )
    if args.variant != "native-pg-fixed":
        first = json.loads((b / "native-pg-fixed/result.json").read_text())
        first_calls = json.loads((b / "native-pg-fixed/cloud-calls.json").read_text())
        first_primary = next(
            c
            for c in first_calls
            if c["request"].get("max_tokens") == first["primary_output_tokens"]
            and any("<user_query>" in m["content"] for m in c["request"]["messages"])
        )
        check(
            "intermediate_original_case_complete_input_but_zero_matching_answers",
            first["seed_sources"] == fixture["sources"]
            and first["primary_calls"] == 1
            and len(first["prepared"][0]["recall"]["sources"]["selected_ids"]) == len(fixture["sources"])
            and not any(s["body"] in first["response"]["reply"] for s in fixture["sources"]),
        )
        check(
            "intermediate_refusal_is_raw_model_not_postprocessing",
            first["response"]["reply"] == first_primary["response"]["choices"][0]["message"]["content"],
        )
        from inference.generation_request import SOURCE_SPEECH_PROVENANCE_POLICY

        final_system = chr(10).join(m["content"] for m in primary[0]["request"]["messages"] if m["role"] == "system")
        first_system = chr(10).join(m["content"] for m in first_primary["request"]["messages"] if m["role"] == "system")
        check("actual_final_source_receipt_policy_present", SOURCE_SPEECH_PROVENANCE_POLICY in final_system)
        check("intermediate_source_receipt_policy_was_absent", SOURCE_SPEECH_PROVENANCE_POLICY not in first_system)
    if "required_source_ids" in fixture and "all_match_packet_token_estimate" in old:
        from db.memory_source_search import literal_project_source_terms

        check("task_sources_distinct_from_fully_stored_distractors", len(required_sources) < len(fixture["sources"]))
        check(
            "baseline_noise_exhausts_provider_budget_before_primary",
            old["all_match_packet_token_estimate"] > 65536
            and old["actual_final_source_status"] == "budget_omitted"
            and old["actual_target_ids_in_wire"] == [],
        )
        check(
            "required_originals_and_expected_output_fit_actual_budgets",
            old["needed_packet_chars"] < 16384
            and old["needed_canonical_estimated_total"] < 65536
            and old["expected_answer_estimate"] < native["primary_output_tokens"],
        )
        check(
            "actual_explicit_project_source_scope",
            trace["literal_project_terms"] == list(literal_project_source_terms(fixture["question"]))
            and trace["literal_project_terms"]
            and trace["indexed_read_count"] == len(required_sources),
        )
        proof = json.loads((b / "query-equivalence-readonly.json").read_text())
        check(
            "sql_membership_rewrite_preserves_rows_scores_order_in_both_databases",
            proof["readonly"]
            and proof["sqlite_exact_rows_scores_order"]
            and proof["pg_exact_rows_scores_order"]
            and proof["sqlite_source_count"] == len(required_sources)
            and proof["pg_post_question_source_count"] == len(required_sources) + 1,
        )
        check(
            "query_equivalence_tied_to_current_source_hash",
            proof["final_source_sha256"]
            == hashlib.sha256(Path("backend/db/memory_source_search.py").read_bytes()).hexdigest()
            and proof["intermediate_source_sha256"]
            == hashlib.sha256((b / "intermediate-exists-memory_source_search.py").read_bytes()).hexdigest(),
        )
    elif fixture.get("case_kind") == "requested_visible_source_successor":
        relation = fixture["declared_order_relation"]
        check(
            "complete_requested_pair_and_output_fit_actual_budget",
            old["needed_packet_chars"] < 16384
            and old["needed_canonical_estimated_total"] < 65536
            and old["expected_answer_estimate"] < native["primary_output_tokens"],
        )
        check(
            "original_unlabeled_following_source_durable_and_window_readable",
            len(old["needed_sources"]) == len(required_sources)
            and any(row["source_message_id"] == relation["following_id"] for row in old["source_windows"][0]["rows"]),
        )
        check(
            "native_requested_order_and_complete_pair",
            trace["requested_source_order"] == "next_visible_after_each_anchor"
            and trace["effective_window_radius"] == 1
            and trace["requested_following"]
            == [{"anchor_id": relation["anchor_id"], "source_ids": [relation["following_id"]]}]
            and trace["following_missing_for_anchors"] == []
            and trace["following_anchor_dependency_omitted"] == 0,
        )
        check(
            "record_order_is_not_semantic_project_or_fact_inference",
            trace["window_semantic_relation"] == "not_inferred",
        )
        ordered = sorted(rows, key=lambda row: (row["observed_at"], row["source_message_id"]))
        positions = [i for i, row in enumerate(ordered) if row["source_message_id"] == relation["anchor_id"]]
        check(
            "independent_current_scoped_source_order_matches_requested_pair",
            len(positions) == 1
            and ordered[positions[0] + 1]["source_message_id"] == relation["following_id"]
            and len({row["observed_at"] for row in rows}) == len(rows),
        )
        proof = json.loads((b / "window-plan-equivalence.json").read_text())
        check(
            "existing_window_queries_bindings_and_rows_unchanged",
            proof["sql_and_bindings_identical"]
            and proof["legacy_rows_identical"]
            and proof["following_source_ids"] == [relation["following_id"]],
        )
        check(
            "direction_metadata_proof_tied_to_current_source_hash",
            proof["source_sha256"]
            == hashlib.sha256(Path("backend/db/memory_source_window.py").read_bytes()).hexdigest(),
        )
        cursor = 0
        ordered_body_offsets = []
        for item in required_sources:
            position = answer.find(item["body"], cursor)
            ordered_body_offsets.append(position)
            cursor = position + len(item["body"]) if position >= 0 else cursor
        check(
            "model_retains_every_full_original_body_in_requested_order",
            all(position >= 0 for position in ordered_body_offsets),
        )
    answer_checks = {s["id"]: s["body"] in answer for s in required_sources}
    assert all(answer_checks.values())
    audit = {
        "checks": checks,
        "chain_passed": sum(checks.values()),
        "chain_total": len(checks),
        "answer_checks": answer_checks,
        "answer_passed": sum(answer_checks.values()),
        "answer_total": len(answer_checks),
        "answer": answer,
        "answer_fidelity": answer_fidelity,
        "qualified_cloud_calls": 4,
        "baseline_auxiliary_calls": 2,
        "phase_cloud_calls": len(phase_calls),
        "qualified_answers": int(strict_answer_passed),
        "strict_answer_contract_passed": strict_answer_passed,
        "answer_contract_failures": int(not strict_answer_passed),
        "backend_chain_passed": sum(
            value
            for key, value in checks.items()
            if key != "answer_exact_original_bodies_in_order_no_content_additions"
        ),
        "backend_chain_total": sum(
            key != "answer_exact_original_bodies_in_order_no_content_additions" for key in checks
        ),
        "phase_primary_calls": actual_primary_count,
        "targeted_new_cases": args.test_new_count,
        "targeted_affected_cases": args.test_affected_count,
        "targeted_unique_passed": args.test_new_count + args.test_affected_count,
        "targeted_executions": test_executions,
        "all_calls_pro": True,
        "no_full_suite": True,
        "no_passing_model_replay": True,
        "no_production_chat_writes": True,
    }
    (b / "native-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ["checks", "answer_checks", "answer"]}))


if __name__ == "__main__":
    asyncio.run(main())
