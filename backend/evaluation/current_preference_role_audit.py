"""Audit recipient routing using real writer records and native requests, without replay."""

import argparse
import asyncio
import json
import os
import re
import secrets
import sys
import types
from dataclasses import asdict
from html import unescape
from pathlib import Path


async def run(args):
    phase = Path(args.phase).resolve()
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    cluster = runtime / "evaluations/r148pg.s3"
    assert phase.parent == runtime / "backups/backend-chain-20261001" and re.fullmatch(r"stage[1-9]\d*", phase.name)
    os.environ.update(
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        DATABASE_PATH=str(phase / "audit-import-only.sqlite"),
    )
    from datetime import datetime

    import asyncpg

    from character.memory_service import _detect_memory_intents, _is_current_record, _is_suppressed
    from character.temporal_projection import project_temporal_record

    old = types.ModuleType("stage_original_memory_service")
    sys.modules[old.__name__] = old
    exec(
        compile(
            (phase / "original-backend_character_memory_service.py").read_text(),
            str(phase / "original-memory-service.py"),
            "exec",
        ),
        old.__dict__,
    )
    fixture = json.loads((phase / "fixture.json").read_text())
    baseline = json.loads((phase / "native-pg-baseline/result.json").read_text())
    fixed = json.loads((phase / "native-pg-fixed/result.json").read_text())
    before_calls = json.loads((phase / "native-pg-baseline/cloud-calls.json").read_text())
    after_calls = json.loads((phase / "native-pg-fixed/cloud-calls.json").read_text())
    calls = before_calls + after_calls
    admission = json.loads((phase / "native-pg-baseline/writer-admission.json").read_text())
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check(
        "fully_informative_three_complete_native_assertions",
        fixture["synthetic"] and len(fixture["seed_messages"]) == 3 and fixture["no_fabricated_memory_seed"],
    )
    check(
        "same_unchanged_query",
        baseline["seed_turns"][0]["source_message"] == fixture["seed_messages"][0]
        and fixed["seed_turns"] == baseline["seed_turns"],
    )
    check(
        "two_native_ordinary_same_owners",
        baseline["chat_auth_statuses"] == fixed["chat_auth_statuses"] == [200, 200]
        and baseline["seed_scope"] == fixed["seed_scope"]
        and baseline["seed_scope"]["owner"] != "1",
    )
    check("baseline14_fixed4_actual_calls", len(before_calls) == 14 and len(after_calls) == 4)
    check(
        "all_requests_and_responses_pro",
        all(c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro" for c in calls),
    )
    check(
        "all_actual_calls200_stop",
        all(c["http_status"] == 200 and c["response"]["choices"][0]["finish_reason"] == "stop" for c in calls),
    )
    check(
        "baseline4_fixed1_actual_primary",
        baseline["http_status"] == fixed["http_status"] == 200
        and baseline["primary_calls"] == 4
        and fixed["primary_calls"] == 1,
    )
    check(
        "successfully_written_seeds_not_replayed",
        fixed["seed_cloud_calls_replayed"] == 0
        and fixed["calls_before_question"] == 0
        and fixed["seed_template_verified"]
        and fixed["seed_original_claim_id_content_verified"],
    )
    check(
        "three_original_real_writer_admissions",
        len(admission) >= 3
        and all(
            admission[i]["input"]["source_message"] == fixture["seed_messages"][i]
            and len(admission[i]["accepted"]) == 1
            for i in range(3)
        ),
    )
    check(
        "raw_writer_not_manufactured",
        all(
            json.loads(admission[i]["raw_response"])["memories"][0]["evidence"] in fixture["seed_messages"][i]
            for i in range(3)
        ),
    )
    check(
        "conditional_raw_admitted_and_stored",
        dict(admission[2]["accepted"][0]["qualifiers"]) == {"condition": "吃过早餐且当天没有胃痛"}
        and baseline["seed_records"][0]["metadata"]["qualifiers"] == {"condition": "吃过早餐且当天没有胃痛"},
    )
    check(
        "original_scheduler_three_commits",
        baseline["scheduler_before_question"]["saved"] == 3 and baseline["scheduler_before_question"]["failed"] == 0,
    )
    check(
        "same_three_claims_before_and_after",
        baseline["seed_records"] == fixed["seed_records"] == fixed["user_fact_records_after"],
    )
    stored = []
    for proof in [baseline, fixed]:
        connection = await asyncpg.connect(
            user="boot", database=proof["database"], host=str(cluster / "socket"), port=25433
        )
        try:
            async with connection.transaction(readonly=True):
                assert await connection.fetchval("SHOW data_directory") == str(cluster / "data")
                owner = proof["seed_scope"]["owner"]
                owner_key = json.dumps(("web", "web-character", owner))
                scope_key = json.dumps(("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner))
                rows = await connection.fetch(
                    "SELECT * FROM character_memories WHERE character_id=$1 AND platform=$2 AND adapter=$3 AND sender_id=$4 AND conversation_type=$5 AND conversation_id=$6",
                    "tsukiyashiro_kisaki",
                    "web",
                    "web-character",
                    owner,
                    "private",
                    owner,
                )
                sources = await connection.fetch(
                    "SELECT * FROM memory_sources WHERE owner_key=$1 AND scope_key=$2", owner_key, scope_key
                )
                links = await connection.fetch(
                    "SELECT l.memory_id,s.source_message_id,s.body,s.owner_key,s.scope_key,s.state FROM memory_source_links l JOIN memory_sources s ON l.source_key=s.source_key WHERE l.memory_id=ANY($1::bigint[])",
                    [row["id"] for row in rows],
                )
                stored.append(
                    {
                        "database": proof["database"],
                        "data_directory": str(cluster / "data"),
                        "owner_key": owner_key,
                        "scope_key": scope_key,
                        "rows": [dict(row) for row in rows],
                        "sources": [dict(row) for row in sources],
                        "links": [dict(row) for row in links],
                    }
                )
        finally:
            await connection.close()
    check(
        "independent_same_actual_three_durable_claims",
        len(stored[0]["rows"]) == len(stored[1]["rows"]) == 3
        and sorted(stored[0]["rows"], key=lambda r: r["id"]) == sorted(stored[1]["rows"], key=lambda r: r["id"]),
    )
    check(
        "three_exact_own_current_source_links",
        all(
            len(s["links"]) == 3
            and all(
                link["owner_key"] == s["owner_key"]
                and link["scope_key"] == s["scope_key"]
                and link["state"] == "recorded"
                for link in s["links"]
            )
            for s in stored
        ),
    )
    check(
        "independent_full_seed_bodies_unchanged",
        all({link["body"] for link in s["links"]} == set(fixture["seed_messages"]) for s in stored),
    )
    check(
        "no_new_fact_from_repeated_query",
        len(fixed["user_fact_records_after"]) == 3 and fixed["scheduler_final"]["saved"] == 0,
    )
    check(
        "source_grants_read_exact_pairs",
        baseline["prepared"][-1]["recall"]["source_receipt_pairs_returned"]
        == fixed["prepared"][-1]["recall"]["source_receipt_pairs_returned"]
        == 3,
    )
    old_intents = old._detect_memory_intents(fixture["question"])
    new_intents = _detect_memory_intents(fixture["question"])
    check("old_recipient_false_other_suppression", old_intents.suppress_preference and not old_intents.preference)
    check("new_recipient_unknown_not_claimed_self", not new_intents.suppress_preference and not new_intents.preference)
    check(
        "actual_baseline_filters_all_three",
        baseline["prepared"][-1]["recall"]["records_read"] == 3
        and baseline["prepared"][-1]["recall"]["usable_records"] == 0
        and baseline["prepared"][-1]["used_memory_ids"] == [],
    )
    by_claim = {
        str(r["id"]): {link["source_message_id"]: link for link in stored[0]["links"] if link["memory_id"] == r["id"]}
        for r in baseline["seed_records"]
    }
    views = [project_temporal_record(row, by_claim[str(row["id"])]) for row in baseline["seed_records"]]
    now = datetime.fromisoformat(baseline["prepared"][-1]["received_at"])
    check(
        "same_original_rows_time_eligible_before_fix",
        all(_is_current_record(row, now, include_pending=False) for row in views),
    )
    check(
        "suppression_causal_view_all_three",
        all(old._is_suppressed(row, old_intents) and not _is_suppressed(row, new_intents) for row in views),
    )
    prep = fixed["prepared"][-1]
    actual_ids = {str(row["id"]) for row in fixed["seed_records"]}
    check(
        "fixed_three_actual_usable_selected",
        prep["recall"]["usable_records"] == prep["recall"]["eligible_records"] == prep["recall"]["selected_count"] == 3
        and set(prep["used_memory_ids"]) == actual_ids,
    )
    check(
        "temporal_modes_not_promoted",
        prep["recall"]["temporal_views"]
        == baseline["prepared"][-1]["recall"]["temporal_views"]
        == {"fact": 0, "asserted_state": 2, "observation": 1},
    )
    check(
        "three_whole_compiled_packets",
        len(prep["memory_packets"]) == 3 and {row["memory_id"] for row in prep["memory_packets"]} == actual_ids,
    )
    conditional = next(row for row in prep["memory_packets"] if dict(row["qualifiers"]))
    check(
        "condition_remains_observation_with_complete_original",
        conditional["temporal_mode"] == "observation"
        and fixture["seed_messages"][2] in conditional["evidence"]
        and dict(conditional["qualifiers"]) == {"condition": "吃过早餐且当天没有胃痛"},
    )
    primary = [c for c in after_calls if any("<user_query>" in m.get("content", "") for m in c["request"]["messages"])][
        0
    ]
    wire = unescape(primary["request"]["messages"][-1]["content"])
    check("original_query_complete_on_real_wire", fixture["question"] in wire)
    check(
        "both_fact_and_conditional_evidence_in_real_wire",
        all(term in wire for term in ["桂花乌龙茶", "高糖可可饮料", "吃过早餐且当天没有胃痛", "低咖啡因咖啡"]),
    )
    check(
        "source_not_in_system_message",
        all(
            not any(message in m["content"] for message in fixture["seed_messages"])
            for m in primary["request"]["messages"]
            if m["role"] == "system"
        ),
    )
    selector = []
    for c in after_calls:
        try:
            payload = json.loads(c["request"]["messages"][-1]["content"])
        except (ValueError, KeyError):
            continue
        if "required_ids" in payload:
            selector.append((c, payload))
    check(
        "real_pro_selector_covers_all_three",
        len(selector) == 1
        and set(selector[0][1]["required_ids"]) == actual_ids
        and {
            d["id"]
            for d in json.loads(selector[0][0]["response"]["choices"][0]["message"]["content"])["decisions"]
            if d["label"] == "use"
        }
        == actual_ids,
    )
    reply = fixed["response"]["reply"]
    raw = primary["response"]["choices"][0]["message"]["content"]
    check("raw_reply_equals_api", reply == raw)
    check(
        "all_preference_values_and_both_conditions_in_answer",
        all(
            word in reply
            for word in ["桂花乌龙茶", "低咖啡因咖啡", "高糖可可饮料", "吃过早餐", "当天没有胃痛", "不喜欢"]
        ),
    )
    check("necessary_conjunction_not_sufficient", "且" in reply and "也不能推断你一定会喝" in reply)
    check("pending_zero_at_both_probe_saves", baseline["sync_pending_final"] == fixed["sync_pending_final"] == 0)
    tests = json.loads((phase / "test-accounting.json").read_text())
    check(
        "targeted_tests_honest_failure_and_repair",
        tests
        == {
            "new_unique": 25,
            "old_affected_unique": 1,
            "unique": 26,
            "executions": 35,
            "successful_executions": 34,
            "initial_failed_executions": 1,
            "repair_executions": 11,
            "full_suite": False,
            "initial_failure_retained": True,
        },
    )
    result = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "phase_calls": 18,
        "baseline_calls": 14,
        "fixed_calls": 4,
        "primary_calls": 5,
        "qualified_complete_chain_lookup_replies": 1,
        "runtime_files_changed": 1,
        "test_accounting": tests,
        "old_intents": asdict(old_intents),
        "new_intents": asdict(new_intents),
        "raw_baseline_reply": baseline["response"]["reply"],
        "raw_fixed_reply": reply,
        "baseline_claim_candidate_ids": [],
        "fixed_claim_candidate_ids": prep["used_memory_ids"],
        "temporal_views": prep["recall"]["temporal_views"],
        "strict_stage71_failure_still_open": True,
        "limits": [
            "Only closed self-report relative clauses defer recipient ownership; no inferred self ownership and unknown grammar retains prior behavior.",
            "Native fixed query still includes actual recent history and original source references; final reply correctness does not establish fact-only/distant-history sufficiency.",
            "Conditional claim remains observation; no unqualified promotion or calendar expiry invention.",
        ],
    }
    (phase / "storage-read.json").write_text(json.dumps(stored, ensure_ascii=False, indent=2, default=str) + chr(10))
    (phase / "native-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(
        json.dumps(
            {
                "checks": str(result["passed"]) + "/" + str(result["total"]),
                "phase_calls": 18,
                "failed": [k for k, v in checks.items() if not v],
            },
            ensure_ascii=False,
        )
    )
    assert all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    asyncio.run(run(parser.parse_args()))
