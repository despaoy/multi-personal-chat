"""Audit actual mixed-subject native evidence without replaying model calls."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path


async def run(args):
    phase = Path(args.phase).resolve()
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    cluster = runtime / "evaluations/r148pg.s3"
    assert phase.parent == runtime / "backups/backend-chain-20261001" and phase.name == "stage78"
    os.environ.update(
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        DATABASE_PATH=str(phase / "native-audit-import-only.sqlite"),
    )
    import asyncpg

    from inference.context_budget import estimated_tokens

    fixture = json.loads((phase / "fixture.json").read_text())
    variants = ["native-pg-resumed", "native-pg-fixed"]
    results = [json.loads((phase / name / "result.json").read_text()) for name in variants]
    baseline, fixed = results
    calls_by_run = [json.loads((phase / name / "cloud-calls.json").read_text()) for name in variants]
    original_calls = json.loads((phase / "native-pg-baseline/cloud-calls.json").read_text())
    calls = original_calls + calls_by_run[0] + calls_by_run[1]
    gates = [json.loads((phase / name / "primary-input-observed.json").read_text()) for name in variants]
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    check(
        "actual_native_authenticated_pg_routes",
        all(
            p["transport"] == "authenticated_ASGI" and p["database_mode"] == "PostgreSQL" and p["http_status"] == 200
            for p in results
        ),
    )
    check(
        "only_unfinished_native_steps_resumed",
        baseline["successful_bridge_turns_replayed"] == 0
        and baseline["seed_model_calls_replayed"] == 0
        and len(baseline["bridge_turns"]) == 9
        and [x["index"] for x in baseline["bridge_turns"]] == list(range(9))
        and all(x["status"] == 200 for x in baseline["bridge_turns"]),
    )
    check(
        "fixed_only_failed_memory_path_query_replayed",
        fixed["successful_bridge_turns_replayed"] == 0
        and fixed["seed_model_calls_replayed"] == 0
        and len(fixed["generation"]) == 1,
    )
    histories = [p["prepared"][-1]["history"] for p in results]
    check(
        "same_actual_recent_history_no_seed_or_previous_task_answer",
        histories[0] == histories[1]
        and len(histories[1]) == 16
        and all(
            fixture["source_message"] not in h["content"] and fixture["question"] not in h["content"]
            for h in histories[1]
        ),
    )
    check(
        "same_exact_three_real_pro_written_claims",
        baseline["seed_records"]
        == fixed["seed_records"]
        == baseline["user_fact_records_after"]
        == fixed["user_fact_records_after"]
        and len(fixed["seed_records"]) == 3
        and fixed["original_seed_claims_exact_verified"],
    )
    recalls = [p["prepared"][-1]["recall"] for p in results]
    check(
        "baseline_actual_all_three_wrongly_filtered",
        recalls[0]["records_read"] == 3
        and recalls[0]["usable_records"] == recalls[0]["selected_count"] == 0
        and recalls[0]["status"] == "all_filtered"
        and not baseline["prepared"][-1]["used_memory_ids"],
    )
    ids = {str(row["id"]) for row in fixed["seed_records"]}
    prepared = fixed["prepared"][-1]
    check(
        "fixed_same_three_usable_and_selected",
        recalls[1]["records_read"] == recalls[1]["usable_records"] == recalls[1]["selected_count"] == 3
        and set(prepared["used_memory_ids"]) == ids
        and {p["memory_id"] for p in prepared["memory_packets"]} == ids,
    )
    check(
        "temporal_observations_not_promoted_to_current_truth",
        recalls[0]["temporal_views"]
        == recalls[1]["temporal_views"]
        == {"fact": 0, "asserted_state": 0, "observation": 3}
        and all(p["temporal_mode"] == "observation" for p in prepared["memory_packets"]),
    )
    check(
        "full_private_original_negation_and_necessary_and_preserved",
        all(fixture["source_message"] in p["evidence"] for p in prepared["memory_packets"])
        and any(
            dict(p["qualifiers"]) == {"condition": "完整表格已提交且身份核验通过"} for p in prepared["memory_packets"]
        )
        and any("不喜欢代办受理" in p["content"] for p in prepared["memory_packets"]),
    )
    check(
        "all_four_public_and_private_full_originals_at_wire",
        all(g["private_source_present"] and all(g["document_bodies_present"].values()) for g in gates),
    )
    check(
        "all_actual_calls_preserved_including_prior_balance_failure",
        len(calls) == 62
        and [len(x) for x in calls_by_run] == [15, 5]
        and len(original_calls) == 42
        and sum(c["http_status"] == 200 for c in calls) == 58
        and sum(c["http_status"] == 402 for c in calls) == 4,
    )
    check(
        "all_actual_requests_and_successful_responses_pro",
        all(
            c["request"]["model"] == "deepseek-v4-pro"
            and (c["http_status"] != 200 or c["response"]["model"] == "deepseek-v4-pro")
            for c in calls
        ),
    )
    check(
        "resumed_and_fixed_calls_all_http200_stop",
        all(
            c["http_status"] == 200 and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in sum(calls_by_run, [])
        ),
    )
    primaries = [
        [
            c
            for c in run_calls
            if any(
                "<user_query>" in m.get("content", "") and fixture["question"] in unescape(m["content"])
                for m in c["request"].get("messages", [])
            )
        ]
        for run_calls in calls_by_run
    ]
    check(
        "actual_final_primary_requests_two_then_one_not_native_reply_count",
        list(map(len, primaries)) == [2, 1]
        and baseline["primary_calls"] == 4
        and fixed["primary_calls"] == 1
        and len(baseline["generation"]) == 3,
    )
    check(
        "api_replies_equal_actual_last_primary_output_without_audit_rewrite",
        all(
            p["response"]["reply"] == pcs[-1]["response"]["choices"][0]["message"]["content"]
            for p, pcs in zip(results, primaries)
        ),
    )
    check(
        "fixed_actual_before_question_dump_reused_no_previous_answer",
        fixed["restore_before_question_not_terminal_answer"]
        and fixed["before_question_backup"] == baseline["before_question_backup"]
        and fixed["before_question_backup"]["prior_same_task_answers"] == 0
        and hashlib.sha256((phase / "native-pg-resumed/before-question.dump").read_bytes()).hexdigest()
        == fixed["before_question_backup"]["database_sha256"],
    )
    storage = []
    for p in results:
        assert re.fullmatch(r"stage3_stage78_native_pg_(resumed|fixed)", p["database"])
        con = await asyncpg.connect(user="boot", database=p["database"], host=str(cluster / "socket"), port=25433)
        try:
            async with con.transaction(readonly=True):
                assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
                docs = await con.fetch(
                    "SELECT id,title,content,knowledge_base_id FROM knowledge_documents "
                    "WHERE title=ANY($1::text[]) ORDER BY id",
                    [d["title"] for d in fixture["documents"]],
                )
                chunks = await con.fetch(
                    'SELECT id,"documentId","chunkIndex",content FROM knowledge_chunks '
                    'WHERE "documentId"=ANY($1::bigint[]) ORDER BY id',
                    [d["id"] for d in docs],
                )
                links = await con.fetch(
                    "SELECT l.memory_id,s.source_message_id,s.body,s.owner_key,s.scope_key,s.state "
                    "FROM memory_source_links l JOIN memory_sources s ON l.source_key=s.source_key "
                    "WHERE l.memory_id=ANY($1::bigint[]) ORDER BY l.memory_id",
                    [row["id"] for row in p["seed_records"]],
                )
                messages = await con.fetch(
                    'SELECT message,reply FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 ORDER BY id',
                    "web",
                    "web-character",
                    "6",
                )
                storage.append(
                    {
                        "documents": [dict(x) for x in docs],
                        "chunks": [dict(x) for x in chunks],
                        "links": [dict(x) for x in links],
                        "messages": [dict(x) for x in messages],
                    }
                )
        finally:
            await con.close()
    check(
        "independent_pg_same_four_complete_originals",
        storage[0]["documents"] == storage[1]["documents"]
        and len(storage[1]["documents"]) == 4
        and all(
            any(d["title"] == x["title"] and d["content"] == x["content"] for d in storage[1]["documents"])
            for x in fixture["documents"]
        ),
    )
    check(
        "independent_pg_same_four_whole_chunks",
        storage[0]["chunks"] == storage[1]["chunks"] and len(storage[1]["chunks"]) == 4,
    )
    check(
        "independent_pg_same_three_own_original_links",
        storage[0]["links"] == storage[1]["links"]
        and len(storage[1]["links"]) == 3
        and all(
            link["body"] == fixture["source_message"]
            and link["owner_key"] == json.dumps(("web", "web-character", "6"))
            and link["state"] == "recorded"
            for link in storage[1]["links"]
        ),
    )
    check(
        "actual_successful_seed_and_nine_native_tasks_not_repeated",
        storage[0]["messages"][:-1] == storage[1]["messages"][:-1]
        and len(storage[1]["messages"]) == 11
        and storage[1]["messages"][0]["message"] == fixture["source_message"]
        and [x["message"] for x in storage[1]["messages"][1:-1]] == [x["message"].strip() for x in fixture["bridges"]],
    )
    ranking = [
        [{k: v for k, v in row.items() if k != "import_time"} for row in p["retrieval"][-1]["results"]] for p in results
    ]
    check(
        "rag_authority_order_contents_scores_unchanged",
        ranking[0] == ranking[1]
        and results[0]["retrieval"][-1]["source_coverage"] == results[1]["retrieval"][-1]["source_coverage"],
    )
    docs_by_id = {d["id"]: d for d in storage[1]["documents"]}
    scopes = []
    for p, pcs in zip(results, primaries):
        plan = p["generation"][-1]["retrieval"]
        content = unescape(pcs[0]["request"]["messages"][-1]["content"])
        match = re.search(r"<retrieval_coverage[^>]*>\n(.*?)\n</retrieval_coverage>", content, re.S)
        assert match
        scope = json.loads(match[1])
        scopes.append(scope)
        valid = []
        for coverage in plan["source_coverage"]:
            receipt = coverage["original_source_receipt"]
            doc = docs_by_id[receipt["document_id"]]
            packet = next(
                (x for x in plan["evidence_packets"] if x.get("document_ids") == [receipt["original_packet_id"]]), None
            )
            valid.append(
                packet is not None
                and coverage["original_status"] == "verified_original_body_admitted"
                and receipt["original_body_chars"] == len(doc["content"])
                and receipt["original_body_sha256"] == hashlib.sha256(doc["content"].encode()).hexdigest()
                and packet["original_body"] == doc["content"]
                and hashlib.sha256(packet["text"].encode()).hexdigest() == receipt["original_packet_sha256"]
                and packet["text"] in content
            )
        check(p["database"] + "_exact_current_db_original_grants_at_actual_wire", len(valid) == 4 and all(valid))
        check(
            p["database"] + "_public_wire_scope_matches_real_grants",
            {s["source_id"]: s["original_status"] for s in scope["sources"]}
            == {s["source_id"]: s["original_status"] for s in plan["source_coverage"]},
        )
    tests = list(ET.parse(phase / "candidate-tests.xml").iter("testcase"))
    check(
        "same_exact_candidate_as_twenty_targeted_tests_no_replay",
        len(tests) == 20
        and all(not list(t.iter("failure")) and not list(t.iter("error")) for t in tests)
        and Path("backend/character/memory_service.py").read_bytes()
        == (phase / "candidate_memory_service.py").read_bytes(),
    )
    for job in ["native-baseline-job.json", "native-resumed-job.json", "native-fixed-job.json"]:
        pid = json.loads((phase / job).read_text())["pid"]
        check(job + "_terminal", not Path(f"/proc/{pid}/cmdline").exists())
    budgets = [
        {
            "variant": variant,
            "request_index": i,
            "estimated_prompt_tokens": sum(estimated_tokens(m["content"]) + 4 for m in c["request"]["messages"]),
            "actual_prompt_tokens": c["response"]["usage"]["prompt_tokens"],
            "output_limit": c["request"]["max_tokens"],
            "safety_margin": 512,
            "configured_window": 65536,
            "actual_fits": c["response"]["usage"]["prompt_tokens"] + c["request"]["max_tokens"] + 512 <= 65536,
        }
        for variant, pcs in zip(variants, primaries)
        for i, c in enumerate(pcs)
    ]
    reply = fixed["response"]["reply"]
    result = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "runtime_memory_fix_native_effective": all(checks.values()),
        "initial_calls": 42,
        "resumed_calls": 15,
        "fixed_calls": 5,
        "phase_calls": 62,
        "http200": 58,
        "http402": 4,
        "completed_native_api_replies": 12,
        "new_completed_native_api_replies_after_resource_restored": 4,
        "final_task_cloud_primary_requests": [2, 1],
        "actual_recall": recalls,
        "native_memory_selection_qualified": 1,
        "native_public_original_grants_qualified": 1,
        "full_native_case_qualified": 0,
        "strict_semantic_answer_qualified": 0,
        "configured_token_budget_qualified": all(x["actual_fits"] for x in budgets),
        "actual_primary_budgets": budgets,
        "actual_scopes": scopes,
        "raw_baseline_primary_replies": [c["response"]["choices"][0]["message"]["content"] for c in primaries[0]],
        "raw_baseline_api_reply": baseline["response"]["reply"],
        "raw_fixed_api_reply": reply,
        "strict_open_findings": [
            "The configured65536 serving estimate undercounts these numeric ASCII archive histories: "
            "actual final primary prompt66907/66964/68116, while estimates39916/40009/41478. "
            "Provider200 is not proof of configured-budget correctness. Preserve full fixtures and investigate estimator; "
            "do not arbitrarily shrink configured history or claim memory repair fixes this.",
            "Fixed raw response reinterprets the independent character preference task as the user preference ranking. "
            "Baseline first raw primary correctly distinguishes character ownership, but existing output guard retries "
            "and returned second response reinterprets ownership. Full semantic understanding remains unqualified. "
            "No style-forcing patch, resampling, source omission or model-size attribution.",
        ],
        "harness_correction_required": "Original resumed probe assumes one primary request per successful API turn; "
        "actual existing output_guard retry adds second final primary request, yielding4 not3. "
        "Original assertion failure occurs after all actual result/calls saved; do not replay successful turns. "
        "Fix future observation count against each real GenerationResult.guard_retried receipt.",
        "candidate_tests": {"new": 19, "directly_affected_old": 1, "unique": 20, "executions": 20},
        "cloud_calls_during_audit": 0,
        "old_open_findings_preserved": True,
    }
    (phase / "native-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "evidence_checks": f"{result['passed']}/{result['total']}",
                "failed_checks": [k for k, value in checks.items() if not value],
                "memory_fix_effective": result["runtime_memory_fix_native_effective"],
                "full_native_case_qualified": 0,
                "configured_token_budget_qualified": result["configured_token_budget_qualified"],
                "phase_calls": 62,
                "http200": 58,
                "http402": 4,
            },
            ensure_ascii=False,
        )
    )
    assert all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    asyncio.run(run(parser.parse_args()))
