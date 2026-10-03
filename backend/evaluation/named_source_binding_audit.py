"""Read-only evidence audit for the complete named-source binding regression."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path


async def run(phase):
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    assert phase == runtime / "backups/backend-chain-20261001/stage81"
    cluster = runtime / "evaluations/r148pg.s3"
    os.environ.update(
        MODEL_PROVIDER="openai_compat",
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
    )
    import asyncpg
    from tokenizers import Tokenizer

    from inference.token_counting import _TOKENIZER_PATH

    def read(name):
        return json.loads((phase / name).read_text())

    fixture = read("fixture.json")
    before = read("native-pg-binding/result.json")
    result = read("native-pg-binding-fixed/result.json")
    before_calls = read("native-pg-binding/cloud-calls.json")
    calls = read("native-pg-binding-fixed/cloud-calls.json")
    gate_before = read("native-pg-binding/primary-input-observed.json")
    gate = read("native-pg-binding-fixed/primary-input-observed.json")
    before_index = read("filter-real-index-before.json")
    after_index = read("filter-real-index-after.json")
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check(
        "complete_original_query_retained_verbatim_in_new_task",
        fixture["question"].startswith(fixture["original_question"] + "\n")
        and not fixture["input_information_reduced"]
        and len(fixture["documents"]) == 4,
    )
    check(
        "actual_baseline_four_originals_missing_and_no_primary_sent",
        not any(gate_before["document_bodies_present"].values())
        and gate_before["private_source_present"]
        and before["primary_calls"] == 0
        and len(before_calls) == 3,
    )
    check(
        "baseline_three_real_calls_http200_not_relabelled_primary",
        all(c["http_status"] == 200 for c in before_calls)
        and before["generation"][0]["model_invoked"]
        and not before["generation"][0]["observed_primary_calls"]
        and before["response"]["abstained"],
    )
    check(
        "actual_before_validated_index_excludes_all_sources_by_incidental_category",
        before_index["validated"]
        and len(before_index["old_bundle"]["results"]) == 3
        and before_index["new_bundle"]["abstained"]
        and not before_index["new_bundle"]["results"]
        and any(c["filters"] == {"category": "角色"} for c in before_index["actual_searches"])
        and all(
            d["category"] == "办理"
            for d in before_index["metadata"]
            if d["title"] in [x["title"] for x in fixture["documents"]]
        ),
    )
    check(
        "offline_after_same_index_full_query_and_all_current_originals",
        after_index["snapshot_unchanged"]
        and all(after_index["requested_whole_originals_present"])
        and after_index["actual_searches"][0]["query"] == fixture["question"]
        and all(not c["filters"] for c in after_index["actual_searches"]),
    )
    check(
        "offline_lineage_and_filter_diagnostics_make_zero_model_calls",
        read("source-lineage-audit.json")["passed"] == 11
        and before_index["model_calls"] == after_index["model_calls"] == 0,
    )
    check(
        "authenticated_postgresql_native_grounded_reply",
        result["transport"] == "authenticated_ASGI"
        and result["database_mode"] == "PostgreSQL"
        and result["http_status"] == 200
        and not result["response"]["abstained"],
    )
    check(
        "no_successful_import_writer_or_history_task_replay",
        result["knowledge_imports_replayed"]
        == result["seed_model_calls_replayed"]
        == result["successful_bridge_turns_replayed"]
        == 0
        and result["bridge_turns_completed"] == 9,
    )
    check(
        "same_original_before_question_dump_and_zero_prior_task_answer",
        before["before_question_backup"] == result["before_question_backup"]
        and result["before_question_backup"]["prior_same_task_answers"] == 0
        and hashlib.sha256((phase / "native-pg-seed/before-question.dump").read_bytes()).hexdigest()
        == "094c6543c291fef26307210937a23d23f7b48aea16ffd20028f1ed99b2d5627e",
    )
    prepared = result["prepared"][-1]
    history = prepared["history"]
    check(
        "same_sixteen_full_prepared_history_messages_without_original_seed_in_recent_window",
        history == before["prepared"][-1]["history"]
        and len(history) == 16
        and all(
            fixture["source_message"] not in h["content"] and fixture["question"] not in h["content"] for h in history
        ),
    )
    check(
        "three_real_claims_unchanged_before_and_after",
        result["seed_records"] == before["seed_records"] == result["user_fact_records_after"]
        and len(result["seed_records"]) == 3,
    )
    check(
        "three_real_records_read_usable_and_selected",
        prepared["recall"]["records_read"]
        == prepared["recall"]["usable_records"]
        == prepared["recall"]["selected_count"]
        == 3
        and prepared["used_memory_ids"] == ["42", "41", "40"],
    )
    check(
        "negation_and_full_conditional_source_preserved_as_user_observations",
        prepared["recall"]["temporal_views"] == {"fact": 0, "asserted_state": 0, "observation": 3}
        and all(
            fixture["source_message"] in p["evidence"] and p["temporal_mode"] == "observation"
            for p in prepared["memory_packets"]
        )
        and any(
            dict(p["qualifiers"]) == {"condition": "完整表格已提交且身份核验通过"} for p in prepared["memory_packets"]
        ),
    )
    primary = [
        c
        for c in calls
        if any(
            "<user_query>" in m["content"] and fixture["question"] in unescape(m["content"])
            for m in c["request"]["messages"]
        )
    ]
    check(
        "one_actual_primary_separate_from_four_real_provider_calls",
        len(calls) == 4
        and len(primary) == result["primary_calls"] == result["generation"][0]["observed_primary_calls"] == 1,
    )
    check(
        "all_seven_phase_provider_calls_are_pro_http200_stop",
        all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in before_calls + calls
        ),
    )
    raw = primary[0]["response"]["choices"][0]["message"]["content"]
    check("actual_reply_equals_unmodified_cloud_reply", raw == result["response"]["reply"])
    wire = primary[0]["request"]["messages"]
    whole = "\n".join(unescape(m["content"]) for m in wire)
    check(
        "all_full_required_public_private_originals_and_question_in_actual_wire",
        all(gate["document_bodies_present"].values())
        and gate["private_source_present"]
        and fixture["source_message"] in whole
        and fixture["question"] in whole
        and all(d["content"] in whole for d in fixture["documents"]),
    )
    check(
        "canonical_history_is_only_six_whole_recent_turns",
        wire[1:-1] == history[-12:]
        and len(wire[1:-1]) == 12
        and all(wire[1:-1][i]["role"] == "user" and wire[1:-1][i + 1]["role"] == "assistant" for i in range(0, 12, 2)),
    )
    tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def official_full(messages):
        text = "<｜begin▁of▁sentence｜>"
        for i, message in enumerate(messages):
            role, body = message["role"], message["content"]
            if role == "system":
                text += body
            elif role == "user":
                text += "<｜User｜>" + body
                if i == len(messages) - 1 or messages[i + 1]["role"] == "assistant":
                    text += "<｜Assistant｜></think>"
            else:
                assert role == "assistant"
                text += body + "<｜end▁of▁sentence｜>"
        return count(text)

    budgets = []
    for i, c in enumerate(before_calls + calls):
        request = c["request"]
        actual = c["response"]["usage"]["prompt_tokens"]
        bound = sum(count(m["content"]) + 4 for m in request["messages"])
        budgets.append(
            dict(
                index=i,
                actual_prompt_tokens=actual,
                input_bound=bound,
                official_full=official_full(request["messages"]),
                output_reserved=request["max_tokens"],
                margin=512,
                window=65536,
            )
        )
    check(
        "all_actual_prompt_counts_match_independent_pinned_official_encoding",
        all(x["actual_prompt_tokens"] == x["official_full"] for x in budgets),
    )
    check(
        "all_bounds_match_pre_send_and_cover_actual",
        all(
            x["input_bound"] >= x["actual_prompt_tokens"] and x["input_bound"] == c["budget"]["input_bound"]
            for x, c in zip(budgets, before_calls + calls)
        ),
    )
    check(
        "all_seven_actual_requests_fit_unchanged_full_65536_with_output_and_margin",
        all(x["input_bound"] + x["output_reserved"] + 512 <= 65536 for x in budgets),
    )
    plan = result["generation"][0]["retrieval"]
    check(
        "grounded_plan_no_guard_retry_or_output_rewrite",
        plan["status"] == "ok"
        and not result["generation"][0]["guard_retried"]
        and not result["generation"][0]["guard_violations"]
        and not result["generation"][0]["guard_fallback"],
    )
    stored = []
    for item in [before, result]:
        assert re.fullmatch(r"stage3_stage81_native_pg_binding(?:_fixed)?", item["database"])
        con = await asyncpg.connect(user="boot", database=item["database"], host=str(cluster / "socket"), port=25433)
        try:
            async with con.transaction(readonly=True):
                assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
                docs = await con.fetch(
                    "SELECT id,title,content,knowledge_base_id FROM knowledge_documents WHERE id=ANY($1) ORDER BY id",
                    [504, 505, 506, 507],
                )
                links = await con.fetch(
                    "SELECT l.memory_id,s.source_message_id,s.body,s.owner_key,s.scope_key,s.state "
                    "FROM memory_source_links l JOIN memory_sources s ON l.source_key=s.source_key "
                    "WHERE l.memory_id=ANY($1::bigint[]) ORDER BY l.memory_id",
                    [40, 41, 42],
                )
                messages = await con.fetch(
                    'SELECT message,reply FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 ORDER BY id',
                    "web",
                    "web-character",
                    "6",
                )
                stored.append(
                    dict(
                        documents=[dict(x) for x in docs],
                        links=[dict(x) for x in links],
                        messages=[dict(x) for x in messages],
                    )
                )
        finally:
            await con.close()
    check(
        "independent_database_all_four_complete_authorized_documents_unchanged",
        stored[0]["documents"] == stored[1]["documents"]
        and len(stored[1]["documents"]) == 4
        and all(
            any(d["title"] == expected["title"] and d["content"] == expected["content"] for d in stored[1]["documents"])
            for expected in fixture["documents"]
        ),
    )
    check(
        "independent_database_three_exact_owner_private_source_links_unchanged",
        stored[0]["links"] == stored[1]["links"]
        and len(stored[1]["links"]) == 3
        and all(
            x["body"] == fixture["source_message"]
            and x["state"] == "recorded"
            and x["owner_key"] == json.dumps(("web", "web-character", "6"))
            for x in stored[1]["links"]
        ),
    )
    check(
        "independent_database_successful_nine_native_history_tasks_unchanged",
        stored[0]["messages"][:-1] == stored[1]["messages"][:-1]
        and len(stored[1]["messages"]) == 11
        and [x["message"] for x in stored[1]["messages"][1:-1]] == [x["message"].strip() for x in fixture["bridges"]],
    )
    docs = {x["id"]: x for x in stored[1]["documents"]}
    grants = []
    for coverage in plan["source_coverage"]:
        receipt = coverage["original_source_receipt"]
        doc = docs[receipt["document_id"]]
        packet = next(
            (p for p in plan["evidence_packets"] if p.get("document_ids") == [receipt["original_packet_id"]]), None
        )
        grants.append(
            packet is not None
            and coverage["original_status"] == "verified_original_body_admitted"
            and receipt["original_body_chars"] == len(doc["content"])
            and receipt["original_body_sha256"] == hashlib.sha256(doc["content"].encode()).hexdigest()
            and packet["original_body"] == doc["content"]
            and hashlib.sha256(packet["text"].encode()).hexdigest() == receipt["original_packet_sha256"]
            and packet["text"] in unescape(wire[-1]["content"])
        )
    check("all_four_current_original_grants_match_fresh_database_and_actual_wire", len(grants) == 4 and all(grants))
    coverage_wire = json.loads(
        re.search(r"<retrieval_coverage[^>]*>\n(.*?)\n</retrieval_coverage>", unescape(wire[-1]["content"]), re.S)[1]
    )
    check(
        "actual_wire_coverage_equals_current_original_grants",
        {x["source_id"]: x["original_status"] for x in coverage_wire["sources"]}
        == {x["source_id"]: x["original_status"] for x in plan["source_coverage"]},
    )
    tests = list(ET.parse(phase / "targeted-tests.xml").iter("testcase"))
    check(
        "thirteen_unique_affected_nodes_pass_once",
        len(tests) == 13
        and len({(x.attrib["classname"], x.attrib["name"]) for x in tests}) == 13
        and all(not list(x) for x in tests),
    )
    tested = read("targeted-test-summary.json")
    check(
        "exact_executed_runtime_and_case_bytes_unchanged",
        all(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha for name, sha in tested["tested_hashes"].items()
        ),
    )
    check(
        "no_added_static_issues",
        not read("static.json")["runtime_added_issues"] and not read("static.json")["new_test_issues"],
    )
    terminal = []
    for file in ["native-binding-job.json", "native-binding-fixed-job.json"]:
        pid = read(file)["pid"]
        try:
            live = bool(Path(f"/proc/{pid}/cmdline").read_bytes()) and (
                Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
            )
        except OSError:
            live = False
        terminal.append(not live)
    check("both_actual_jobs_terminal", all(terminal))
    report = dict(
        checks=checks,
        passed=sum(checks.values()),
        total=len(checks),
        native_filter_counterexample_fixed=all(checks.values()),
        phase_calls=7,
        http200=7,
        completed_native_api_replies=2,
        baseline_abstention_api_replies=1,
        new_grounded_native_api_replies=1,
        actual_primary_calls=1,
        configured_token_budget_qualified=True,
        actual_call_budgets=budgets,
        main_actual_prompt_tokens=budgets[-1]["actual_prompt_tokens"],
        main_input_bound=budgets[-1]["input_bound"],
        native_private_memory_selected=3,
        native_current_public_original_grants=4,
        semantic_qualification_scope="new_complete_augmented_case",
        raw_api_reply=raw,
        model_calls_during_audit=0,
    )
    (phase / "native-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    print(json.dumps({k: v for k, v in report.items() if k not in {"checks", "raw_api_reply", "actual_call_budgets"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.phase.resolve()))
