"""Read-only audit of named public sources across independent knowledge bases."""

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
    assert phase == runtime / "backups/backend-chain-20261001/stage82"
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

    case = read("fixture.json")
    seed = read("native-pg-seed/before-question.json")
    diagnostic = read("scope-seed-diagnostic.json")
    before = read("actual-api-retrieval-before.json")
    after = read("actual-api-retrieval-after.json")
    result = read("native-pg-fixed/result.json")
    calls = read("native-pg-fixed/cloud-calls.json")
    gate = read("native-pg-fixed/primary-input-observed.json")
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check(
        "full_parent_question_documents_and_all_private_information_retained",
        case["question"].startswith(case["original_question"] + "\n")
        and len(case["documents"]) == 4
        and len(case["bridges"]) == 9
        and not case["input_information_reduced"],
    )
    check(
        "new_four_library_layout_created_through_actual_authenticated_admin_apis",
        len(diagnostic["operations"]) == 3
        and all(x["create_status"] == x["update_status"] == 200 for x in diagnostic["operations"])
        and len({x["knowledge_base_id"] for x in diagnostic["documents"]}) == 4,
    )
    check(
        "before_actual_retrieval_missing_existing_complete_fourth_original",
        diagnostic["whole_originals_present"] == [True, True, True, False]
        and before["unresolved_requested_titles"] == ["绿泽寄件受理"]
        and before["requested_source_scope"] == "same_anchor_knowledge_base_and_original_filter"
        and set(diagnostic["ranked_anchor_bases"]) == {3, 4, 5}
        and set(diagnostic["available_requested_bases"]) == {3, 4, 5, 6},
    )
    check(
        "new_scope_setup_and_offline_diagnostics_have_zero_provider_calls",
        diagnostic["cloud_attempts"] == diagnostic["model_calls"] == after["model_calls"] == 0,
    )
    check(
        "after_same_actual_ranking_and_confidence_all_four_originals_available",
        after["bundle"]["results"][:3] == before["results"][:3]
        and after["bundle"]["confidence"] == before["confidence"]
        and all(after["whole_originals_present"])
        and after["bundle"]["requested_source_scope"] == "original_filter"
        and after["bundle"]["unresolved_requested_titles"] == [],
    )
    check(
        "actual_frozen_index_507_records_validated_without_snapshot_rewrite",
        after["validated_index"]
        and after["actual_indexed_count"] == 507
        and hashlib.sha256(
            (phase / "native-pg-seed/before-question-vectors/index_snapshot.zip").read_bytes()
        ).hexdigest()
        == after["frozen_before_question_vector_sha256"],
    )
    check(
        "new_frozen_dump_matches_real_authenticated_scope_setup_no_prior_answer",
        hashlib.sha256((phase / "native-pg-seed/before-question.dump").read_bytes()).hexdigest()
        == seed["before_question_backup"]["database_sha256"]
        == "00416458b561625a1db2be7ebae265ea4dffdfa5b8efee9c312479a71b6317c6"
        and seed["before_question_backup"]["prior_same_task_answers"] == 0
        and result["before_question_backup"] == seed["before_question_backup"],
    )
    check(
        "actual_native_authenticated_postgresql_grounded_api_result",
        result["transport"] == "authenticated_ASGI"
        and result["database_mode"] == "PostgreSQL"
        and result["http_status"] == 200
        and not result["response"]["abstained"],
    )
    check(
        "no_successful_import_memory_write_or_nine_history_tasks_replayed",
        result["knowledge_imports_replayed"]
        == result["seed_model_calls_replayed"]
        == result["successful_bridge_turns_replayed"]
        == 0
        and result["bridge_turns_completed"] == 9,
    )
    prepared = result["prepared"][-1]
    history = prepared["history"]
    check(
        "prepared_history_remains_sixteen_complete_messages_without_source_or_prior_question",
        len(history) == 16
        and all(case["source_message"] not in x["content"] and case["question"] not in x["content"] for x in history),
    )
    check(
        "three_real_original_claims_before_after_unchanged",
        result["seed_records"] == seed["seed_records"] == result["user_fact_records_after"]
        and len(result["seed_records"]) == 3,
    )
    check(
        "all_three_real_memories_read_usable_selected",
        prepared["recall"]["records_read"]
        == prepared["recall"]["usable_records"]
        == prepared["recall"]["selected_count"]
        == 3
        and prepared["used_memory_ids"] == ["42", "41", "40"],
    )
    check(
        "whole_private_original_and_negation_and_condition_preserved_as_observation",
        prepared["recall"]["temporal_views"] == {"fact": 0, "asserted_state": 0, "observation": 3}
        and all(
            case["source_message"] in x["evidence"] and x["temporal_mode"] == "observation"
            for x in prepared["memory_packets"]
        )
        and any(
            dict(x["qualifiers"]) == {"condition": "完整表格已提交且身份核验通过"} for x in prepared["memory_packets"]
        ),
    )
    primary = [
        c
        for c in calls
        if any(
            "<user_query>" in x["content"] and case["question"] in unescape(x["content"])
            for x in c["request"]["messages"]
        )
    ]
    check(
        "four_actual_pro_http200_stop_calls_one_actual_primary",
        len(calls) == 4
        and len(primary) == result["primary_calls"] == result["generation"][0]["observed_primary_calls"] == 1
        and all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
    )
    raw = primary[0]["response"]["choices"][0]["message"]["content"]
    check("native_api_reply_is_exact_unmodified_real_cloud_output", raw == result["response"]["reply"])
    wire = primary[0]["request"]["messages"]
    whole = "\n".join(unescape(x["content"]) for x in wire)
    check(
        "four_complete_public_originals_private_original_and_full_new_query_at_actual_wire",
        all(gate["document_bodies_present"].values())
        and gate["private_source_present"]
        and case["source_message"] in whole
        and case["question"] in whole
        and all(x["content"] in whole for x in case["documents"]),
    )
    check(
        "only_existing_budget_retains_six_whole_recent_native_turns",
        len(wire[1:-1]) == 12
        and wire[1:-1] == history[-12:]
        and all(wire[1:-1][i]["role"] == "user" and wire[1:-1][i + 1]["role"] == "assistant" for i in range(0, 12, 2)),
    )
    check(
        "native_grounded_plan_no_output_guard_retry_or_rewrite",
        result["generation"][0]["retrieval"]["status"] == "ok"
        and not result["generation"][0]["guard_retried"]
        and not result["generation"][0]["guard_violations"]
        and not result["generation"][0]["guard_fallback"],
    )
    tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def official_full(messages):
        text = "<｜begin▁of▁sentence｜>"
        for i, m in enumerate(messages):
            role, body = m["role"], m["content"]
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
    for i, c in enumerate(calls):
        messages = c["request"]["messages"]
        budgets.append(
            dict(
                index=i,
                actual_prompt_tokens=c["response"]["usage"]["prompt_tokens"],
                input_bound=sum(count(x["content"]) + 4 for x in messages),
                official_full=official_full(messages),
                output_reserved=c["request"]["max_tokens"],
                margin=512,
                window=65536,
            )
        )
    check(
        "all_real_prompt_tokens_match_independent_official_pinned_encoding",
        all(x["official_full"] == x["actual_prompt_tokens"] for x in budgets),
    )
    check(
        "all_real_bounds_match_pre_send_cover_api_and_reserve_output_inside_65536",
        all(
            x["input_bound"] == c["budget"]["input_bound"]
            and x["input_bound"] >= x["actual_prompt_tokens"]
            and x["input_bound"] + x["output_reserved"] + 512 <= 65536
            for x, c in zip(budgets, calls)
        ),
    )
    stored = []
    for database in ["stage3_stage82_scope_seed", result["database"]]:
        assert re.fullmatch(r"stage3_stage82_(?:scope_seed|native_pg_fixed)", database)
        con = await asyncpg.connect(user="boot", database=database, host=str(cluster / "socket"), port=25433)
        try:
            async with con.transaction(readonly=True):
                assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
                docs = await con.fetch(
                    "SELECT id,title,content,category,knowledge_base_id FROM knowledge_documents "
                    "WHERE id=ANY($1) ORDER BY id",
                    [504, 505, 506, 507],
                )
                links = await con.fetch(
                    "SELECT l.memory_id,s.body,s.owner_key,s.scope_key,s.state "
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
        "independent_current_database_four_distinct_bases_and_complete_originals_unchanged",
        stored[0]["documents"] == stored[1]["documents"]
        and len(stored[1]["documents"]) == 4
        and {x["knowledge_base_id"] for x in stored[1]["documents"]} == {3, 4, 5, 6}
        and all(
            any(x["content"] == d["content"] and x["title"] == d["title"] for x in stored[1]["documents"])
            for d in case["documents"]
        ),
    )
    check(
        "independent_current_database_exact_three_user_owner_source_links_unchanged",
        stored[0]["links"] == stored[1]["links"]
        and len(stored[1]["links"]) == 3
        and all(
            x["body"] == case["source_message"]
            and x["state"] == "recorded"
            and x["owner_key"] == json.dumps(("web", "web-character", "6"))
            for x in stored[1]["links"]
        ),
    )
    check(
        "independent_database_original_seed_and_nine_native_history_tasks_unchanged",
        len(stored[0]["messages"]) == 10
        and len(stored[1]["messages"]) == 11
        and stored[0]["messages"] == stored[1]["messages"][:-1]
        and [x["message"] for x in stored[1]["messages"][1:-1]] == [x["message"].strip() for x in case["bridges"]],
    )
    plan = result["generation"][0]["retrieval"]
    docs = {x["id"]: x for x in stored[1]["documents"]}
    grants = []
    for coverage in plan["source_coverage"]:
        receipt = coverage["original_source_receipt"]
        doc = docs[receipt["document_id"]]
        packet = next(
            (x for x in plan["evidence_packets"] if x.get("document_ids") == [receipt["original_packet_id"]]), None
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
    check("four_current_database_original_authority_grants_match_full_actual_wire", len(grants) == 4 and all(grants))
    coverage = json.loads(
        re.search(r"<retrieval_coverage[^>]*>\n(.*?)\n</retrieval_coverage>", unescape(wire[-1]["content"]), re.S)[1]
    )
    check(
        "wire_scope_is_exact_current_original_authority_receipt_scope",
        {x["source_id"]: x["original_status"] for x in coverage["sources"]}
        == {x["source_id"]: x["original_status"] for x in plan["source_coverage"]},
    )
    tests = list(ET.parse(phase / "targeted-tests.xml").iter("testcase"))
    check(
        "twenty_one_unique_directly_affected_nodes_pass_once",
        len(tests) == 21
        and len({(x.attrib["classname"], x.attrib["name"]) for x in tests}) == 21
        and all(not list(x) for x in tests),
    )
    tested = read("targeted-test-summary.json")
    check(
        "exact_executed_source_and_full_fixture_hashes_unchanged",
        all(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha for name, sha in tested["tested_hashes"].items()
        ),
    )
    static = read("static.json")
    check(
        "no_added_static_issues_in_runtime_or_updated_scope_tests",
        not any(static["runtime_added_issues"].values()) and not static["new_test_issues"],
    )
    terminal = []
    for name in ["scope-seed-job.json", "native-fixed-job.json"]:
        pid = read(name)["pid"]
        try:
            live = bool(Path(f"/proc/{pid}/cmdline").read_bytes()) and (
                Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
            )
        except OSError:
            live = False
        terminal.append(not live)
    check("actual_scope_setup_and_native_jobs_terminal", all(terminal))
    check(
        "raw_semantic_errors_preserved_without_relabelling_full_case_pass",
        "偏好上你不选择代办" in raw and "不能推出你最喜欢哪一个" in raw,
    )
    report = dict(
        checks=checks,
        passed=sum(checks.values()),
        total=len(checks),
        native_cross_kb_source_fix_qualified=all(checks.values()),
        phase_calls=len(calls),
        http200=4,
        completed_native_api_replies=1,
        actual_primary_calls=1,
        native_current_public_original_grants=4,
        native_private_memory_selected=3,
        configured_token_budget_qualified=True,
        actual_call_budgets=budgets,
        main_actual_prompt_tokens=budgets[-1]["actual_prompt_tokens"],
        main_input_bound=budgets[-1]["input_bound"],
        full_native_case_qualified=0,
        strict_semantic_answer_qualified=0,
        open_semantic_findings=[
            "The independent role-favorite question is initially recast as the user's overall ranking, although the final actor-evidence paragraph separately preserves role unknown.",
            "Dislike of agency handling is promoted to an actual not-choose statement without an action receipt.",
            "Mail evidence absence is worded as never provided rather than limited to obtained scope.",
        ],
        raw_api_reply=raw,
        model_calls_during_audit=0,
    )
    (phase / "native-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    print(json.dumps({k: v for k, v in report.items() if k not in {"checks", "actual_call_budgets", "raw_api_reply"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.phase.resolve()))
