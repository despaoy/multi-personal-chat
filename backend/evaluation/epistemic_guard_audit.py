"""Audit the recorded guard counterexample and a fresh complete native Pro case."""

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
    assert phase == runtime / "backups/backend-chain-20261001/stage80"
    os.environ.update(
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        DATABASE_PATH=str(phase / "audit-import-only.sqlite"),
        MODEL_PROVIDER="openai_compat",
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
    )
    import asyncpg
    from tokenizers import Tokenizer

    from inference.token_counting import _TOKENIZER_PATH, token_counter_info

    old_root = phase.parent / "stage79/native-pg-fixed"
    root = phase / "native-pg-fixed"
    old = json.loads((old_root / "result.json").read_text())
    result = json.loads((root / "result.json").read_text())
    calls = json.loads((root / "cloud-calls.json").read_text())
    fixture = json.loads((phase / "fixture.json").read_text())
    calibration = json.loads((phase.parent / "stage79/tokenizer-actual-request-comparison.json").read_text())
    tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def full_chat(messages):
        # Independently reproduce the official basic nonthinking format.
        # Parse pinned data only; never execute downloaded model Python.
        text = "<｜begin▁of▁sentence｜>"
        for index, message in enumerate(messages):
            assert set(message) == {"role", "content"}
            role, content = message["role"], message["content"]
            if role == "system":
                text += content
            elif role == "user":
                text += "<｜User｜>" + content
                if index == len(messages) - 1 or messages[index + 1]["role"] == "assistant":
                    text += "<｜Assistant｜></think>"
            else:
                assert role == "assistant"
                text += content + "<｜end▁of▁sentence｜>"
        return count(text)

    check(
        "actual_authenticated_postgresql_native_reply",
        result["transport"] == "authenticated_ASGI"
        and result["database_mode"] == "PostgreSQL"
        and result["http_status"] == 200,
    )
    check(
        "only_failed_question_run_no_successful_seed_or_bridge_replay",
        result["seed_model_calls_replayed"] == result["successful_bridge_turns_replayed"] == 0
        and len(result["generation"]) == 1
        and result["bridge_turns_completed"] == 9,
    )
    check(
        "same_before_question_dump_no_prior_task_answer",
        result["restore_before_question_not_terminal_answer"]
        and result["before_question_backup"] == old["before_question_backup"]
        and result["before_question_backup"]["prior_same_task_answers"] == 0
        and hashlib.sha256((phase / "native-pg-seed/before-question.dump").read_bytes()).hexdigest()
        == result["before_question_backup"]["database_sha256"],
    )
    history = result["prepared"][-1]["history"]
    check(
        "same_complete_prepared_history_and_unmodified_original_fixture",
        history == old["prepared"][-1]["history"]
        and len(history) == 16
        and (phase / "fixture.json").read_bytes() == (phase.parent / "stage78/fixture.json").read_bytes()
        and all(
            fixture["source_message"] not in h["content"] and fixture["question"] not in h["content"] for h in history
        ),
    )
    check(
        "same_three_real_pro_written_claims_before_and_after",
        result["seed_records"] == old["seed_records"] == result["user_fact_records_after"]
        and len(result["seed_records"]) == 3
        and result["original_seed_claims_exact_verified"],
    )
    prepared = result["prepared"][-1]
    recall = prepared["recall"]
    ids = {str(x["id"]) for x in result["seed_records"]}
    check(
        "same_three_usable_selected_memory_ids",
        recall["records_read"] == recall["usable_records"] == recall["selected_count"] == 3
        and set(prepared["used_memory_ids"]) == ids
        and {p["memory_id"] for p in prepared["memory_packets"]} == ids,
    )
    check(
        "private_original_negation_and_necessary_and_kept_as_observations",
        recall["temporal_views"] == {"fact": 0, "asserted_state": 0, "observation": 3}
        and all(
            p["temporal_mode"] == "observation" and fixture["source_message"] in p["evidence"]
            for p in prepared["memory_packets"]
        )
        and any(
            dict(p["qualifiers"]) == {"condition": "完整表格已提交且身份核验通过"} for p in prepared["memory_packets"]
        )
        and any("不喜欢代办受理" in p["content"] for p in prepared["memory_packets"]),
    )
    check(
        "existing_twenty_requests_calibrate_without_new_model_calls",
        len(calibration["observations"]) == 20
        and calibration["all_body_plus_four_cover_actual"]
        and all(x["actual_minus_official_full"] == 0 for x in calibration["observations"]),
    )
    check(
        "pinned_offline_counter_used",
        token_counter_info()["mode"] == "deepseek_v4_pro_bpe" and token_counter_info()["network_requests"] == 0,
    )
    check(
        "all_five_actual_pro_calls_saved_http200_stop",
        len(calls) == 5
        and all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
    )
    budgets = []
    for index, call in enumerate(calls):
        request = call["request"]
        messages = request["messages"]
        actual = call["response"]["usage"]["prompt_tokens"]
        bound = sum(count(m["content"]) + 4 for m in messages)
        budgets.append(
            dict(
                index=index,
                message_count=len(messages),
                official_full_chat=full_chat(messages),
                input_bound=bound,
                actual_prompt_tokens=actual,
                output_limit=request["max_tokens"],
                safety_margin=512,
                configured_window=65536,
                actual_fits=actual + request["max_tokens"] + 512 <= 65536,
            )
        )
    check(
        "all_actual_api_prompt_counts_match_independent_official_basic_encoding",
        all(b["official_full_chat"] == b["actual_prompt_tokens"] for b in budgets),
    )
    check(
        "all_input_bounds_cover_actual_and_match_pre_send_measurements",
        all(
            b["input_bound"] >= b["actual_prompt_tokens"] and b["input_bound"] == c["budget"]["input_bound"]
            for b, c in zip(budgets, calls)
        ),
    )
    check(
        "all_actual_requests_reserve_output_and_margin_within_unchanged_65536",
        all(b["actual_fits"] and b["input_bound"] + b["output_limit"] + 512 <= 65536 for b in budgets),
    )
    primaries = [
        c
        for c in calls
        if any(
            "<user_query>" in m["content"] and fixture["question"] in unescape(m["content"])
            for m in c["request"]["messages"]
        )
    ]
    check(
        "one_actual_primary_and_one_native_api_reply_counted_separately",
        len(primaries) == result["primary_calls"] == result["generation"][0]["observed_primary_calls"] == 1
        and result["generation"][0]["model_invoked"]
        and result["generation"][0]["response_mode"] == "generated",
    )
    check(
        "actual_api_reply_equals_raw_cloud_reply_without_rewrite",
        result["response"]["reply"] == primaries[-1]["response"]["choices"][0]["message"]["content"],
    )
    wire = primaries[0]["request"]["messages"]
    kept = wire[1:-1]
    check(
        "canonical_budget_prunes_only_whole_recent_history_turns",
        len(kept) == 12
        and kept == history[-12:]
        and all(kept[i]["role"] == "user" and kept[i + 1]["role"] == "assistant" for i in range(0, len(kept), 2)),
    )
    whole = "\n".join(unescape(m["content"]) for m in wire)
    check(
        "every_required_complete_public_private_original_and_current_question_at_wire",
        fixture["question"] in whole
        and fixture["source_message"] in whole
        and all(d["content"] in whole for d in fixture["documents"]),
    )
    check(
        "guard_did_not_retry_or_replace_actual_answer",
        not result["generation"][0]["guard_retried"]
        and not result["generation"][0]["guard_fallback"]
        and not result["generation"][0]["guard_violations"],
    )

    def ranking(p):
        return [{k: v for k, v in row.items() if k != "import_time"} for row in p["retrieval"][-1]["results"]]

    check(
        "rag_authority_ranking_contents_scores_and_original_receipts_unchanged",
        ranking(old) == ranking(result)
        and old["retrieval"][-1]["source_coverage"] == result["retrieval"][-1]["source_coverage"],
    )
    storage = []
    for p in [old, result]:
        assert re.fullmatch(r"stage3_stage(79|80)_native_pg_fixed", p["database"])
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
                    [r["id"] for r in p["seed_records"]],
                )
                messages = await con.fetch(
                    'SELECT message,reply FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 ORDER BY id',
                    "web",
                    "web-character",
                    "6",
                )
                storage.append(
                    dict(
                        documents=[dict(x) for x in docs],
                        chunks=[dict(x) for x in chunks],
                        links=[dict(x) for x in links],
                        messages=[dict(x) for x in messages],
                    )
                )
        finally:
            await con.close()
    check(
        "independent_database_four_complete_originals_unchanged",
        storage[0]["documents"] == storage[1]["documents"]
        and len(storage[1]["documents"]) == 4
        and all(
            any(
                d["content"] == expected["content"] and d["title"] == expected["title"] for d in storage[1]["documents"]
            )
            for expected in fixture["documents"]
        ),
    )
    check(
        "independent_database_four_full_chunks_unchanged",
        storage[0]["chunks"] == storage[1]["chunks"] and len(storage[1]["chunks"]) == 4,
    )
    check(
        "independent_database_same_three_private_own_source_links",
        storage[0]["links"] == storage[1]["links"]
        and len(storage[1]["links"]) == 3
        and all(
            x["body"] == fixture["source_message"]
            and x["state"] == "recorded"
            and x["owner_key"] == json.dumps(("web", "web-character", "6"))
            for x in storage[1]["links"]
        ),
    )
    check(
        "same_successful_native_history_persisted_without_replay",
        storage[0]["messages"][:-1] == storage[1]["messages"][:-1]
        and len(storage[1]["messages"]) == 11
        and storage[1]["messages"][0]["message"] == fixture["source_message"]
        and [x["message"] for x in storage[1]["messages"][1:-1]] == [x["message"].strip() for x in fixture["bridges"]],
    )
    plan = result["generation"][-1]["retrieval"]
    docs = {d["id"]: d for d in storage[1]["documents"]}
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
    check("all_four_current_original_grants_match_full_final_wire", len(grants) == 4 and all(grants))
    scope = json.loads(
        re.search(r"<retrieval_coverage[^>]*>\n(.*?)\n</retrieval_coverage>", unescape(wire[-1]["content"]), re.S)[1]
    )
    check(
        "actual_wire_coverage_matches_original_receipts",
        {s["source_id"]: s["original_status"] for s in scope["sources"]}
        == {s["source_id"]: s["original_status"] for s in plan["source_coverage"]},
    )
    tests = list(ET.parse(phase / "targeted-tests.xml").iter("testcase"))
    check(
        "seventeen_unique_direct_tests_pass_once",
        len(tests) == 17
        and len({(t.attrib["classname"], t.attrib["name"]) for t in tests}) == 17
        and all(not list(t) for t in tests),
    )
    pid = json.loads((phase / "native-fixed-job.json").read_text())["pid"]
    check("actual_job_terminal", not Path(f"/proc/{pid}/cmdline").exists())
    static = json.loads((phase / "static.json").read_text())
    check(
        "changed_code_has_no_added_static_issues",
        not any(static["new_file_issues"].values()) and not any(static["runtime_added_issues"].values()),
    )
    tested = json.loads((phase / "targeted-test-summary.json").read_text())
    check(
        "runtime_and_fixture_bytes_identical_to_seventeen_executed_tests",
        all(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha for name, sha in tested["tested_hashes"].items()
        ),
    )
    diagnostic = json.loads((phase / "baseline-guard-diagnostic.json").read_text())
    check(
        "full_recorded_native_counterexample_preserved_and_reproduced_without_cloud_calls",
        diagnostic["legacy_match"] == ["我无法回答"]
        and diagnostic["captured_generator_calls"] == 2
        and diagnostic["actual_model_calls"] == diagnostic["native_api_replies"] == 0
        and diagnostic["fixture_sha256"]
        == tested["tested_hashes"]["backend/tests/fixtures/deepseek_epistemic_guard_case.json"]
        and diagnostic["fixture_bytes"] == 166558
        and hashlib.sha256((phase / "original-backend_character_output_guard.py").read_bytes()).hexdigest()
        == diagnostic["original_guard_sha256"],
    )
    old_primaries = [
        c
        for c in json.loads((old_root / "cloud-calls.json").read_text())
        if any(
            "<user_query>" in m["content"] and fixture["question"] in unescape(m["content"])
            for m in c["request"]["messages"]
        )
    ]
    check(
        "new_native_primary_request_identical_to_prior_budget_verified_main_request",
        len(old_primaries) == 1 and primaries[0]["request"] == old_primaries[0]["request"],
    )
    comparisons = result["guard_validation_comparisons"]
    check(
        "new_actual_guard_observed_with_factual_check_enabled_without_false_retry",
        len(comparisons) == 1
        and comparisons[0]["actual_guard"]["factual_task"]
        and comparisons[0]["actual_guard"] == result["prepared"][-1]["prepared_reply_guard"]
        and comparisons[0]["actual_reply"] == result["response"]["reply"]
        and comparisons[0]["current_violations"] == comparisons[0]["previous_violations"] == []
        and comparisons[0]["comparison_model_calls"] == 0
        and result["generation"][0]["reply_guard_mode"] == "lightweight",
    )
    counterexample_coverage = any(
        "factual_task_style_drift" in x["previous_violations"]
        and "factual_task_style_drift" not in x["current_violations"]
        for x in comparisons
    )
    check("fresh_native_not_misreported_as_old_counterexample_trigger_coverage", not counterexample_coverage)
    report = dict(
        checks=checks,
        passed=sum(checks.values()),
        total=len(checks),
        recorded_native_guard_counterexample_fixed=all(checks.values()),
        fresh_native_chain_verified=all(checks.values()),
        native_guard_counterexample_trigger_observed=counterexample_coverage,
        phase_calls=len(calls),
        http200=sum(c["http_status"] == 200 for c in calls),
        completed_native_api_replies=1,
        native_memory_selection_qualified=1,
        native_public_original_grants_qualified=1,
        configured_token_budget_qualified=all(b["actual_fits"] for b in budgets),
        full_native_case_qualified=0,
        strict_semantic_answer_qualified=0,
        observed_role_preference_unknown_preserved=False,
        actual_call_budgets=budgets,
        prepared_history_messages=16,
        canonical_history_messages=len(kept),
        raw_api_reply=result["response"]["reply"],
        actual_generation_receipt=result["generation"][-1],
        actual_scope=scope,
        candidate_tests=dict(new=13, directly_affected_old=4, unique=17, executions=17),
        cloud_calls_during_audit=0,
        actual_guard_comparisons=comparisons,
        actual_decision=result["prepared"][-1]["actual_decision"],
        strict_open_findings=[
            "Fresh raw reply asserts 我没有自己的喜好 instead of reporting no retrieved evidence "
            "of role preference, then says unknown. Absence of visible evidence is not evidence "
            "that the character has no preferences. Strict semantic/full-case qualification stays0. "
            "This model output did not trigger the old generic inability branch; report fresh "
            "native counterexample trigger coverage0, not proof that this semantic finding is fixed. "
            "Preserve raw reply and previous stage71/77/78/79 findings without style forcing or resampling."
        ],
    )
    (phase / "native-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                evidence_checks=f"{report['passed']}/{report['total']}",
                failed_checks=[n for n, v in checks.items() if not v],
                phase_calls=len(calls),
                budget_qualified=report["configured_token_budget_qualified"],
                full_native_case_qualified=0,
            ),
            ensure_ascii=False,
        )
    )
    assert all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    asyncio.run(run(parser.parse_args()))
