"""Read-only receipts for a complete reported-action versus preference case."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from html import unescape
from pathlib import Path


async def run(phase):
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    assert phase.parent == runtime / "backups/backend-chain-20261001"
    assert phase.name in {"stage83", "stage84", "stage85", "stage86"}
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
    result = read("native-pg-report/result.json")
    calls = read("native-pg-report/cloud-calls.json")
    gate = read("native-pg-report/primary-input-observed.json")
    pre = read("preflight.json")
    review = read("semantic-review.json")
    checks = {}
    document_count = 5 if phase.name in {"stage85", "stage86"} else 4

    def check(name, value):
        checks[name] = bool(value)

    check(
        "new_complete_case_preserves_original_question_and_all_full_sources",
        case["question"].startswith(case["original_question"] + "\n")
        and len(case["documents"]) == document_count
        and len(case["bridges"]) == 9
        and not case["input_information_reduced"]
        and case["current_user_action_report"] in case["question"]
        and not case["reported_action_verified_by_tool"],
    )
    check(
        "actual_before_question_dump_and_zero_prior_answers",
        hashlib.sha256((phase / "native-pg-seed/before-question.dump").read_bytes()).hexdigest()
        == seed["before_question_backup"]["database_sha256"]
        and (phase.name in {"stage85", "stage86"} or seed["before_question_backup"]["database_sha256"]
             == "00416458b561625a1db2be7ebae265ea4dffdfa5b8efee9c312479a71b6317c6")
        and seed["before_question_backup"]["prior_same_task_answers"] == 0
        and result["before_question_backup"] == seed["before_question_backup"],
    )
    check(
        "actual_authenticated_postgresql_native_api_grounded_reply",
        result["transport"] == "authenticated_ASGI"
        and result["database_mode"] == "PostgreSQL"
        and result["http_status"] == 200
        and not result["response"]["abstained"]
        and result["auth_statuses"] == result["chat_auth_statuses"] == [200, 200],
    )
    check(
        "successful_writer_import_and_nine_history_tasks_not_replayed",
        result["knowledge_imports_replayed"]
        == result["seed_model_calls_replayed"]
        == result["successful_bridge_turns_replayed"]
        == 0
        and result["bridge_turns_completed"] == 9,
    )
    prepared = result["prepared"][-1]
    check(
        "three_real_original_preference_claims_selected_as_observations",
        result["seed_records"] == seed["seed_records"]
        and len(result["seed_records"]) == 3
        and prepared["used_memory_ids"] == ["42", "41", "40"]
        and prepared["recall"]["records_read"]
        == prepared["recall"]["usable_records"]
        == prepared["recall"]["selected_count"]
        == 3
        and prepared["recall"]["temporal_views"] == {"fact": 0, "asserted_state": 0, "observation": 3},
    )
    check(
        "whole_private_source_negation_and_and_use_condition_not_rewritten",
        all(
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
        "four_real_official_pro_http200_stop_requests_one_primary",
        len(calls) == 4
        and len(primary) == result["primary_calls"] == 1
        and all(
            c["http_status"] == 200
            and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
    )
    raw = primary[0]["response"]["choices"][0]["message"]["content"]
    wire = primary[0]["request"]["messages"]
    whole = "\n".join(unescape(x["content"]) for x in wire)
    check(
        "raw_reply_neither_rewritten_nor_guard_retried",
        raw == result["response"]["reply"]
        and len(result["generation"]) == 1
        and result["generation"][0]["observed_primary_calls"] == 1
        and not result["generation"][0]["guard_retried"]
        and not result["generation"][0]["guard_fallback"],
    )
    check(
        "actual_wire_contains_every_full_original_and_full_new_user_report",
        all(gate["document_bodies_present"].values())
        and gate["private_source_present"]
        and case["source_message"] in whole
        and case["question"] in whole
        and case["current_user_action_report"] in whole
        and all(x["content"] in whole for x in case["documents"]),
    )
    final_user = unescape(wire[-1]["content"])
    speech = json.loads(re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", final_user, re.S)[1])
    check(
        "json_decoded_ten_whole_historical_originals_not_false_substring_omission",
        len(speech["records"]) == 10
        and speech["speaker_role"] == "user"
        and speech["described_subject"] == speech["current_validity"] == "not_resolved"
        and any(x["text"] == case["source_message"] for x in speech["records"])
        and all(any(row["text"] == x["message"].strip() for row in speech["records"]) for x in case["bridges"]),
    )
    check(
        "trusted_attribution_history_scope_and_actual_source_visibility_present",
        all(
            x in wire[0]["content"]
            for x in ["【用户历史依据范围】", "【本轮召回的历史原话】", "长期记忆参考中的‘用户’始终指当前对话者"]
        ),
    )
    history = prepared["history"]
    check(
        "unchanged_budget_keeps_six_complete_recent_native_turns",
        len(history) == 16
        and wire[1:-1] == history[-12:]
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

    budgets = [
        dict(
            actual=c["response"]["usage"]["prompt_tokens"],
            official=official_full(c["request"]["messages"]),
            bound=sum(count(x["content"]) + 4 for x in c["request"]["messages"]),
            reserved=c["request"]["max_tokens"],
            margin=512,
            window=65536,
        )
        for c in calls
    ]
    check(
        "all_actual_api_token_counts_match_independent_official_encoding",
        all(x["actual"] == x["official"] for x in budgets),
    )
    check(
        "all_actual_pre_send_bounds_cover_model_and_fit_unchanged_serving_budget",
        all(
            x["bound"] == c["budget"]["input_bound"]
            and x["bound"] >= x["actual"]
            and x["bound"] + x["reserved"] + 512 <= 65536
            for x, c in zip(budgets, calls)
        ),
    )
    database = result["database"]
    assert database == "stage3_" + phase.name + "_native_pg_report"
    con = await asyncpg.connect(user="boot", database=database, host=str(cluster / "socket"), port=25433)
    try:
        async with con.transaction(readonly=True):
            assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
            docs = [
                dict(x)
                for x in await con.fetch(
                    "SELECT id,title,content,category,knowledge_base_id FROM knowledge_documents "
                    "WHERE id=ANY($1) ORDER BY id",
                    ([504, 505, 506, 507, 508] if phase.name in {"stage85", "stage86"} else [504, 505, 506, 507]),
                )
            ]
            links = [
                dict(x)
                for x in await con.fetch(
                    "SELECT l.memory_id,s.body,s.owner_key,s.scope_key,s.state FROM memory_source_links l "
                    "JOIN memory_sources s ON l.source_key=s.source_key "
                    "WHERE l.memory_id=ANY($1::bigint[]) ORDER BY l.memory_id",
                    [40, 41, 42],
                )
            ]
            messages = [
                dict(x)
                for x in await con.fetch(
                    'SELECT message,reply FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 ORDER BY id',
                    "web",
                    "web-character",
                    "6",
                )
            ]
    finally:
        await con.close()
    check(
        "actual_database_four_complete_distinct_public_kb_originals",
        len(docs) == document_count
        and {x["knowledge_base_id"] for x in docs} == {3, 4, 5, 6}
        and all(
            any(x["title"] == d["title"] and x["content"] == d["content"] for x in docs) for d in case["documents"]
        ),
    )
    check(
        "actual_database_three_original_user_owner_links_and_full_source",
        len(links) == 3
        and all(
            x["body"] == case["source_message"]
            and x["state"] == "recorded"
            and x["owner_key"] == json.dumps(("web", "web-character", "6"))
            for x in links
        ),
    )
    check(
        "actual_database_nine_original_archive_turns_and_only_one_new_question",
        len(messages) == 11
        and [x["message"] for x in messages[1:-1]] == [x["message"].strip() for x in case["bridges"]]
        and messages[-1]["message"] == case["question"]
        and messages[-1]["reply"] == raw,
    )
    plan = result["generation"][0]["retrieval"]
    by_id = {x["id"]: x for x in docs}
    grants = []
    for coverage in plan["source_coverage"]:
        receipt = coverage["original_source_receipt"]
        doc = by_id[receipt["document_id"]]
        packet = next(
            (x for x in plan["evidence_packets"] if x.get("document_ids") == [receipt["original_packet_id"]]), None
        )
        grants.append(
            packet is not None
            and coverage["original_status"] == "verified_original_body_admitted"
            and receipt["original_body_chars"] == len(doc["content"])
            and receipt["original_body_sha256"] == hashlib.sha256(doc["content"].encode()).hexdigest()
            and packet["original_body"] == doc["content"]
            and packet["text"] in final_user
        )
    check("four_database_verified_original_grants_at_actual_primary_wire", len(grants) == document_count and all(grants))
    check(
        "runtime_hashes_match_tested_candidate_and_original_configs",
        all(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
            for name, sha in {**pre["original_hashes"], **pre["config_hashes"],
                             **({name: (read("targeted-test-summary.json")["published_fixture_observation_only_sha256"] if name.endswith("deepseek_known_role_source_case.json") else sha) for name, sha in read("targeted-test-summary.json")["tested_hashes"].items()} if phase.name == "stage85" else {})}.items()
        ),
    )
    pid = read("native-report-job.json")["pid"]
    try:
        live = bool(Path(f"/proc/{pid}/cmdline").read_bytes()) and (
            Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
        )
    except OSError:
        live = False
    check("actual_native_job_terminal", not live)
    check(
        "manual_review_quotes_bound_to_exact_actual_raw_answer",
        review["raw_reply_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        and review["case_question_sha256"] == hashlib.sha256(case["question"].encode()).hexdigest()
        and all(x["quotes"] and all(q in raw for q in x["quotes"]) for x in review["criteria"])
        and review["prior_phase82_case_retested"] is False,
    )
    if phase.name in {"stage84", "stage85", "stage86"}:
        check(
            "new_current_and_condition_satisfied_while_actual_use_explicitly_denied",
            case["current_state"]["form_submitted"] is True
            and case["current_state"]["identity_verification_passed"] is True
            and case["expected_accelerated_use_condition_satisfied"] is True
            and case["expected_actual_accelerated_use_from_user_report"] is False
            and "没有实际使用窗口受理、加速受理或寄件受理" in case["question"]
            and "身份核验失败" not in case["question"]
            and "在本轮核验失败时" not in case["question"],
        )
        check(
            "assistant_review_correctly_attributed_without_human_review_claim",
            review["assessor"] == "assistant"
            and review["human_review_performed"] is False
            and review["prior_phase83_case_retested"] is False,
        )
    if phase.name == "stage85":
        public_case = json.loads(Path("backend/tests/fixtures/deepseek_known_role_source_case.json").read_text())
        public_case.pop("observed_semantic_review")
        public_case.pop("actual_results")
        check("published_observations_do_not_change_executed_case_input",
              hashlib.sha256((json.dumps(public_case, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
              == read("targeted-test-summary.json")["tested_hashes"]["backend/tests/fixtures/deepseek_known_role_source_case.json"])
        diagnostic = read("role-seed-diagnostic.json")
        check("new_role_source_authenticated_without_provider_or_user_memory_injection",
              diagnostic["login_status"] == diagnostic["new_import_status"] == 200
              and diagnostic["new_document_id"] == 508
              and diagnostic["cloud_attempts"] == diagnostic["model_calls"] == 0
              and len(diagnostic["claims"]) == len(result["user_fact_records_after"]) == 3
              and seed["source_layout_parent_backup_sha256"]
              == "00416458b561625a1db2be7ebae265ea4dffdfa5b8efee9c312479a71b6317c6")
        check("actual_same_index_baseline_zero_originals_fixed_all_five",
              diagnostic["whole_originals_present"] == [False] * 5
              and read("actual-api-retrieval-after.json")["whole_originals_present"] == [True] * 5)
    if phase.name == "stage86":
        diagnostic = read("role-seed-diagnostic.json")
        before = read("before-revision-retrieval.json")
        after = read("seed-retrieval.json")
        old_body = case["previous_role_preference_source"]["content"]
        new_body = case["role_preference_source"]["content"]
        check("real_authenticated_same_document_revision_after_old_cache_warm",
              diagnostic["login_status"] == diagnostic["new_import_status"] == 200
              and diagnostic["mutation_kind"] == "actual_authenticated_same_doc_content_update"
              and diagnostic["new_document_id"] == 508
              and diagnostic["old_retrieval_warmed"] is True
              and any(x.get("original_body") == old_body for x in before["original_source_packets"]))
        check("actual_after_revision_retrieval_and_wire_admit_only_new_whole_role_body",
              diagnostic["whole_originals_present"] == [True] * 5
              and any(x.get("original_body") == new_body for x in after["original_source_packets"])
              and new_body in whole and old_body not in whole
              and diagnostic["previous_role_body_sha256"] != diagnostic["current_role_body_sha256"])
        check("same_role_revision_setup_no_provider_or_user_memory_injection",
              diagnostic["cloud_attempts"] == diagnostic["model_calls"] == 0
              and len(diagnostic["claims"]) == len(result["user_fact_records_after"]) == 3
              and seed["source_layout_parent_backup_sha256"]
              == "811c457bf9479250479da8e1afc580ce0ae5abd743e20265d5a8722b113e9f55")
    review_passed = all(x["passed"] for x in review["criteria"])
    report = dict(
        checks=checks,
        passed=sum(checks.values()),
        total=len(checks),
        chain_receipts_qualified=all(checks.values()),
        phase_calls=len(calls),
        http200=sum(c["http_status"] == 200 for c in calls),
        actual_primary_calls=len(primary),
        completed_native_api_replies=1,
        actual_call_budgets=budgets,
        main_actual_prompt_tokens=budgets[-1]["actual"],
        main_input_bound=budgets[-1]["bound"],
        new_case_semantic_criteria=f"{sum(x['passed'] for x in review['criteria'])}/{len(review['criteria'])}",
        full_native_case_qualified=int(all(checks.values()) and review_passed),
        prior_phase82_full_native_case_qualified=0,
        prior_phase81_full_native_case_qualified=1,
        prior_phase83_full_native_case_qualified=(1 if phase.name in {"stage84", "stage85", "stage86"} else None),
        semantic_review_assessor=review.get("assessor", "assistant"),
        prior_phase85_full_native_case_qualified=(1 if phase.name == "stage86" else None),
        current_role_revision_receipts_qualified=(all(checks.values()) if phase.name == "stage86" else None),
        runtime_code_changed=(phase.name == "stage85"),
        production_restarted=False,
        audit_provider_calls=0,
        unproven_prior_semantic_root_cause="unattributed",
        source_trace_difference_is_preparation_vs_canonical_budget=True,
        raw_api_reply=raw,
        review_limitations=review["limitations"],
    )
    (phase / "native-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"raw_api_reply", "checks", "actual_call_budgets", "review_limitations"}
            },
            ensure_ascii=False,
        )
    )
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, type=Path)
    asyncio.run(run(parser.parse_args().phase.resolve()))
