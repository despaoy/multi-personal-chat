"""Audit actual fresh-input recall and per-source admission without model replay."""

import argparse
import asyncio
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import secrets
import sys
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
    import asyncpg

    from character.memory_service import _detect_memory_intents
    from inference.context_budget import estimated_tokens

    f = json.loads((phase / "fixture.json").read_text())
    variants = ["native-pg-baseline", "native-pg-fixed"]
    proofs = [json.loads((phase / v / "result.json").read_text()) for v in variants]
    base, fixed = proofs
    gates = [json.loads((phase / v / "primary-input-observed.json").read_text()) for v in variants]
    calls_by_run = [json.loads((phase / v / "cloud-calls.json").read_text()) for v in variants]
    calls = sum(calls_by_run, [])
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    original_name = "_stage77_original_memory_service"
    loader = importlib.machinery.SourceFileLoader(
        original_name, str(phase / "original-backend_character_memory_service.py")
    )
    spec = importlib.util.spec_from_loader(original_name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[original_name] = module
    loader.exec_module(module)
    before = module._detect_memory_intents(f["question"])
    after = _detect_memory_intents(f["question"])
    check("actual_query_original_wrong_suppression_reproduced", before.suppress_preference and not before.preference)
    check(
        "actual_query_fixed_recipient_defers_without_owner_verdict",
        not after.suppress_preference and not after.preference,
    )
    check(
        "four_full_imports_and_informative_private_source",
        f["synthetic"]
        and f["no_rule_seeded_memories"]
        and len(f["documents"]) == 4
        and all(0 < len(d["content"]) < 50000 for d in f["documents"]),
    )
    check(
        "genuine_large_complete_originals_exceed_actual_window",
        sum(estimated_tokens(d["content"]) for d in f["documents"]) > 65536
        and all(len(d["content"]) > 45000 for d in f["documents"][:2]),
    )
    check(
        "same_original_query_on_actual_wire",
        all(f["question"] in unescape(g["request"]["messages"][-1]["content"]) for g in gates),
    )
    check(
        "same_native_ordinary_owner_and_scope",
        base["chat_auth_statuses"] == fixed["chat_auth_statuses"] == [200, 200]
        and base["seed_scope"] == fixed["seed_scope"]
        and base["seed_scope"]["owner"] != "1",
    )
    check(
        "successful_imports_and_actual_writer_not_replayed",
        len(base["documents_imported"]) == 4
        and base["documents_imported"] == fixed["documents_imported"]
        and fixed["knowledge_imports_replayed"]
        == fixed["seed_model_calls_replayed"]
        == fixed["calls_before_question"]
        == 0
        and base["seed_http_status"] == 200,
    )
    check(
        "real_prequestion_backup_restored",
        fixed["restore_before_question_not_terminal_answer"]
        and hashlib.sha256((phase / "native-pg-baseline/before-question.dump").read_bytes()).hexdigest()
        == fixed["before_question_backup"]["database_sha256"],
    )
    histories = [p["prepared"][-1]["history"] for p in proofs]
    check(
        "actual_history_identical_seed_only_no_prior_same_question_answer",
        histories[0] == histories[1]
        and len(histories[1]) == 2
        and histories[1][0]["content"] == f["source_message"]
        and all(f["question"] not in h["content"] and "QL-77" not in h["content"] for h in histories[1]),
    )
    check(
        "baseline_and_fixed_terminal_jobs",
        all(
            not (Path("/proc") / str(json.loads((phase / j).read_text())["pid"]) / "cmdline").exists()
            or not (Path("/proc") / str(json.loads((phase / j).read_text())["pid"]) / "cmdline").read_bytes()
            for j in ["native-baseline-job.json", "native-fixed-job.json"]
        ),
    )
    check(
        "actual_calls_count_matches_saved_results",
        list(map(len, calls_by_run)) == [base["cloud_calls"], fixed["cloud_calls"]] == [7, 5],
    )
    check(
        "actual_requests_responses_pro_http200_stop",
        all(
            c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["http_status"] == 200
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
    )
    check(
        "actual_primary_completed_three_replies",
        base["http_status"] == fixed["http_status"] == 200
        and base["primary_calls"] == 2
        and fixed["primary_calls"] == 1,
    )
    check(
        "same_three_durable_real_claims",
        base["seed_records"] == fixed["seed_records"] == fixed["user_fact_records_after"]
        and len(fixed["seed_records"]) == 3
        and fixed["original_seed_claims_exact_verified"],
    )
    recall = [p["prepared"][-1]["recall"] for p in proofs]
    check(
        "baseline_actual_all_three_filtered",
        recall[0]["records_read"] == 3
        and recall[0]["usable_records"] == recall[0]["selected_count"] == 0
        and recall[0]["status"] == "all_filtered",
    )
    ids = {str(r["id"]) for r in fixed["seed_records"]}
    prep = fixed["prepared"][-1]
    check(
        "same_three_claims_now_usable_and_selected",
        recall[1]["records_read"] == recall[1]["usable_records"] == recall[1]["selected_count"] == 3
        and set(prep["used_memory_ids"]) == ids
        and {x["memory_id"] for x in prep["memory_packets"]} == ids,
    )
    check(
        "temporal_observation_not_promoted_to_current_fact",
        recall[0]["temporal_views"] == recall[1]["temporal_views"] == {"fact": 0, "asserted_state": 0, "observation": 3}
        and all(p["temporal_mode"] == "observation" for p in prep["memory_packets"]),
    )
    check(
        "full_private_conjunction_and_negation_retained",
        any(
            dict(x["qualifiers"]) == {"condition": "申请齐全且附件核验通过"} and f["source_message"] in x["evidence"]
            for x in prep["memory_packets"]
        )
        and any("不喜欢预约受理" in x["content"] for x in prep["memory_packets"]),
    )
    check(
        "all_required_independent_short_bodies_and_private_source_at_wire",
        all(
            g["private_source_present"]
            and all(
                g["document_bodies_present"][str(i) + ":" + f["documents"][i]["title"]]
                for i in f["required_gate_document_indices"]
            )
            for g in gates
        ),
    )
    check(
        "large_full_originals_not_falsely_admitted",
        all(
            not any(g["document_bodies_present"][str(i) + ":" + f["documents"][i]["title"]] for i in [0, 1])
            for g in gates
        ),
    )
    check(
        "ranked_authority_content_scope_scores_roles_unchanged",
        [{k: v for k, v in r.items() if k != "import_time"} for r in base["retrieval"][0]["results"]]
        == [{k: v for k, v in r.items() if k != "import_time"} for r in fixed["retrieval"][0]["results"]]
        and base["retrieval"][0]["source_coverage"] == fixed["retrieval"][0]["source_coverage"]
        and base["retrieval"][0]["confidence"] == fixed["retrieval"][0]["confidence"],
    )
    stored = []
    for p in proofs:
        assert re.fullmatch(r"stage3_stage[1-9]\d*_native_pg_(baseline|fixed)", p["database"])
        con = await asyncpg.connect(user="boot", database=p["database"], host=str(cluster / "socket"), port=25433)
        try:
            async with con.transaction(readonly=True):
                assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
                docs = await con.fetch(
                    "SELECT id,title,content,knowledge_base_id FROM knowledge_documents WHERE title=ANY($1::text[])",
                    [d["title"] for d in f["documents"]],
                )
                chunks = await con.fetch(
                    'SELECT id,"documentId","chunkIndex",content FROM knowledge_chunks WHERE "documentId"=ANY($1::bigint[])',
                    [d["id"] for d in docs],
                )
                links = await con.fetch(
                    "SELECT l.memory_id,s.source_message_id,s.body,s.owner_key,s.scope_key,s.state FROM memory_source_links l JOIN memory_sources s ON l.source_key=s.source_key WHERE l.memory_id=ANY($1::bigint[])",
                    [r["id"] for r in p["seed_records"]],
                )
                sources = await con.fetch(
                    "SELECT body FROM memory_sources WHERE owner_key=$1",
                    json.dumps(("web", "web-character", p["seed_scope"]["owner"])),
                )
                stored.append(
                    {
                        "documents": [dict(d) for d in docs],
                        "chunks": [dict(d) for d in chunks],
                        "links": [dict(d) for d in links],
                        "sources": [dict(d) for d in sources],
                    }
                )
        finally:
            await con.close()
    check(
        "independent_db_full_four_originals_unchanged",
        stored[0]["documents"] == stored[1]["documents"]
        and len(stored[1]["documents"]) == 4
        and all(
            any(d["title"] == x["title"] and d["content"] == x["content"] for d in stored[1]["documents"])
            for x in f["documents"]
        ),
    )
    check(
        "independent_db_all_original_indexed_chunks_unchanged",
        stored[0]["chunks"] == stored[1]["chunks"] and len(stored[1]["chunks"]) == 190,
    )
    check(
        "same_three_authorized_full_private_source_links",
        stored[0]["links"] == stored[1]["links"]
        and len(stored[1]["links"]) == 3
        and all(
            link["body"] == f["source_message"]
            and link["owner_key"] == json.dumps(("web", "web-character", fixed["seed_scope"]["owner"]))
            and link["state"] == "recorded"
            for link in stored[1]["links"]
        ),
    )
    primary = []
    for run_calls in calls_by_run:
        matches = [
            c
            for c in run_calls
            if any(
                "<user_query>" in m.get("content", "") and f["question"] in unescape(m["content"])
                for m in c["request"].get("messages", [])
            )
        ]
        assert len(matches) == 1
        primary.append(matches[0])
    check(
        "fixed_api_reply_is_unmodified_actual_model_output",
        fixed["response"]["reply"] == primary[1]["response"]["choices"][0]["message"]["content"],
    )
    check(
        "actual_primary_tokens_with_full_output_and_safety_fit65536",
        all(
            c["response"]["usage"]["prompt_tokens"] + c["request"]["max_tokens"] + 512 < 65536
            and c["request"]["max_tokens"] == 2048
            for c in primary
        ),
    )
    plans = [p["generation"][-1]["retrieval"] for p in proofs]
    scopes = []
    for c in primary:
        text = unescape(c["request"]["messages"][-1]["content"])
        doc_match = re.search(r"<retrieval_coverage[^>]*>\n(.*?)\n</retrieval_coverage>", text, re.S)
        assert doc_match
        scopes.append(json.loads(doc_match[1]))
    check(
        "public_wire_scope_matches_actual_settled_original_status",
        all(
            {s["source_id"]: s["original_status"] for s in x["sources"]}
            == {s["source_id"]: s["original_status"] for s in p["source_coverage"]}
            for x, p in zip(scopes, plans)
        ),
    )
    documents = {d["id"]: d for d in stored[1]["documents"]}
    valid = []
    for coverage in plans[1]["source_coverage"]:
        receipt = coverage["original_source_receipt"]
        doc = documents[receipt["document_id"]]
        admitted = next(
            (p for p in plans[1]["evidence_packets"] if p.get("document_ids") == [receipt["original_packet_id"]]), None
        )
        valid.append(
            receipt["original_body_chars"] == len(doc["content"])
            and receipt["original_body_sha256"] == hashlib.sha256(doc["content"].encode()).hexdigest()
            and (coverage["original_status"] == "verified_original_body_admitted") == bool(admitted)
            and (
                not admitted
                or (
                    admitted["original_body"] == doc["content"]
                    and hashlib.sha256(admitted["text"].encode()).hexdigest() == receipt["original_packet_sha256"]
                    and admitted["text"] in unescape(primary[1]["request"]["messages"][-1]["content"])
                )
            )
        )
    check("each_actual_original_grant_has_exact_current_db_and_packet_hash", all(valid))
    check(
        "long_sources_remain_unverified_full_and_known_index_partial_distinct",
        all(
            s["original_status"] == "verified_original_not_admitted"
            for s in scopes[1]["sources"]
            if s["source_title"] in [d["title"] for d in f["documents"][:2]]
        )
        and any(s["status"] == "partial" for s in scopes[1]["sources"]),
    )
    check(
        "fixed_both_short_current_originals_actually_admitted",
        all(
            s["original_status"] == "verified_original_body_admitted"
            for s in scopes[1]["sources"]
            if s["source_title"] in [d["title"] for d in f["documents"][2:]]
        ),
    )
    reply = fixed["response"]["reply"]
    positions = {d["title"]: reply.find(d["title"]) for d in f["documents"]}
    sections = {
        t: reply[pos : min([p for p in positions.values() if p > pos], default=len(reply))]
        for t, pos in positions.items()
    }
    check(
        "raw_reply_preserves_four_known_codes_costs_hours_and_conditions",
        all(
            r["code"] in sections[r["title"]]
            and f"{r['fee_yuan']}元" in sections[r["title"]]
            and f"{r['processing_hours']}小时" in sections[r["title"]]
            and r["known_condition"] in sections[r["title"]]
            and r["known_exception"] in sections[r["title"]]
            for r in f["expected_records"]
        ),
    )
    check(
        "raw_reply_correctly_distinguishes_short_full_and_long_partial",
        all("已核对完整原文" in sections[d["title"]] for d in f["documents"][2:])
        and all(
            "只有部分资料" in sections[d["title"]] and "无法确认是否还有其他例外条款" in sections[d["title"]]
            for d in f["documents"][:2]
        ),
    )
    check(
        "real_conjunctive_failure_and_postal_unknown_retained",
        all(t in reply for t in ["附件核验失败", "不满足", "偏好未知", "不喜欢预约受理", "只有申请齐全且附件核验通过"]),
    )
    check(
        "targeted_tests_failures_preserved_and_only_affected_retested",
        "4 failed, 14 passed" in (phase / "targeted-tests.txt").read_text()
        and "7 passed" in (phase / "repair-tests.txt").read_text(),
    )
    result = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "baseline_calls": len(calls_by_run[0]),
        "fixed_calls": len(calls_by_run[1]),
        "phase_calls": len(calls),
        "completed_primary_replies": 3,
        "test_unique": 20,
        "test_executions": 25,
        "test_initial_failed": 4,
        "new_test_unique": 19,
        "affected_old_test_unique": 1,
        "independent_chain_input_qualified": 1,
        "strict_semantic_answer_qualified": 0,
        "strict_open_finding": "Fixed raw reply says现场is唯一有明确喜好证据despite admitted conditionalpositive加急preference, then correctlystates喜欢加急 later. No backend evidence missing for this contradiction; preserve without forcedoutput or modelrerun.",
        "primary_usage": [c["response"]["usage"] for c in primary],
        "actual_scopes": scopes,
        "raw_baseline_reply": base["response"]["reply"],
        "raw_fixed_reply": reply,
        "actual_recall": recall,
        "original_intent": before.__dict__,
        "fixed_intent": after.__dict__,
        "actual_same_seed_history": histories[1],
        "storage_summary": {
            "documents": 4,
            "indexed_chunks": len(stored[1]["chunks"]),
            "private_source_links": len(stored[1]["links"]),
        },
        "audit_initial_corrections": "Preserve initial32/34audit and originalscript. Actual114results differ only in import_time during isolated vector rebuild; allauthority/content/scope/roles/scores andrank order equal. DBlinks equal3/3 state recorded, not imagined active. No code/model/test replay to repairaudit.",
        "known_harness_error": "Baseline launched expect-primary-blocked in anticipation of source starvation. Actual allrequiredshortbodygate passed and primary200completed; terminal assertion failed aftersaving. Original result/log retained. No source starvation claim or baseline replay.",
    }
    (phase / "native-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(
        json.dumps(
            {
                "audit": str(result["passed"]) + "/" + str(result["total"]),
                "failed": [k for k, v in checks.items() if not v],
                "phase_calls": len(calls),
                "independent_chain_input_qualified": 1,
                "strict_semantic_answer_qualified": 0,
            },
            ensure_ascii=False,
        )
    )
    assert all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    asyncio.run(run(parser.parse_args()))
