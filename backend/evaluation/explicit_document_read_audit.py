"""Audit complete native multi-document reads and scoped personal evidence without model replay."""

import argparse
import asyncio
import json
import os
import re
import secrets
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

    fixture = json.loads((phase / "fixture.json").read_text())
    proofs = [json.loads((phase / v / "result.json").read_text()) for v in ["native-pg-baseline", "native-pg-fixed"]]
    baseline, fixed = proofs
    calls_by_run = [
        json.loads((phase / v / "cloud-calls.json").read_text()) for v in ["native-pg-baseline", "native-pg-fixed"]
    ]
    calls = sum(calls_by_run, [])
    gates = [
        json.loads((phase / v / "primary-input-observed.json").read_text())
        for v in ["native-pg-baseline", "native-pg-fixed"]
    ]
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check(
        "complete_synthetic_four_docs_and_private_source",
        fixture["synthetic"]
        and len(fixture["documents"]) == 4
        and len(fixture["expected_records"]) == 4
        and fixture["no_rule_seeded_memories"],
    )
    check(
        "same_full_query_at_both_actual_wire_gates",
        all(fixture["question"] in unescape(g["request"]["messages"][-1]["content"]) for g in gates),
    )
    check(
        "same_auth_ordinary_owner_and_scope",
        baseline["chat_auth_statuses"] == fixed["chat_auth_statuses"] == [200, 200]
        and baseline["seed_scope"] == fixed["seed_scope"]
        and baseline["seed_scope"]["owner"] != "1",
    )
    check(
        "four_actual_native_imports_once",
        len(baseline["documents_imported"]) == 4
        and baseline["documents_imported"] == fixed["documents_imported"]
        and fixed["knowledge_imports_replayed"] == 0,
    )
    check(
        "successful_real_writer_seed_not_replayed",
        baseline["seed_http_status"] == 200
        and fixed["seed_template_verified"]
        and fixed["seed_model_calls_replayed"] == fixed["calls_before_question"] == 0
        and fixed["original_seed_claims_exact_verified"],
    )
    check("six_baseline_five_fixed_actual_calls", list(map(len, calls_by_run)) == [6, 5])
    check(
        "all_actual_calls_pro_200_stop",
        all(
            c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
            and c["http_status"] == 200
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
    )
    check(
        "baseline_query_blocked_before_model",
        baseline["http_status"] == 500
        and baseline["primary_calls"] == 1
        and all(
            "<user_query>" not in m.get("content", "") or fixture["question"] not in unescape(m.get("content", ""))
            for c in calls_by_run[0]
            for m in c["request"]["messages"]
        ),
    )
    check("fixed_original_query_actual_primary_once", fixed["http_status"] == 200 and fixed["primary_calls"] == 1)
    check(
        "whole_private_source_at_both_gates",
        all(
            g["private_source_present"]
            and fixture["source_message"] in unescape(g["request"]["messages"][-1]["content"])
            for g in gates
        ),
    )
    check("baseline_missing_oat_only", list(gates[0]["document_bodies_present"].values()) == [False, True, True, True])
    check(
        "fixed_all_four_complete_bodies",
        all(gates[1]["document_bodies_present"].values()) and len(gates[1]["document_bodies_present"]) == 4,
    )
    check(
        "actual_ranked_three_unchanged_plus_requested_root",
        fixed["retrieval"][0]["results"][:3] == baseline["retrieval"][0]["results"]
        and len(fixed["retrieval"][0]["results"]) == 4
        and fixed["retrieval"][0]["results"][3]["retrieval_role"] == "requested_source",
    )
    check(
        "confidence_and_abstention_not_boosted",
        fixed["retrieval"][0]["confidence"] == baseline["retrieval"][0]["confidence"]
        and not fixed["retrieval"][0]["abstained"]
        and fixed["retrieval"][0]["results"][3]["score"] == 0,
    )
    check(
        "explicit_scope_and_no_unresolved_titles",
        fixed["retrieval"][0]["requested_source_titles"] == [d["title"] for d in fixture["documents"]]
        and fixed["retrieval"][0]["requested_source_scope"] == "same_anchor_knowledge_base_and_original_filter"
        and fixed["retrieval"][0]["unresolved_requested_titles"] == [],
    )
    check(
        "four_indexed_scope_counts_complete",
        len(fixed["retrieval"][0]["source_coverage"]) == 4
        and all(
            len(r["indexed_document_ids"]) == len(r["retrieved_document_ids"]) == 1
            for r in fixed["retrieval"][0]["source_coverage"]
        ),
    )
    check(
        "same_three_real_seed_claims_and_no_rag_promotion",
        len(baseline["seed_records"]) == 3
        and baseline["seed_records"] == fixed["seed_records"] == fixed["user_fact_records_after"]
        and fixed["scheduler_final"]["saved"] == 0,
    )
    stored = []
    for proof in proofs:
        assert re.fullmatch(r"stage3_stage[1-9]\d*_native_pg_(baseline|fixed)", proof["database"])
        con = await asyncpg.connect(user="boot", database=proof["database"], host=str(cluster / "socket"), port=25433)
        try:
            async with con.transaction(readonly=True):
                assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
                owner = proof["seed_scope"]["owner"]
                owner_key = json.dumps(("web", "web-character", owner))
                scope_key = json.dumps(("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner))
                rows = await con.fetch(
                    "SELECT * FROM character_memories WHERE character_id=$1 AND platform=$2 AND adapter=$3 AND sender_id=$4 AND conversation_type=$5 AND conversation_id=$6",
                    "tsukiyashiro_kisaki",
                    "web",
                    "web-character",
                    owner,
                    "private",
                    owner,
                )
                links = await con.fetch(
                    "SELECT l.memory_id,s.source_message_id,s.body,s.owner_key,s.scope_key,s.state FROM memory_source_links l JOIN memory_sources s ON l.source_key=s.source_key WHERE l.memory_id=ANY($1::bigint[])",
                    [r["id"] for r in rows],
                )
                docs = await con.fetch(
                    "SELECT id,title,content,knowledge_base_id FROM knowledge_documents WHERE title=ANY($1::text[])",
                    [d["title"] for d in fixture["documents"]],
                )
                chunks = await con.fetch(
                    'SELECT id,"documentId","chunkIndex",content FROM knowledge_chunks WHERE "documentId"=ANY($1::bigint[])',
                    [d["id"] for d in docs],
                )
                stored.append(
                    {
                        "database": proof["database"],
                        "data_directory": str(cluster / "data"),
                        "owner_key": owner_key,
                        "scope_key": scope_key,
                        "rows": [dict(r) for r in rows],
                        "links": [dict(r) for r in links],
                        "documents": [dict(r) for r in docs],
                        "chunks": [dict(r) for r in chunks],
                    }
                )
        finally:
            await con.close()
    check(
        "independent_exact_same_three_claims",
        len(stored[0]["rows"]) == len(stored[1]["rows"]) == 3
        and sorted(stored[0]["rows"], key=lambda r: r["id"]) == sorted(stored[1]["rows"], key=lambda r: r["id"]),
    )
    check(
        "independent_three_own_live_full_source_links",
        all(
            len(s["links"]) == 3
            and all(
                link["body"] == fixture["source_message"]
                and link["state"] == "recorded"
                and link["owner_key"] == s["owner_key"]
                and link["scope_key"] == s["scope_key"]
                for link in s["links"]
            )
            for s in stored
        ),
    )
    check(
        "independent_full_four_db_document_bodies",
        all(
            len(s["documents"]) == 4
            and {d["title"]: d["content"] for d in s["documents"]}
            == {d["title"]: d["content"] for d in fixture["documents"]}
            for s in stored
        ),
    )
    check(
        "independent_four_single_chunks_and_authorized_base",
        all(
            len(s["chunks"]) == 4
            and all(
                d["knowledge_base_id"] == fixed["retrieval"][0]["results"][0]["knowledge_base_id"]
                and len([c for c in s["chunks"] if c["documentId"] == d["id"] and d["content"] in c["content"]]) == 1
                for d in s["documents"]
            )
            for s in stored
        ),
    )
    check(
        "native_final_indexed_metadata_matches_db_content",
        all(
            any(r["document_id"] == d["id"] and d["content"] in r["content"] for r in fixed["retrieval"][0]["results"])
            for d in stored[1]["documents"]
        ),
    )
    prep = fixed["prepared"][-1]
    ids = {str(r["id"]) for r in fixed["seed_records"]}
    check(
        "all_three_memories_selected_before_and_after",
        all(
            p["prepared"][-1]["recall"]["selected_count"] == 3 and set(p["prepared"][-1]["used_memory_ids"]) == ids
            for p in proofs
        ),
    )
    check(
        "no_unqualified_current_temporal_promotion",
        prep["recall"]["temporal_views"]
        == baseline["prepared"][-1]["recall"]["temporal_views"]
        == {"fact": 0, "asserted_state": 0, "observation": 3},
    )
    conditional = next(p for p in prep["memory_packets"] if dict(p["qualifiers"]))
    check(
        "whole_conjunctive_private_condition_packet",
        conditional["temporal_mode"] == "observation"
        and fixture["source_message"] in conditional["evidence"]
        and dict(conditional["qualifiers"]) == {"condition": "已经吃过早餐且当天没有胃痛"},
    )
    primary = [
        c
        for c in calls_by_run[1]
        if fixture["question"] in unescape(c["request"]["messages"][-1]["content"])
        and "<user_query>" in c["request"]["messages"][-1]["content"]
    ]
    check("one_actual_full_wire_primary", len(primary) == 1 and primary[0]["request"] == gates[1]["request"])
    main = primary[0]
    reply = fixed["response"]["reply"]
    check("raw_primary_reply_equals_api", reply == main["response"]["choices"][0]["message"]["content"])
    usage = main["response"]["usage"]
    check(
        "actual_full_wire_fits_pro65536_with_output_reserve",
        usage["prompt_tokens"] + main["request"]["max_tokens"] + 512 < 65536
        and main["request"]["max_tokens"] == fixture["primary_output_tokens"] == 2048,
    )
    values = []
    for i, record in enumerate(fixture["expected_records"]):
        segment = (
            reply.split(record["code"], 1)[1].split(fixture["expected_records"][i + 1]["code"], 1)[0]
            if i < 3
            else reply.split(record["code"], 1)[1]
        )
        values.append(
            all(
                term in segment
                for term in [
                    record["name"],
                    f"{record['caffeine_mg']} 毫克",
                    f"{record['sugar_g']} 克",
                    f"{record['price_yuan']} 元",
                    f"{record['temperature_c']} 摄氏度",
                ]
            )
        )
    check("all_six_fields_match_each_document_separately", all(values))
    coffee = reply.split("FJ-842", 1)[1].split("FJ-843", 1)[0]
    cocoa = reply.split("FJ-843", 1)[1].split("FJ-844", 1)[0]
    herbal = reply.split("FJ-844", 1)[1]
    check(
        "coffee_conjunctive_necessary_condition_not_met",
        all(t in coffee for t in ["不符合", "已吃早餐", "当天没有胃痛", "本轮当天有胃痛", "必要条件未满足"]),
    )
    check("cocoa_excluded_from_explicit_dislike", all(t in cocoa for t in ["不符合", "不喜欢高糖可可"]))
    check("unknown_like_and_no_sufficient_inference", "未知" in herbal and "不代表你本轮一定会饮用" in reply)
    check("pending_zero_saved", baseline["sync_pending_final"] == fixed["sync_pending_final"] == 0)
    check(
        "new24_affected10_only",
        "24 passed" in (phase / "new-tests.txt").read_text()
        and "10 passed" in (phase / "affected-tests.txt").read_text(),
    )
    result = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "phase_calls": len(calls),
        "baseline_calls": len(calls_by_run[0]),
        "fixed_calls": len(calls_by_run[1]),
        "primary_calls": baseline["primary_calls"] + fixed["primary_calls"],
        "qualified_complete_chain_input_queries": 1,
        "strict_semantic_answer_qualified": 0,
        "raw_fixed_reply": reply,
        "actual_prompt_tokens": usage["prompt_tokens"],
        "runtime_files_changed": 2,
        "new_unique": 24,
        "affected_unique": 10,
        "test_unique": 34,
        "test_executions": 34,
        "full_suite": False,
        "strict_stage71_failure_still_open": True,
        "limits": [
            "Closed leading explicit exact-title task only; abstained/no-anchor retrieval retains prior behavior, unresolved titles stay unknown, original scope filters and base boundary apply.",
            "Indexed full scope is not automatically raw original completeness; this synthetic case independently verifies all four native single-chunk originals.",
            "Recent history and original speech still assist; no distant-history or fact-only sufficiency claim.",
            "Raw answer says oat is not restricted by stomach pain; no source supplies general medical safety, so strict semantic correctness is not certified. Absence of a retrieved herbal preference does not prove absence from all history.",
            "Stage71 strict no-additions output failure remains open.",
        ],
    }
    (phase / "storage-read.json").write_text(json.dumps(stored, ensure_ascii=False, indent=2, default=str) + chr(10))
    (phase / "native-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(
        json.dumps(
            {
                "checks": str(result["passed"]) + "/" + str(result["total"]),
                "failed": [k for k, v in checks.items() if not v],
                "actual_calls": len(calls),
            },
            ensure_ascii=False,
        )
    )
    assert all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    asyncio.run(run(parser.parse_args()))
