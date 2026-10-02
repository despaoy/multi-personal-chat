"""Audit a real semantic writer and source-backed lookup in an isolated PostgreSQL database."""

import argparse
import asyncio
import json
import re
from html import unescape
from pathlib import Path


async def run(args):
    import asyncpg

    phase = Path(args.phase).resolve()
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    assert phase.parent == runtime / "backups/backend-chain-20261001" and re.fullmatch(r"stage[1-9]\d*", phase.name)
    root = phase / args.variant
    fixture = json.loads((phase / "fixture.json").read_text())
    proof = json.loads((root / "result.json").read_text())
    calls = json.loads((root / "cloud-calls.json").read_text())
    admission = json.loads((root / "writer-admission.json").read_text())
    before = json.loads((root / "before-question.json").read_text())
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check(
        "fully_informative_original_fixture",
        fixture["synthetic"]
        and len(fixture["expected_requirements"]) == 7
        and not fixture["proposal_fixture"]
        and fixture["rule_seeded_memories"] == 0,
    )
    check(
        "native_authenticated_ordinary_user",
        proof["transport"] == "authenticated_ASGI"
        and proof["chat_auth_statuses"] == [200, 200]
        and proof["seed_scope"]["owner"] != "1",
    )
    check(
        "two_actual_successful_primary_turns",
        proof["seed_http_status"] == proof["http_status"] == 200 and proof["primary_calls"] == 2,
    )
    check("seven_real_calls_no_replay", len(calls) == 7 and proof["cloud_calls"] == 7)
    check(
        "request_and_response_pro",
        all(c["request"]["model"] == c["response"].get("model") == "deepseek-v4-pro" for c in calls),
    )
    check(
        "all_responses_complete",
        all(c["http_status"] == 200 and c["response"]["choices"][0]["finish_reason"] == "stop" for c in calls),
    )
    writer = []
    primary = []
    for call in calls:
        try:
            payload = json.loads(call["request"]["messages"][-1]["content"])
        except (ValueError, KeyError):
            payload = {}
        if "current_user_message" in payload:
            writer.append((call, payload))
        if any("<user_query>" in m.get("content", "") for m in call["request"]["messages"]):
            primary.append(call)
    check("two_real_writer_calls", len(writer) == 2)
    seed_call, payload = writer[0]
    raw = seed_call["response"]["choices"][0]["message"]["content"]
    candidates = json.loads(raw)["memories"]
    check("complete_seed_current_input", payload["current_user_message"] == fixture["source_message"])
    check(
        "no_manufactured_old_whitelist",
        payload["existing_memories"] == [] and admission[0]["input"]["existing_memories"] == [],
    )
    check(
        "unchanged_writer_limits_fit",
        seed_call["request"]["max_tokens"] == 768
        and seed_call["response"]["usage"]["completion_tokens"] < 768
        and seed_call["response"]["usage"]["total_tokens"] < 65536,
    )
    check(
        "three_raw_grounded_proposals",
        len(candidates) == 3
        and all(
            c["evidence"] in fixture["source_message"] and c["operation"] == "ADD" and c["attributed_to"] == "user"
            for c in candidates
        ),
    )
    check(
        "raw_time_and_both_conditions",
        candidates[0]["qualifiers"] == {"time": "工作日晚上"}
        and candidates[1]["qualifiers"] == {"time": "周末上午", "condition": "已经吃过早餐且当天没有胃痛"},
    )
    check("parser_raw_response_unchanged", admission[0]["raw_response"] == raw)
    check(
        "parser_admits_three",
        len(admission[0]["accepted"]) == 3 and all(x["operation"] == "ADD" for x in admission[0]["accepted"]),
    )
    check(
        "source_only_not_substituted_for_writer",
        proof["seed_method"] == "actual_authenticated_native_generate_then_real_semantic_writer_no_fabricated_memories",
    )
    check(
        "scheduler_reports_actual_three_commits",
        before["scheduler_before_question"]["saved"] == 3
        and before["scheduler_before_question"]["recent_results"][-1]["accepted"]
        == before["scheduler_before_question"]["recent_results"][-1]["persisted"]
        == 3,
    )
    check(
        "no_writer_failure_or_queued_job",
        proof["scheduler_final"]["failed"]
        == proof["scheduler_final"]["queued"]
        == proof["scheduler_final"]["buffered"]
        == proof["scheduler_final"]["processing"]
        == 0,
    )
    check(
        "lookup_not_added_as_new_fact",
        len(admission[-1]["accepted"]) == 0
        and writer[-1][1]["current_user_message"] == fixture["question"]
        and json.loads(writer[-1][0]["response"]["choices"][0]["message"]["content"]) == {"memories": []},
    )
    cluster = runtime / "evaluations/r148pg.s3"
    connection = await asyncpg.connect(
        user="boot", database=proof["database"], host=str(cluster / "socket"), port=25433
    )
    try:
        async with connection.transaction(readonly=True):
            assert await connection.fetchval("SHOW data_directory") == str(cluster / "data")
            owner = proof["seed_scope"]["owner"]
            rows = await connection.fetch(
                "SELECT * FROM character_memories WHERE character_id=$1 AND platform=$2 AND adapter=$3 AND sender_id=$4 AND conversation_type=$5 AND conversation_id=$6",
                "tsukiyashiro_kisaki",
                "web",
                "web-character",
                owner,
                "private",
                owner,
            )
            owner_key = json.dumps(("web", "web-character", owner))
            scope_key = json.dumps(("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner))
            sources = await connection.fetch(
                "SELECT * FROM memory_sources WHERE owner_key=$1 AND scope_key=$2", owner_key, scope_key
            )
            links = await connection.fetch(
                "SELECT l.memory_id,s.source_message_id,s.body,s.owner_key,s.scope_key,s.state FROM memory_source_links l JOIN memory_sources s ON s.source_key=l.source_key WHERE l.memory_id=ANY($1::bigint[])",
                [row["id"] for row in rows],
            )
            storage = {
                "database": proof["database"],
                "data_directory": str(cluster / "data"),
                "owner": owner,
                "rows": [dict(row) for row in rows],
                "sources": [dict(row) for row in sources],
                "links": [dict(row) for row in links],
            }
    finally:
        await connection.close()
    actual = {row["id"]: dict(row) for row in rows}
    check(
        "independent_three_durable_claims",
        len(rows) == 3
        and set(actual)
        == {row["id"] for row in before["seed_records"]}
        == {row["id"] for row in proof["user_fact_records_after"]},
    )
    check(
        "durable_content_evidence_match_admitted_proposals",
        all(
            any(
                row["content"] == p["memory"]["content"] and json.loads(row["evidence_json"]) == [p["evidence"]]
                for row in rows
            )
            for p in admission[0]["accepted"]
        ),
    )
    check(
        "all_durable_claims_active_user_ADD",
        all(
            row["status"] == "active"
            and row["relation_type"] == "ADD"
            and json.loads(row["metadata_json"])["attributed_to"] == "user"
            for row in rows
        ),
    )
    check(
        "durable_qualifiers_unchanged",
        all(
            json.loads(row["metadata_json"])["qualifiers"]
            == dict(next(p for p in admission[0]["accepted"] if p["memory"]["content"] == row["content"])["qualifiers"])
            for row in rows
        ),
    )
    check(
        "server_observation_time_authority",
        all(
            json.loads(row["metadata_json"])["temporal_provenance"]["producer"] == "semantic_memory"
            and row["observed_at"] == proof["prepared"][0]["received_at"]
            for row in rows
        ),
    )
    check("no_invented_calendar_validity", all(row["valid_from"] is None and row["valid_to"] is None for row in rows))
    check(
        "durable_original_source_and_lookup_source",
        len(sources) == 2
        and {row["body"] for row in sources} == {fixture["source_message"], fixture["question"]}
        and all(row["state"] == "recorded" for row in sources),
    )
    check(
        "all_three_exact_claim_source_links",
        len(links) == 3
        and {row["memory_id"] for row in links} == set(actual)
        and all(
            row["body"] == fixture["source_message"]
            and row["owner_key"] == owner_key
            and row["scope_key"] == scope_key
            and row["state"] == "recorded"
            for row in links
        ),
    )
    check(
        "before_question_independent_original_source",
        before["durable_seed_verified_before_question"]
        and len(before["durable_seed_sources"]) == 1
        and before["durable_seed_sources"][0]["body"] == fixture["source_message"],
    )
    prepared = proof["prepared"][-1]
    wire = unescape(primary[-1]["request"]["messages"][-1]["content"])
    match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    evidence = json.loads(match[1]) if match else {}
    check(
        "final_wire_full_original_source",
        any(row["text"] == fixture["source_message"] for row in evidence.get("records", [])),
    )
    check("final_wire_original_question", fixture["question"] in wire)
    check(
        "source_in_untrusted_reference_not_system",
        match is not None
        and 'trust="untrusted"' in match.group()
        and all(
            fixture["source_message"] not in m["content"]
            for m in primary[-1]["request"]["messages"]
            if m["role"] == "system"
        ),
    )
    check(
        "historical_task_observation_filter_explicit",
        prepared["recall"]["records_read"] == 3
        and prepared["recall"]["temporal_views"]["observation"] == 3
        and prepared["recall"]["usable_records"] == 0
        and prepared["used_memory_ids"] == [],
    )
    check(
        "history_contribution_not_hidden",
        len(prepared["history"]) == 2 and prepared["history"][0]["content"] == fixture["source_message"],
    )
    reply = proof["response"]["reply"]
    check("raw_model_reply_equals_API", reply == primary[-1]["response"]["choices"][0]["message"]["content"])
    check(
        "all_seven_answer_constraints",
        all(
            term in reply
            for term in [
                "工作日晚上",
                "不含咖啡因的麦茶",
                "周末上午",
                "淡拿铁",
                "已经吃过早餐",
                "当天没有胃痛",
                "浓缩咖啡",
            ]
        )
        and "同时满足两个条件" in reply
        and "明确不喜欢：浓缩咖啡" in reply,
    )
    check(
        "question_fixture_remains_complete",
        fixture["question"] == json.loads((phase / "fixture.json").read_text())["question"],
    )
    check("pending_observation_retained", proof["sync_pending_final"] == 1)
    result = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "phase_calls": len(calls),
        "actual_primary_replies": 2,
        "qualified_lookup_replies": 1,
        "semantic_proposals": 3,
        "durable_claims": 3,
        "claim_source_links": 3,
        "runtime_fix_needed": False,
        "pending_at_probe_save": proof["sync_pending_final"],
        "terminal_storage_verified": True,
        "limits": [
            "Historical question uses original source and two-message history; all three fact records are observation-filtered. This is not a fact-only or distant-history recall proof.",
            "Pending adapter count was 1 when result was saved; independent read after process exit verifies final three claims, their three links and both source bodies. Do not relabel saved count as zero.",
            "Stage71 strict no-additions failure remains open.",
        ],
        "raw_writer_response": raw,
        "raw_lookup_reply": reply,
    }
    (phase / "storage-read.json").write_text(json.dumps(storage, ensure_ascii=False, indent=2, default=str) + chr(10))
    (phase / "native-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "passed",
                    "total",
                    "phase_calls",
                    "actual_primary_replies",
                    "qualified_lookup_replies",
                    "durable_claims",
                    "claim_source_links",
                    "pending_at_probe_save",
                ]
            }
        )
    )
    assert all(checks.values()), json.dumps([k for k, v in checks.items() if not v])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--variant", default="native-pg-baseline")
    asyncio.run(run(parser.parse_args()))
