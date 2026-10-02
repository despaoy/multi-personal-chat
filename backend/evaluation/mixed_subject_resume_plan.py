"""Validate the real provider-block restoration point without any model request."""

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path


async def run(args):
    phase = Path(args.phase).resolve()
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    assert phase.parent == runtime / "backups/backend-chain-20261001" and re.fullmatch(r"stage[1-9]\d*", phase.name)
    proof = json.loads((phase / "partial-resume.json").read_text())
    fixture = json.loads((phase / "fixture.json").read_text())
    storage = json.loads((phase / "provider-block-storage.json").read_text())
    root = phase / "native-pg-baseline"
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check(
        "original_task_terminal",
        proof["baseline_job_terminal"]
        and not (
            Path("/proc") / str(json.loads((phase / "native-baseline-job.json").read_text())["pid"]) / "cmdline"
        ).exists(),
    )
    check(
        "actual_provider_balance_error_preserved",
        proof["provider_block"]
        == {"http_status": 402, "reason": "Insufficient Balance", "request_model": "deepseek-v4-pro"},
    )
    check(
        "successful_native_seed_and_seven_distinct_bridge_turns",
        proof["seed_http_status"] == 200
        and len(proof["successful_bridge_turns"]) == 7
        and [x["index"] for x in proof["successful_bridge_turns"]] == list(range(7)),
    )
    check(
        "only_two_unfinished_bridge_indices",
        proof["remaining_bridge_indices"] == [7, 8] and len(fixture["bridges"]) == 9,
    )
    check(
        "main_query_not_yet_attempted_or_answered",
        not proof["final_question_attempted"] and all(m["message"] != fixture["question"] for m in storage["messages"]),
    )
    check(
        "actual_snapshot_hash_bound",
        hashlib.sha256((root / "provider-block.dump").read_bytes()).hexdigest()
        == proof["resume_backup"]["database_sha256"],
    )
    check(
        "real_private_claims_and_full_own_source_retained",
        len(storage["claims"]) == 3
        and all(r["sender_id"] == r["conversation_id"] == "6" for r in storage["claims"])
        and any(
            s["body"] == fixture["source_message"] and s["owner_key"] == json.dumps(("web", "web-character", "6"))
            for s in storage["sources"]
        ),
    )
    check(
        "eight_actual_completed_native_messages_retained",
        len(storage["messages"]) == 8
        and storage["messages"][0]["message"] == fixture["source_message"]
        and [x["message"] for x in storage["messages"][1:]] == [x["message"].strip() for x in fixture["bridges"][:7]],
    )
    import asyncpg

    cluster = runtime / "evaluations/r148pg.s3"
    con = await asyncpg.connect(user="boot", database=proof["database"], host=str(cluster / "socket"), port=25433)
    try:
        async with con.transaction(readonly=True):
            assert await con.fetchval("SHOW data_directory") == str(cluster / "data")
            current = await con.fetch(
                'SELECT id,message,reply,"sourceMessageId","characterId","sessionId" FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id',
                "web",
                "web-character",
                "6",
                "tsukiyashiro_kisaki",
            )
            check(
                "current_completed_rows_still_exact_snapshot_rows",
                json.loads(json.dumps([dict(r) for r in current], default=str)) == storage["messages"],
            )
    finally:
        await con.close()
    check("official_vector_snapshot_present", (root / "provider-block-vectors/index_snapshot.zip").is_file())
    output = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "cloud_requests": 0,
        "provider_access_restored": False,
        "automatic_start": False,
        "candidate_deployed": False,
        "start_bridge_index": 7,
        "remaining_bridge_indices": [7, 8],
        "actual_main_query_pending": True,
        "successful_steps_to_replay": 0,
        "full_native_case_qualified": 0,
        "resume_command_after_external_resource_restored": "PYTHONPATH=backend <runtime>/venv/bin/python backend/evaluation/mixed_subject_history_probe.py --phase "
        + str(phase)
        + " --variant native-pg-resumed --resume-provider-block --provider-access-restored --api-key-file <runtime>/config/deepseek-evaluation-api-key.txt",
        "failed_path_seed_after_resumed_main": "Use --reuse-verified-seed --seed-variant native-pg-resumed, from its actual before-question dump, never from a terminal answered state.",
    }
    (phase / "resume-plan-audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + chr(10))
    print(
        json.dumps(
            {
                "resume_plan_checks": str(output["passed"]) + "/" + str(output["total"]),
                "unfinished_bridge_indices": [7, 8],
                "main_not_attempted": True,
                "zero_cloud_requests": True,
                "candidate_deployed": False,
            }
        )
    )
    assert all(checks.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True)
    asyncio.run(run(p.parse_args()))
