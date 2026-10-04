"Bind partially completed native history to actual receipts and a durable PG snapshot."

import hashlib
import json
from pathlib import Path

from evaluation.native_successor_snapshot import _artifact
from evaluation.source_transport import complete_source_in_transport


def validate_partial_history(proof, phase, visited):
    from evaluation.native_seed_provenance import expected_seed_record_count

    phase = Path(phase).resolve()
    ref = proof["native_partial_history_seed_origin"]
    origin = _artifact(phase, ref, "preparation-observed.json", "preparation_sha256")
    progress = _artifact(phase, ref, "history-advancement-progress.json", "progress_sha256")
    calls = _artifact(phase, ref, "cloud-calls.json", "cloud_calls_sha256")
    admissions = _artifact(phase, ref, "writer-admission.json", "admission_sha256")
    stop = _artifact(phase, ref, "evaluation-budget-stop.json", "budget_stop_sha256")
    previous = ref["prior_seed"]
    assert int(previous["phase"][5:]) <= int(ref["phase"][5:])
    identity = (previous["phase"], previous["variant"])
    assert identity not in visited
    prior = _artifact(phase, previous, "before-question.json", "proof_sha256")
    count = expected_seed_record_count(prior, phase.parent / previous["phase"], _visited=visited)
    snap_ref = ref["snapshot"]
    assert int(snap_ref["phase"][5:]) >= int(ref["phase"][5:])
    snapshot = _artifact(phase, snap_ref, "partial-snapshot.json", "snapshot_sha256")
    case = _artifact(phase, snap_ref, "case.json", "case_sha256")
    assert origin["synthetic_only"] is True and origin["transport"] == "authenticated_ASGI"
    assert origin["database_mode"] == "PostgreSQL" and origin["chat_auth_statuses"] == [200, 200]
    assert origin["seed_template_verified"] is True
    assert origin["seed_scope"] == prior["seed_scope"] == proof["seed_scope"] == snapshot["seed_scope"]
    assert origin["seed_scope"]["owner"] != "1"
    assert snapshot["transport"] == "actual_readonly_PostgreSQL_then_pg_dump"
    assert snapshot["database"] == origin["database"]
    rows = snapshot["records"]
    assert all(type(r["id"]) is int and r["id"] > 0 for r in rows)
    before, after = ({r["id"]: r for r in group} for group in (prior["seed_records"], rows))
    assert len(rows) == len(after) == count and after == before
    assert rows == proof["seed_records"]
    assert snapshot["sources"] == proof["durable_seed_sources"]
    assert snapshot["source_fence"] == ""
    backup = snapshot["backup"]
    assert backup == proof["before_question_backup"] and backup["prior_same_task_answers"] == 0
    assert backup["successful_seed_claim_ids"] == [r["id"] for r in rows]
    dump = (phase.parent / snap_ref["phase"] / snap_ref["variant"] / "before-question.dump").read_bytes()
    assert len(dump) == backup["bytes"] and hashlib.sha256(dump).hexdigest() == backup["database_sha256"]
    inherited = prior["history_advancement_turns"]
    assert progress[:len(inherited)] == inherited
    attempted = progress[len(inherited):]
    assert len(attempted) >= 2 and attempted[-1]["status"] == 500
    successful, failed = attempted[:-1], attempted[-1]
    assert all(t["status"] == 200 and not t["response"]["abstained"] for t in successful)
    assert [t["index"] for t in progress] == list(range(len(progress)))
    assert all(t["task_id"] == case["history_advancement_tasks"][t["index"]]["id"] for t in progress)
    completed = progress[:-1]
    assert snapshot["successful_history_receipts"] == proof["history_advancement_turns"] == completed
    assert proof["history_advancement_turns_completed"] == len(completed)
    assert snapshot["failed_native_receipt"] == failed
    assert snapshot["failed_native_message_rows"] == []
    assert snapshot["failed_history_kept_without_success_override"] is True
    assert snapshot["unexecuted_final_query_absent"] is True and snapshot["failed_task_no_paid_main"] is True
    assert stop["next_send_blocked"] is True and stop["actual_calls"] == len(calls)
    assert calls and all(c["http_status"] == 200 and c["request"]["model"] == "deepseek-flash" for c in calls)
    assert all(not a["accepted"] for a in admissions)
    generated = origin["generation"]
    assert len(generated) == len(successful) and len(origin["prepared"]) == len(successful) + 1
    primary = [c for c in calls if c["request"].get("max_tokens") == 2048
               and any("<user_query>" in m["content"] for m in c["request"]["messages"])]
    assert len(primary) == sum(g["observed_primary_calls"] for g in generated)
    assert all(g["model_invoked"] and g["observed_primary_calls"] >= 1 for g in generated)
    messages = snapshot["successful_native_message_rows"]
    assert len(messages) == len(successful)
    scope_row = prior["seed_records"][0]
    owner = prior["seed_scope"]["owner"]
    character = scope_row["character_id"]
    assert prior["seed_scope"].get("character", character) == character
    for turn, message in zip(successful, messages, strict=True):
        task = case["history_advancement_tasks"][turn["index"]]
        assert message["message"] == task["message"] and message["reply"] == turn["response"]["reply"]
        assert message["senderId"] == message["userId"] == owner
        assert message["characterId"] == character
        assert message["platform"] == scope_row["platform"] and message["adapter"] == scope_row["adapter"]
        assert message["conversationType"] == "private" and message["branchId"] is None
        assert message["sessionId"] == ref["phase"] + "-advance-" + str(turn["index"] + 1)
        assert any(complete_source_in_transport(m["content"], task["message"])
                   for call in primary for m in call["request"]["messages"] if m["role"] == "user")
    sources = {s["source_message_id"]: s for s in snapshot["sources"]}
    old_sources = {s["source_message_id"]: s for s in prior["durable_seed_sources"]}
    assert len(sources) == len(snapshot["sources"]) and all(sources[k] == v for k, v in old_sources.items())
    extra = [sources[k] for k in sources.keys() - old_sources.keys()]
    recorded = [s for s in extra if s["state"] == "recorded"]
    pending = [s for s in extra if s["state"] == "pending"]
    assert len(recorded) == len(successful) and len(pending) == 1 and len(extra) == len(recorded) + 1
    owner_key = json.dumps((scope_row["platform"], scope_row["adapter"], owner))
    scope_key = json.dumps((character, scope_row["platform"],
                            scope_row["adapter"], owner, "private", owner))
    assert all(s["owner_key"] == owner_key and s["scope_key"] == scope_key for s in extra)
    assert pending[0]["body"] is None and pending[0]["observed_at"] == origin["prepared"][-1]["received_at"]
    assert all(pending[0]["source_message_id"] not in r["source_message_ids"] for r in rows)
    assert {s["source_message_id"] for s in recorded} == {m["sourceMessageId"] for m in messages}
    assert all(any(s["body"] == m["message"] for m in messages) for s in recorded)
    return count
