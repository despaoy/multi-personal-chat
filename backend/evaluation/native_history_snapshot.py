"""Reuse actual native history advancement without replay or count overrides."""

import hashlib
from pathlib import Path

from evaluation.native_successor_snapshot import _artifact
from evaluation.source_transport import complete_source_in_transport


def validate_history_restore(proof, phase, visited):
    from evaluation.native_seed_provenance import expected_seed_record_count

    phase = Path(phase).resolve()
    ref = proof["native_history_seed_origin"]
    origin = _artifact(phase, ref, "result.json", "result_sha256")
    calls = _artifact(phase, ref, "cloud-calls.json", "cloud_calls_sha256")
    prior_ref = ref["restored_seed"]
    prior = _artifact(phase, prior_ref, "before-question.json", "proof_sha256")
    assert prior_ref["variant"] != ref["variant"] or prior_ref["phase"] != ref["phase"]
    assert int(prior_ref["phase"][5:]) <= int(ref["phase"][5:])
    count = expected_seed_record_count(prior, phase.parent / prior_ref["phase"], _visited=visited)
    assert origin["synthetic_only"] is True and origin["transport"] == "authenticated_ASGI"
    assert origin["database_mode"] == "PostgreSQL"
    prefix_only = ref.get("successful_prefix_only") is True
    assert origin["http_status"] == (500 if prefix_only else 200)
    assert origin["chat_auth_statuses"] == [200, 200] and origin["seed_template_verified"]
    assert origin["seed_scope"] == prior["seed_scope"] == proof["seed_scope"]
    assert origin["seed_scope"]["owner"] != "1"
    assert origin["seed_records"] == origin["user_fact_records_after"] == prior["seed_records"] == proof["seed_records"]
    assert len(proof["seed_records"]) == count
    assert origin["durable_seed_verified_before_question"] and proof["durable_seed_verified_before_question"]
    assert origin["before_question_backup"] == proof["before_question_backup"]
    assert proof["before_question_backup"]["prior_same_task_answers"] == 0
    assert proof["before_question_backup"]["successful_seed_claim_ids"] == [r["id"] for r in proof["seed_records"]]
    root = phase.parent / ref["phase"] / ref["variant"]
    assert (
        hashlib.sha256((root / "before-question.dump").read_bytes()).hexdigest()
        == proof["before_question_backup"]["database_sha256"]
    )
    assert origin["durable_seed_sources"] == proof["durable_seed_sources"]
    before = {s["source_message_id"]: s for s in prior["durable_seed_sources"]}
    after = {s["source_message_id"]: s for s in proof["durable_seed_sources"]}
    assert len(after) == len(proof["durable_seed_sources"])
    assert all(after[key] == source for key, source in before.items())
    status = origin["scheduler_final"]
    assert status["saved"] == status["erased"] == status["failed"] == 0
    assert origin["cloud_calls"] == len(calls) and calls
    assert all(c["http_status"] == 200 and c["request"]["model"] == "deepseek-flash" for c in calls)
    inherited = origin["history_advancement_turns_inherited"]
    previous_turns = prior.get("history_advancement_turns", [])
    turns = origin["history_advancement_turns"]
    assert inherited == len(previous_turns) and turns[:inherited] == previous_turns
    assert turns == proof["history_advancement_turns"]
    assert origin["history_advancement_turns_completed"] == len(turns)
    new_turns = turns[inherited:]
    assert new_turns and all(t["status"] == 200 and not t["response"]["abstained"] for t in new_turns)
    prefix = calls[: origin["calls_before_question"]]
    primary = [
        c
        for c in prefix
        if c["request"].get("max_tokens") == origin["primary_output_tokens"]
        and any("<user_query>" in m["content"] for m in c["request"]["messages"])
    ]
    generated = origin["generation"][: len(new_turns)]
    assert len(generated) == len(new_turns) and all(g["observed_primary_calls"] >= 1 for g in generated)
    assert len(primary) == sum(g["observed_primary_calls"] for g in generated)
    if prefix_only:
        before_question = _artifact(phase, ref, "before-question.json", "before_question_sha256")
        blocked = _artifact(phase, ref, "last-sent-request.json", "blocked_request_sha256")
        fields = ("seed_scope", "seed_records", "durable_seed_sources", "before_question_backup",
                  "history_advancement_turns", "history_advancement_turns_completed",
                  "history_advancement_turns_inherited", "calls_before_question")
        assert all(before_question[field] == origin[field] for field in fields)
        assert origin["calls_before_question"] == len(calls)
        assert before_question["generation"] == origin["generation"] == generated
        assert origin["primary_calls"] == len(primary)
        assert before_question["scheduler_before_question"] == origin["scheduler_final"]
        assert before_question["writer_admission_before_question"] == origin["writer_admission"]
        assert len(before_question["prepared"]) == len(new_turns)
        assert origin["prepared"][:-1] == before_question["prepared"]
        assert len(origin["prepared"]) == len(new_turns) + 1
        assert blocked["model"] == "deepseek-flash" and blocked["max_tokens"] == origin["primary_output_tokens"]
        assert any("<user_query>" in m["content"] for m in blocked["messages"])
        assert all(call["request"] != blocked for call in calls)
        assert proof["history_advancement_turns_completed"] == len(turns)
    # A full source must be from a real completed task request; the failed
    # final question is outside this before-question snapshot and prefix.
    for source_id in after.keys() - before.keys():
        source = after[source_id]
        assert source["state"] == "recorded"
        assert any(
            complete_source_in_transport(m["content"], source["body"])
            for call in primary
            for m in call["request"]["messages"]
            if m["role"] == "user"
        )
    return count
