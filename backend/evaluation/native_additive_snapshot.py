"""Verify a source-backed native ADD snapshot without inventing a record count."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from evaluation.native_successor_snapshot import _artifact


def validate_source_addition(proof, phase, visited):
    from character.memory_llm import parse_llm_proposals
    from evaluation.native_seed_provenance import expected_seed_record_count

    phase = Path(phase).resolve()
    ref = proof["native_additive_seed_origin"]
    origin = _artifact(phase, ref, "result.json", "result_sha256")
    calls = _artifact(phase, ref, "cloud-calls.json", "cloud_calls_sha256")
    snap = _artifact(phase, ref, "before-question.json", "snapshot_sha256")
    prior_ref = ref["prior_seed"]
    identity = (prior_ref["phase"], prior_ref["variant"])
    assert identity not in visited
    prior = _artifact(phase, prior_ref, "before-question.json", "proof_sha256")
    count = expected_seed_record_count(prior, phase.parent / prior_ref["phase"], _visited=visited | {identity})
    assert int(prior_ref["phase"][5:]) <= int(ref["phase"][5:])
    assert origin["synthetic_only"] and origin["transport"] == "authenticated_ASGI"
    assert origin["database_mode"] == "PostgreSQL" and origin["http_status"] == 200
    assert origin["chat_auth_statuses"] == [200, 200] and origin["seed_template_verified"]
    assert origin["additional_source_status"] == 200 and not origin["additional_source_response"]["abstained"]
    assert origin["additional_source_written_by_real_native_turn"] is True
    assert origin["seed_scope"] == prior["seed_scope"] == proof["seed_scope"] == snap["seed_scope"]
    assert origin["seed_scope"]["owner"] != "1"
    old, new = prior["seed_records"], snap["seed_records"]
    assert len(old) == count
    assert new == origin["seed_records"] == origin["user_fact_records_after"] == proof["seed_records"]
    assert all(type(row["id"]) is int and row["id"] > 0 for row in [*old, *new])
    before, after = ({row["id"]: row for row in group} for group in (old, new))
    assert len(before) == len(old) and len(after) == len(new) == count + 1
    assert all(after[row["id"]] == row for row in old)
    (added,) = [row for row in new if row["id"] not in before]
    assert added["relation_type"] == "ADD" and added["status"] == "active"
    assert added["parent_memory_id"] is None and added["supersedes_memory_id"] is None
    scope = ("character_id", "platform", "adapter", "sender_id", "conversation_type", "conversation_id", "scope_level")
    assert all(added[key] == old[0][key] for key in scope)
    assert added["sender_id"] == origin["seed_scope"]["owner"]
    assert origin["cloud_calls"] == len(calls) and 0 < snap["calls_before_question"] <= len(calls)
    assert all(
        call["http_status"] == 200 and call["request"]["model"] in {"deepseek-flash", "deepseek-v4-pro"}
        for call in calls
    )
    prefix = calls[: snap["calls_before_question"]]
    assert len(snap["generation"]) == 1 and snap["generation"][0]["model_invoked"]
    assert not snap["generation"][0]["guard_fallback"]
    assert (
        sum(any("<user_query>" in message["content"] for message in call["request"]["messages"]) for call in prefix)
        == snap["generation"][0]["observed_primary_calls"]
    )
    status = snap["scheduler_before_question"]
    assert status["saved"] == 1 and status["failed"] == status["erased"] == 0
    assert (
        origin["scheduler_final"]["saved"] == 1
        and origin["scheduler_final"]["failed"] == origin["scheduler_final"]["erased"] == 0
    )
    admitted = [
        (entry, proposal) for entry in snap["writer_admission_before_question"] for proposal in entry["accepted"]
    ]
    ((entry, proposal),) = admitted
    assert proposal["operation"] == "ADD" and not proposal["target_memory_id"] and not proposal["target_memory_key"]
    actual = parse_llm_proposals(entry["raw_response"], **entry["input"])
    assert [json.loads(json.dumps(asdict(item))) for item in actual] == entry["accepted"]
    source_body = entry["input"]["source_message"]
    matched = []
    for call in prefix:
        if call["request"].get("max_tokens") != 768:
            continue
        try:
            payload = json.loads(call["request"]["messages"][-1]["content"])
        except (ValueError, TypeError, IndexError):
            continue
        if (
            payload.get("current_user_message") == source_body
            and call["response"]["choices"][0]["message"]["content"] == entry["raw_response"]
        ):
            matched.append(call)
    assert len(matched) == 1
    assert added["content"] == proposal["memory"]["content"] and added["memory_key"] == proposal["memory"]["memory_key"]
    assert added["memory_type"] == proposal["memory"]["memory_type"]
    assert added["evidence"] == [proposal["evidence"]] and proposal["evidence"] in source_body
    assert dict(proposal["qualifiers"]) == added["qualifiers"] == added["metadata"]["qualifiers"]
    assert proposal["attributed_to"] == added["attributed_to"] == "user"
    assert added["valid_from"] == (proposal["valid_from"] or None) and added["valid_to"] == (
        proposal["valid_to"] or None
    )
    sources = {row["source_message_id"]: row for row in snap["durable_seed_sources"]}
    assert len(sources) == len(snap["durable_seed_sources"])
    assert snap["durable_seed_sources"] == origin["durable_seed_sources"] == proof["durable_seed_sources"]
    assert all(sources[row["source_message_id"]] == row for row in prior["durable_seed_sources"])
    assert len(added["source_message_ids"]) == 1
    source_id = added["source_message_ids"][0]
    assert source_id not in {row["source_message_id"] for row in prior["durable_seed_sources"]}
    source = sources[source_id]
    assert source["state"] == "recorded" and source["body"] == source_body
    assert source["owner_key"] == json.dumps((added["platform"], added["adapter"], added["sender_id"]))
    assert source["scope_key"] == json.dumps(
        tuple(
            added[key]
            for key in ("character_id", "platform", "adapter", "sender_id", "conversation_type", "conversation_id")
        )
    )
    assert source["observed_at"] == added["observed_at"]
    assert any(row["source_message_id"] == source_id and row["persisted"] == 1 for row in status["recent_results"])
    assert snap["durable_seed_verified_before_question"] and proof["durable_seed_verified_before_question"]
    for field in ("history_advancement_turns", "history_advancement_turns_completed"):
        assert snap[field] == origin[field] == prior[field] == proof[field]
    backup = snap["before_question_backup"]
    assert backup == origin["before_question_backup"] == proof["before_question_backup"]
    assert backup["prior_same_task_answers"] == 0
    assert backup["successful_seed_claim_ids"] == [row["id"] for row in new]
    dump = (phase.parent / ref["phase"] / ref["variant"] / "before-question.dump").read_bytes()
    assert hashlib.sha256(dump).hexdigest() == backup["database_sha256"] and len(dump) == backup["bytes"]
    return len(new)
