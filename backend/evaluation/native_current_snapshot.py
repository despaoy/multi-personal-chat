"Bind an immediate native supersession to its prior seed, source and durable snapshot."

import hashlib
from pathlib import Path

from evaluation.native_successor_snapshot import _artifact


def inherit_history_receipts(proof, prior):
    "Copy completed authenticated history receipts without replaying their requests."
    if "history_advancement_turns" not in prior:
        return
    turns = prior["history_advancement_turns"]
    assert isinstance(turns, list)
    assert prior["history_advancement_turns_completed"] == len(turns)
    assert all(t["status"] == 200 and not t["response"]["abstained"] for t in turns)
    proof["history_advancement_turns"] = list(turns)
    proof["history_advancement_turns_completed"] = len(turns)


def current_successor(old, new):
    "Validate a real row delta without creating or normalizing any record."
    assert all(type(r["id"]) is int and r["id"] > 0 for r in [*old, *new])
    before, after = ({r["id"]: r for r in rows} for rows in (old, new))
    assert len(before) == len(old) and len(after) == len(new) == len(old) + 1
    added = [r for r in new if r["id"] not in before]
    assert len(added) == 1
    successor = added[0]
    target = successor["supersedes_memory_id"]
    previous, historical = before[target], after[target]
    assert previous["status"] == "active" and historical["status"] == "superseded"
    assert successor["status"] == "active" and successor["relation_type"] == "SUPERSEDE"
    assert successor["memory_key"] == previous["memory_key"]
    assert all(after[r["id"]] == r for r in old if r["id"] != target)
    mutable = {"status", "valid_to", "updated_at"}
    assert {k: v for k, v in historical.items() if k not in mutable} == {
        k: v for k, v in previous.items() if k not in mutable
    }
    scope = ("character_id", "platform", "adapter", "sender_id", "conversation_type",
             "conversation_id", "scope_level")
    assert all(successor[k] == previous[k] for k in scope)
    assert "deferred_mutation" not in successor.get("metadata", {})
    return successor


def validate_current_successor(proof, phase, origin, calls, old, new, provenance, visited):
    from evaluation.native_seed_provenance import expected_seed_record_count

    phase = Path(phase).resolve()
    prior_ref = provenance["prior_seed"]
    assert int(prior_ref["phase"][5:]) <= int(provenance["phase"][5:])
    identity = (prior_ref["phase"], prior_ref["variant"])
    assert identity not in visited
    prior = _artifact(phase, prior_ref, "before-question.json", "proof_sha256")
    assert expected_seed_record_count(prior, phase.parent / prior_ref["phase"], _visited=visited | {identity}) == len(old)
    assert prior["seed_records"] == old and origin["seed_scope"] == prior["seed_scope"] == proof["seed_scope"]
    assert origin["before_question_backup"] == prior["before_question_backup"]
    assert origin["durable_seed_sources"] == prior["durable_seed_sources"]
    if "history_advancement_turns" in prior:
        assert proof["history_advancement_turns"] == prior["history_advancement_turns"]
        assert proof["history_advancement_turns_completed"] == prior["history_advancement_turns_completed"]
    assert origin["author_only"] is True and origin["seed_template_verified"] is True
    assert origin["additional_source_status"] == 200 and not origin["additional_source_response"]["abstained"]
    assert len(origin["generation"]) == 1 and origin["generation"][0]["model_invoked"]
    assert not origin["generation"][0]["guard_retried"]
    ref = provenance["post_update_snapshot"]
    assert int(ref["phase"][5:]) >= int(provenance["phase"][5:])
    snap = _artifact(phase, ref, "post-update-snapshot.json", "snapshot_sha256")
    assert snap["transport"] == "actual_readonly_PostgreSQL_then_pg_dump"
    assert snap["database"] == origin["database"]
    assert snap["seed_scope"] == proof["seed_scope"]
    assert snap["records"] == proof["seed_records"] == new
    assert snap["sources"] == proof["durable_seed_sources"]
    assert snap["backup"] == proof["before_question_backup"]
    assert snap["source_fence"] == snap["prior_source_fence"]
    root = phase.parent / ref["phase"] / ref["variant"]
    dump = (root / "before-question.dump").read_bytes()
    assert hashlib.sha256(dump).hexdigest() == snap["backup"]["database_sha256"]
    assert len(dump) == snap["backup"]["bytes"]
    assert snap["backup"]["prior_same_task_answers"] == 0
    assert snap["backup"]["successful_seed_claim_ids"] == [r["id"] for r in new]
    successor = current_successor(old, new)
    status = origin["scheduler_final"]
    assert status["saved"] == 1 and status["failed"] == status["erased"] == 0
    accepted = [(a, p) for a in origin["writer_admission"] for p in a["accepted"]]
    assert len(accepted) == 1
    admission, proposal = accepted[0]
    assert proposal["operation"] == "SUPERSEDE"
    assert proposal["target_memory_id"] == str(successor["supersedes_memory_id"])
    assert proposal["target_memory_key"] == successor["memory_key"]
    assert proposal["memory"]["content"] == successor["content"]
    assert proposal["memory"]["memory_key"] == successor["memory_key"]
    assert proposal["attributed_to"] == successor["attributed_to"] == "user"
    assert dict(proposal["qualifiers"]) == successor["qualifiers"]
    assert successor["valid_from"] == (proposal.get("valid_from") or None)
    assert successor["valid_to"] == (proposal.get("valid_to") or None)
    assert any(str(c["response"]["choices"][0]["message"]["content"]).strip()
               == str(admission["raw_response"]).strip() for c in calls)
    sources = {s["source_message_id"]: s for s in snap["sources"]}
    assert len(sources) == len(snap["sources"])
    assert all(sources[s["source_message_id"]] == s for s in prior["durable_seed_sources"])
    source_id = status["recent_results"][-1]["source_message_id"]
    assert successor["source_message_ids"] == [source_id]
    source = sources[source_id]
    assert source["state"] == "recorded" and source["body"] == admission["input"]["source_message"]
    assert successor["evidence"] == [proposal["evidence"]] and proposal["evidence"] in source["body"]
    assert source["observed_at"] == successor["observed_at"]
    return len(new)
