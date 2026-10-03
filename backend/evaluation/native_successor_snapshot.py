"""Validate a deferred native successor against its authenticated prior seed and PG snapshot."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def _artifact(phase, ref, name, digest):
    assert re.fullmatch(r"stage[1-9]\d*", ref["phase"])
    assert re.fullmatch(r"native-pg(?:-[a-z]{1,12})*", ref["variant"])
    assert int(ref["phase"][5:]) <= int(phase.name[5:])
    root = phase.parent / ref["phase"] / ref["variant"]
    assert root.resolve().parent == (phase.parent / ref["phase"]).resolve()
    raw = (root / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref[digest]
    return json.loads(raw)


def validate_deferred_successor(proof, phase, origin, calls, old, new, provenance):
    """This does not author records, certify model dates or schedule future changes."""
    from evaluation.native_seed_provenance import expected_seed_record_count

    phase = Path(phase).resolve()
    assert all(type(r["id"]) is int and r["id"] > 0 for r in [*old, *new])
    prior = provenance["prior_seed"]
    assert prior["phase"] == provenance["phase"]
    prior_proof = _artifact(phase, prior, "before-question.json", "proof_sha256")
    assert prior_proof["native_update_seed_origin"]["phase"] != prior["phase"]
    assert int(prior_proof["native_update_seed_origin"]["phase"][5:]) < int(prior["phase"][5:])
    assert expected_seed_record_count(prior_proof, phase.parent / prior["phase"]) == len(old)
    assert prior_proof["seed_records"] == old
    assert prior_proof["seed_scope"] == origin["seed_scope"]
    assert prior_proof["before_question_backup"] == origin["before_question_backup"]
    assert prior_proof["durable_seed_sources"] == origin["durable_seed_sources"]
    snap_ref = provenance["post_update_snapshot"]
    assert int(snap_ref["phase"][5:]) >= int(provenance["phase"][5:])
    snapshot = _artifact(phase, snap_ref, "post-update-snapshot.json", "snapshot_sha256")
    assert snapshot["transport"] == "actual_readonly_PostgreSQL_then_pg_dump"
    assert snapshot["database"] == origin["database"]
    assert snapshot["seed_scope"] == proof["seed_scope"] == origin["seed_scope"]
    assert snapshot["records"] == new == proof["seed_records"]
    assert snapshot["backup"] == proof["before_question_backup"]
    assert snapshot["sources"] == proof["durable_seed_sources"]
    assert snapshot["source_fence"] == snapshot["prior_source_fence"]
    root = phase.parent / snap_ref["phase"] / snap_ref["variant"]
    assert (
        hashlib.sha256((root / "before-question.dump").read_bytes()).hexdigest()
        == snapshot["backup"]["database_sha256"]
    )
    before = {r["id"]: r for r in old}
    after = {r["id"]: r for r in new}
    assert len(before) == len(old) and len(after) == len(new) == len(old) + 1
    assert all(after[r["id"]] == r for r in old)
    added = [r for r in new if r["id"] not in before]
    assert len(added) == 1
    successor = added[0]
    target = successor["parent_memory_id"]
    previous = before[target]
    assert successor["relation_type"] == "COEXIST" and successor["status"] == "active"
    assert successor["supersedes_memory_id"] is None and previous["status"] == "active"
    assert successor["memory_key"] == previous["memory_key"]
    scope = ("character_id", "platform", "adapter", "sender_id", "conversation_type", "conversation_id", "scope_level")
    assert all(successor[k] == previous[k] for k in scope)
    status = origin["scheduler_final"]
    assert status["saved"] == 1 and status["failed"] == status["erased"] == 0
    accepted = [(a, p) for a in origin["writer_admission"] for p in a["accepted"]]
    assert len(accepted) == 1
    admission, proposal = accepted[0]
    assert successor["valid_from"] == (proposal.get("valid_from") or None)
    assert successor["valid_to"] == (proposal.get("valid_to") or None)
    assert proposal["operation"] in {"SUPERSEDE", "RETRACT", "MERGE"}
    assert proposal["target_memory_id"] == str(target) and proposal["target_memory_key"] == previous["memory_key"]
    assert proposal["memory"]["content"] == successor["content"]
    assert proposal["memory"]["memory_key"] == successor["memory_key"]
    assert proposal["attributed_to"] == successor["attributed_to"] == "user"
    assert dict(proposal["qualifiers"]) == successor["qualifiers"]
    assert any(
        str(c["response"]["choices"][0]["message"]["content"]).strip() == str(admission["raw_response"]).strip()
        for c in calls
    )
    assert successor["evidence"] == [proposal["evidence"]]
    sources = {s["source_message_id"]: s for s in snapshot["sources"]}
    assert len(sources) == len(snapshot["sources"])
    assert all(sources[s["source_message_id"]] == s for s in origin["durable_seed_sources"])
    source_id = status["recent_results"][-1]["source_message_id"]
    assert successor["source_message_ids"] == [source_id]
    source = sources[source_id]
    assert source["state"] == "recorded" and source["body"] == admission["input"]["source_message"]
    # The literal original is never replaced with the parser's whitespace view.
    from character.deferred_memory_mutation import deferred_evidence_start, deferred_source_start

    start = deferred_evidence_start(
        proposal["evidence"], source["body"], observed_at=datetime.fromisoformat(source["observed_at"])
    )
    if start is None:
        # The writer first handles a complete parser-normalized first-person
        # clause. Verify that exact evidence occurs only once in the original,
        # including its literal date, before accepting the recorded deferral.
        needle = "".join(proposal["evidence"].split())
        compact = "".join(source["body"].split())
        position = compact.find(needle)
        assert needle and position >= 0 and compact.find(needle, position + 1) < 0
        from character.quoted_erasure_authority import masked_quotes

        masked = "".join(masked_quotes(source["body"])[0].split())
        assert masked.find(needle) >= 0
        start = deferred_source_start(proposal["evidence"], observed_at=datetime.fromisoformat(source["observed_at"]))
    assert start is not None
    deferred = successor["metadata"]["deferred_mutation"]
    assert deferred["original_operation"] == proposal["operation"]
    assert deferred["not_a_scheduled_replacement"] is True
    assert deferred["expression"] == start.text
    assert datetime.fromisoformat(deferred["earliest_possible_start"]) == start.lower
    assert deferred["source_observed_at"] == source["observed_at"] == successor["observed_at"]
    assert successor["metadata"]["temporal_provenance"]["validity_authority"] == "unverified"
    assert proof["before_question_backup"]["successful_seed_claim_ids"] == [r["id"] for r in new]
    return len(new)
