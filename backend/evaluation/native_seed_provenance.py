"""Accept a successor seed only when bound to a completed real native update artifact."""

import hashlib
import json
import re
from pathlib import Path


def expected_seed_record_count(proof, phase):
    provenance = proof.get("native_update_seed_origin")
    if provenance is None:
        return 4 if proof.get("additional_source_written_by_real_native_turn") else 3
    try:
        phase = Path(phase).resolve()
        origin_phase = provenance["phase"]
        variant = provenance["variant"]
        assert re.fullmatch(r"stage[1-9]\d*", origin_phase)
        assert re.fullmatch(r"native-pg(?:-[a-z]{1,12})*", variant)
        root = phase.parent / origin_phase / variant
        raw = (root / "result.json").read_bytes()
        cloud_raw = (root / "cloud-calls.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance["result_sha256"]
        assert hashlib.sha256(cloud_raw).hexdigest() == provenance["cloud_calls_sha256"]
        origin = json.loads(raw)
        calls = json.loads(cloud_raw)
        assert origin["synthetic_only"] and origin["transport"] == "authenticated_ASGI"
        assert origin["database_mode"] == "PostgreSQL" and origin["http_status"] == 200
        assert origin["chat_auth_statuses"] == [200, 200]
        assert origin["seed_scope"]["owner"] != "1" and origin["seed_scope"] == proof["seed_scope"]
        assert origin["primary_calls"] == 1 and origin["cloud_calls"] == len(calls)
        assert calls and all(
            c["http_status"] == 200 and c["request"]["model"] in {"deepseek-flash", "deepseek-v4-pro"} for c in calls
        )
        old, new = origin["seed_records"], origin["user_fact_records_after"]
        assert len(old) in {3, 4} and len(new) == len(old) + 1
        assert proof["seed_records"] == new
        assert all(isinstance(r["id"], int) and not isinstance(r["id"], bool) and r["id"] > 0 for r in [*old, *new])
        old_by_id, new_by_id = {r["id"]: r for r in old}, {r["id"]: r for r in new}
        assert len(old_by_id) == len(old) and len(new_by_id) == len(new)
        added = [r for r in new if r["id"] not in old_by_id]
        assert len(added) == 1
        successor = added[0]
        target = successor["supersedes_memory_id"]
        previous, historical = old_by_id[target], new_by_id[target]
        assert successor["relation_type"] == "SUPERSEDE" and successor["status"] == "active"
        assert previous["status"] == "active" and historical["status"] == "superseded"
        assert successor["memory_key"] == previous["memory_key"]
        assert all(new_by_id[r["id"]] == r for r in old if r["id"] != target)
        mutable = {"status", "valid_to", "updated_at"}
        assert {k: v for k, v in historical.items() if k not in mutable} == {
            k: v for k, v in previous.items() if k not in mutable
        }
        scope = (
            "character_id",
            "platform",
            "adapter",
            "sender_id",
            "conversation_type",
            "conversation_id",
            "scope_level",
        )
        assert all(successor.get(k) == previous.get(k) for k in scope)
        status = origin["scheduler_final"]
        assert status["saved"] == 1 and status["failed"] == status["erased"] == 0
        accepted_admissions = [a for a in origin["writer_admission"] if a["accepted"]]
        admissions = [p for a in accepted_admissions for p in a["accepted"]]
        assert len(admissions) == 1 and admissions[0]["operation"] == "SUPERSEDE"
        assert admissions[0]["target_memory_id"] == str(target)
        assert admissions[0]["target_memory_key"] == successor["memory_key"]
        admitted = accepted_admissions[0]
        assert any(
            str(c.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
            == str(admitted["raw_response"]).strip()
            for c in calls
        )
        source_message = admitted["input"]["source_message"]
        assert successor["evidence"] == [admissions[0]["evidence"]]
        assert all(e in source_message for e in successor["evidence"])
        sources = successor["source_message_ids"]
        assert sources and status["recent_results"][-1]["source_message_id"] in sources
        assert proof["before_question_backup"]["successful_seed_claim_ids"] == [r["id"] for r in new]
        return len(new)
    except (AssertionError, KeyError, TypeError, AttributeError, OSError, ValueError) as error:
        raise ValueError("Invalid native update seed provenance") from error
