"""Actual native field and original-receipt authority, not display-only labels."""

from html import unescape


def audit_receipts(p, f, calls):
    checks = dict(p["pg_receipt_checks"])
    g = p["generation"][-1]
    prepared = p["prepared_diagnostics"][-1]
    diags = p["generation_diagnostics"]
    primary = [c for c in calls if c["request"].get("max_tokens") == 1024]
    wire = "\n".join(unescape(m.get("content", "")) for c in primary for m in c["request"]["messages"])
    reply = g["response"].get("reply", "")
    packets = prepared.get("receipt_memory_packets", [])
    target = next((item for item in packets if item["memory_id"] == str(p["claim_ids"]["target"])), {})
    checks.update(
        native_fixture_login=p["fixture_auth_statuses"] == [200, 200],
        single_new_complete_question=len(p["generation"]) == 1
        and g["id"] == f["cases"][0]["id"]
        and g["http_status"] == 200,
        real_pro_calls_http200=bool(calls)
        and all(
            c["http_status"] == 200
            and c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        one_actual_primary=len(primary) == 1,
        actual_window65536=len(diags) == 1 and diags[0]["context_window_tokens"] == 65536,
        empty_new_user_history=prepared["history"] == [],
        actual_current_major_packet=target.get("memory_key") == "user_major"
        and target.get("temporal_mode") != "observation"
        and f["target"]["body"] in target.get("evidence", []),
        target_packet_source_identity=target.get("source_message_ids") == [f["target"]["id"]],
        target_packet_unconditional=target.get("status") == "active"
        and not target.get("qualifiers")
        and not target.get("historical")
        and target.get("complete_original_source") is not False,
        actual_candidate_receipt_read=p["candidate_diagnostics"][-1]["recall"].get("source_authority_reader")
        == "exact_claim_source_pairs"
        and p["candidate_diagnostics"][-1]["recall"]["source_receipt_pairs_requested"]
        == p["candidate_diagnostics"][-1]["recall"]["source_receipt_pairs_returned"]
        == 101,
        actual_target_used=str(p["claim_ids"]["target"]) in prepared["used_memory_ids"],
        actual_full_source_in_model_input=f["target"]["body"] in wire and f["expected_major"] in wire,
        answer_current_major_correct=f["expected_major"] in reply
        and not any(term in reply for term in ["不能确认你的专业", "没有你的专业", "库存记录是你的专业"]),
        seeded_authority_unchanged=p["seeded_rows_unchanged_after_generation"],
        database_queue_drained=p["sync_pending_final"] == 0,
    )
    return checks
