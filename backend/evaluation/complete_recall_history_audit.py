"""Validate the same old source reaches candidates, selector, actual wire and plan."""

from html import unescape


def audit_history(p, f, calls):
    checks = dict(p["pg_recall_checks"])
    g = p["generation"][-1]
    prepared = p["prepared_diagnostics"][-1]
    candidate = p["candidate_diagnostics"][-1]
    primary = [c for c in calls if c["request"].get("max_tokens") == 1024]
    wire = "\n".join(unescape(m.get("content", "")) for c in primary for m in c["request"]["messages"])
    reply = g["response"].get("reply", "")
    compact = reply.lower().replace(" ", "")
    target = next(
        (item for item in prepared.get("recall_memory_packets", []) if item["memory_id"] == str(p["target_claim_id"])),
        {},
    )
    checks.update(
        native_fixture_login=p["fixture_auth_statuses"] == [200, 200],
        single_new_complete_question=len(p["generation"]) == 1
        and g["http_status"] == 200
        and g["id"] == f["cases"][0]["id"],
        real_pro_calls_http200=bool(calls)
        and all(
            c["http_status"] == 200
            and c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        one_actual_primary=len(primary) == 1,
        actual_window65536=len(p["generation_diagnostics"]) == 1
        and p["generation_diagnostics"][0]["context_window_tokens"] == 65536,
        provided_history_exact=prepared["history"] == f["history"],
        actual_memory_candidates_have_old_target=any(
            item["memory_id"] == str(p["target_claim_id"]) and f["target"]["body"] in item["evidence"]
            for item in candidate["items"]
        ),
        server_owned_history_input=p.get("server_history_seeded_complete") is True,
        actual_selector_complete_history=any(
            c["request"].get("max_tokens") == 2048
            and __import__("json").loads(c["request"]["messages"][-1]["content"]).get("history") == f["history"]
            for c in calls
        ),
        actual_original_target_used=str(p["target_claim_id"]) in prepared["used_memory_ids"],
        actual_original_source_identity=tuple(target.get("source_message_ids", ())) == (f["target"]["id"],),
        actual_full_original_source_bound=f["target"]["body"] in target.get("evidence", [])
        and target.get("status") == "active",
        actual_full_source_and_topic_in_wire=f["target"]["body"] in wire and f["topic"] in wire,
        answer_device_and_vram="3060" in compact and "12gb" in compact,
        answer_offline_and_existing_small_model="离线" in reply
        and any(term in reply for term in ["小模型", "mini", "Mini", "0.6B", "0.6b"]),
        answer_respects_training_boundary=any(
            term in reply
            for term in [
                "不引入训练",
                "不训练",
                "不新增训练",
                "未训练",
                "不从零",
                "禁止从零",
                "不做从零",
                "不能从零",
                "不训练大模型",
                "不做大模型训练",
                "不从头训练",
            ]
        ),
        seeded_claims_unchanged=p["seeded_claims_unchanged_after_generation"],
        database_queue_drained=p["sync_pending_final"] == 0,
    )
    return checks
