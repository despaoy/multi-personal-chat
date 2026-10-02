"""Verify a complete source packet and its final conditions in the actual call."""

import json
import re
from html import unescape

from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens


def audit_deferred_source(p, f, calls):
    checks = dict(p["pg_source_checks"])
    d = p["prepared_diagnostics"][-1]
    g = p["generation"][-1]
    gd = p["generation_diagnostics"][0]
    final = gd["final_character_context"]
    primary = [c for c in calls if c["request"].get("max_tokens") == 1024]
    messages = primary[0]["request"]["messages"] if len(primary) == 1 else []
    wire = "\n".join(m.get("content", "") for m in messages)
    match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    packet = json.loads(unescape(match[1])) if match else {}
    finalpacket = json.loads(final["episodic_reference_context"]) if final["episodic_reference_context"] else {}
    pending = json.loads(d["source_candidate_context"]) if d["source_candidate_context"] else {}
    reply = g["response"].get("reply", "")
    compact = reply.lower().replace(" ", "")

    def full(data):
        return any(
            row["source_id"] == f["target"]["id"] and row["text"] == f["target"]["body"]
            for row in data.get("records", [])
        )

    checks.update(
        native_fixture_login=p["fixture_auth_statuses"] == [200, 200],
        one_new_complete_request=len(p["generation"]) == 1
        and g["id"] == f["cases"][0]["id"]
        and g["http_status"] == 200,
        history_server_owned=p["server_history_seeded_complete"] is True and d["history"] == f["history"],
        producer_deferred_whole_original=d["raw_source_status"] == "budget_omitted"
        and d["episodic_context"] == ""
        and full(pending),
        actual_final_admitted_whole_original=final["memory_source_status"] == "available"
        and final["source_candidate_context"] == ""
        and full(finalpacket),
        no_claim_or_typed_candidate_invention=p["claims_before_generation"] == []
        and p["candidate_diagnostics"][-1]["items"] == []
        and d["recall_memory_packets"] == []
        and len(final["memory_packets"]) == 0,
        actual_whole_original_in_wire=full(packet),
        actual_final_source_unresolved=packet.get("described_subject") == "not_resolved"
        and packet.get("current_validity") == "not_resolved",
        actual_source_not_system_rule=all(
            "<dialogue_evidence" not in m.get("content", "") for m in messages if m["role"] == "system"
        ),
        actual_window65536=gd["context_window_tokens"] == 65536,
        actual_complete_request_budget_fits=bool(messages)
        and sum(estimated_tokens(m.get("content", "")) + 4 for m in messages) + 1024 + CONTEXT_SAFETY_MARGIN_TOKENS
        <= 65536,
        one_actual_primary=len(primary) == 1,
        deepseek_pro_http200_stop=bool(calls)
        and all(
            c["http_status"] == 200
            and c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        three_complete_days=all(
            re.search(r"第\s*(?:" + str(i) + "|" + word + r")\s*天", reply)
            for i, word in [(1, "一"), (2, "二"), (3, "三")]
        ),
        answer_final_network_denial=bool(
            re.search(
                r"(?:禁止|不能|不得|不|无需)[^。！？!?；;，,\n]{0,5}联网|联网[^。！？!?；;，,\n]{0,10}(?:禁止|不允许|不能|不可以)|全程离线",
                reply,
            )
        ),
        answer_final_finetune_denial=bool(
            re.search(
                r"(?:禁止|不能|不得|不|无需)[^。！？!?；;，,\n]{0,5}微调|微调[^。！？!?；;，,\n]{0,10}(?:禁止|不允许|不能|不可以)",
                reply,
            )
        ),
        answer_final_validation_tuning_denial=bool(
            re.search(
                r"验证[^。！？!?；;，,\n]{0,40}(?:不|禁止|不得|不会)[^。！？!?；;，,\n]{0,20}调参|验证集[^。！？!?；;，,\n]{0,10}调参[^。！？!?；;，,\n]{0,10}(?:禁止|不允许|不能|不可以)|(?:禁止|不能|不得|不使用|不用|不)[^。！？!?；;，,\n]{0,8}验证集[^。！？!?；;，,\n]{0,8}调参",
                reply,
            )
        ),
        answer_confirmed_device_and_vram="3060" in compact and "12gb" in compact,
        answer_confirmed_data_split=all(term in compact for term in ["1800", "1440", "360", "三类"]),
        answer_cached_ready_model="0.6b" in compact and "缓存" in reply,
        original_sources_unchanged=p["original_source_rows_unchanged"],
        database_work_finished=p["sync_pending_final"] == 0,
    )
    return checks
