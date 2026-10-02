"""Evidence checks for complete original speech, not inferred current facts."""

import json
import re
from html import unescape


def audit_contextual_source(p, f, calls):
    checks = dict(p["pg_source_checks"])
    d = p["prepared_diagnostics"][-1]
    g = p["generation"][-1]
    packet = json.loads(d["episodic_context"]) if d["episodic_context"] else {}
    primary = [c for c in calls if c["request"].get("max_tokens") == 1024]
    wire = "\n".join(unescape(m.get("content", "")) for c in primary for m in c["request"]["messages"])
    reply = g["response"].get("reply", "")
    compact = reply.lower().replace(" ", "")
    checks.update(
        native_fixture_login=p["fixture_auth_statuses"] == [200, 200],
        single_complete_new_request=len(p["generation"]) == 1
        and g["id"] == f["cases"][0]["id"]
        and g["http_status"] == 200,
        server_history_complete=p.get("server_history_seeded_complete") is True and d["history"] == f["history"],
        no_typed_candidate_or_fact_invention=p["claims_before_generation"] == []
        and p["candidate_diagnostics"][-1]["items"] == []
        and d["used_memory_ids"] == []
        and d["recall_memory_packets"] == [],
        actual_raw_source_available=d["raw_source_status"] == "available"
        and d["raw_source_diagnostics"]["contextual_search_enabled"] is True,
        actual_full_original_identity_and_body=any(
            r["source_id"] == f["target"]["id"] and r["text"] == f["target"]["body"] for r in packet.get("records", [])
        ),
        actual_unclassified_speech_preserved=packet.get("speaker_role") == "user"
        and packet.get("described_subject") == "not_resolved"
        and packet.get("current_validity") == "not_resolved",
        one_real_primary=len(primary) == 1,
        actual_original_body_and_topic_in_wire=f["target"]["body"] in wire and f["history"][0]["content"] in wire,
        original_source_not_system_instruction=bool(primary)
        and all(
            f["target"]["body"] not in m.get("content", "")
            for c in primary
            for m in c["request"]["messages"]
            if m["role"] == "system"
        ),
        actual_model_window65536=len(p["generation_diagnostics"]) == 1
        and p["generation_diagnostics"][0]["context_window_tokens"] == 65536,
        actual_deepseek_pro_http200_stop=bool(calls)
        and all(
            c["http_status"] == 200
            and c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        answer_device_and_vram="3060" in compact and "12gb" in compact,
        answer_offline="离线" in reply,
        answer_cached_small_model=any(s in compact for s in ["0.6b", "mini"])
        and any(s in reply for s in ["缓存", "本地"]),
        answer_no_training=any(
            s in reply
            for s in [
                "禁止训练",
                "不训练",
                "不做训练",
                "不引入训练",
                "不进行训练",
                "不涉及训练",
                "不搞训练",
                "仅推理",
                "只推理",
                "不训练/微调",
                "不做微调",
                "禁止微调",
                "不微调",
            ]
        ),
        seven_complete_days=all(re.search(r"第\s*" + str(day) + r"\s*天", reply) for day in range(1, 8)),
        answer_given_dataset_counts=all(term in compact for term in ["1500", "1200", "300", "四类"]),
        validation_not_for_tuning=bool(re.search(r"验证.{0,40}(?:不|禁止|不得|不会).{0,30}调参", reply)),
        original_sources_unchanged=p["original_source_rows_unchanged"],
        database_work_finished=p["sync_pending_final"] == 0,
    )
    return checks
