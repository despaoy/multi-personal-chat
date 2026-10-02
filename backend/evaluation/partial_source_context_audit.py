"""Audit one real read of a persisted partial original-source observation."""

import json
import re
from html import unescape


def audit_source_completeness(proof, fixture, cloud_calls):
    generation = proof["generation"][0]
    actual = [c for c in cloud_calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    messages = actual[0]["request"]["messages"] if len(actual) == 1 else []
    wire = unescape(messages[-1]["content"]) if messages else ""
    block = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
    packets = [json.loads(line[2:]) for line in block[1].splitlines() if line.startswith("- {")] if block else []
    selected = [x for x in packets if x.get("id") == "6"]
    packet = selected[0] if len(selected) == 1 else {}
    before = proof["owner_before"]
    after = proof["owner_after_ack"]
    row = before["claims"][0]
    metadata = json.loads(row["metadata_json"])
    projection = metadata["erasure_evidence_projection"]
    expected = json.loads(row["evidence_json"])
    source_id = row["source_message_id"]
    done = proof["terminal_receipt"]
    semantic = done["context"]["memory_completion"]["semantic_receipt"]
    reply = generation["response"].get("replyText", "")
    guards = dict(
        normal_owner_auth=proof["auth_statuses"] == [200, 200],
        actual_signed_auth=proof["unauthorized_status"] == 401 and proof["signed_auth_statuses"] == [200, 200, 200],
        complete_new_read_input=len(fixture["cases"][-1]["message"]) > 200,
        actual_inherited_role=proof["actual_inherited_role"]["character_id"] == "tsukiyashiro_kisaki",
        real_current_pro=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        actual_one_read_only=len(proof["generation"]) == 1
        and len(actual) == 1
        and fixture["cases"][-1]["message"] in wire,
        no_current_erasure_authorization=proof["actual_intent_gate"] is False,
        actual_read_history_empty=proof["generation_diagnostics"][0]["history"] == [],
        real_saved_partial_record=before["claims"] == fixture["actual_prior_records"]
        and row["id"] == 6
        and projection.get("version") == 1
        and projection.get("complete_original_source") is False,
        no_memory_mutation_before_ack=proof["owner_before_ack"]["claims"] == before["claims"]
        and proof["owner_before_ack"]["sources"] == before["sources"],
        physical_delivery_and_duplicate=done["status"] == proof["duplicate_receipt"]["status"] == "delivered"
        and done["owner_matches_actual_token"]
        and proof["acknowledgements"][-1]["response"]["changed"] is False,
        original_reply_and_clock=done["original_reply_preserved"]
        and proof["generated_receipt"]["context"]["completion_snapshot"] == done["context"]["completion_snapshot"],
        actual_read_completion=done["context"]["memory_completion"]["state"] == "completed"
        and semantic.get("accepted", 0) == 0
        and semantic.get("persisted", 0) == 0,
        prior_claims_and_links_unchanged=after["claims"] == before["claims"]
        and after["source_links"] == before["source_links"],
        original_archive_unchanged=before["original_archive_sha256"] == after["original_archive_sha256"],
        duplicate_no_models=proof["acknowledgements"][-1]["cloud_call_range"][0]
        == proof["acknowledgements"][-1]["cloud_call_range"][1],
        parent_and_web_archive_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"]
        and proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        no_old_calls=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
        all_actual_handles_terminal=proof["delivery_worker_terminal"]
        and proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["lock_waiters_remaining"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == 0,
        evidence_not_system_rules=all(
            not any(text in m["content"] for text in expected) for m in messages if m.get("role") == "system"
        ),
    )
    source_blocks = [
        json.loads(m[1]) for m in re.finditer(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    ]
    full_sources = [x for p in source_blocks for x in p.get("records", [])]
    reviewed = []
    for call in cloud_calls[slice(*generation["cloud_call_range"])]:
        for message in call["request"]["messages"]:
            if message.get("role") != "user":
                continue
            try:
                data = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict) and data.get("query") == fixture["cases"][-1]["message"]:
                reviewed.extend(x for x in data.get("candidates", []) if x.get("id") == "6")
    business = dict(
        literal_original_fragments_in_actual_packet=packet.get("evidence") == expected,
        original_source_identity_and_clock=packet.get("source_message_ids") == [source_id]
        and packet.get("observed_at") == row["observed_at"],
        unresolved_quoted_subject_on_wire=packet.get("speaker_role") == "user"
        and packet.get("content_semantics") == "quoted_source"
        and packet.get("subject_scope") == "not_resolved"
        and packet.get("temporal_mode") == "observation",
        old_full_source_not_recalled=not any(x.get("source_id") == source_id for x in full_sources),
        erased_color_not_reintroduced="深蓝色" not in json.dumps([packets, full_sources], ensure_ascii=False),
        actual_partial_completeness_on_wire=packet.get("complete_original_source") is False
        and packet.get("source_completeness") == "partial",
        actual_original_fragment_coordinates_on_wire=packet.get("source_fragments") == projection["sources"],
        full_partial_packet_selected_for_review=len(reviewed) == 1
        and reviewed[0].get("complete_original_source") is False
        and reviewed[0].get("source_fragments") == projection["sources"]
        and reviewed[0].get("evidence") == expected,
        course_facts_understood=all(value in reply for value in ["海庭鹤林", "MC-845-D", "2026", "12", "5"])
        and any(day in reply for day in ["周六", "星期六", "礼拜六"]),
        partial_source_understood=bool(
            re.search(
                r"(?:不是|并非).{0,8}完整原话|(?:完整原始来源|完整原话).{0,10}(?:false|为否)|(?:来源|原话).{0,8}(?:是|为|标记).{0,8}partial",
                reply,
            )
        ),
        known_original_gap_on_wire=packet.get("known_source_gaps")
        == [dict(source_message_id=source_id, span=[91, 122])]
        and "Unicode" in packet.get("source_fragment_position_note", ""),
        known_gap_understood=bool(re.search(r"(?:91.{0,8}122|空缺|缺口|不连续|并不连续|并非连续)", reply))
        and not bool(re.search(r"(?:之间是否连续|连续性).{0,12}(?:材料没有说明|无法判断|未知)", reply)),
        missing_text_not_reconstructed="深蓝色" not in reply
        and bool(
            re.search(
                r"(?:不|无法|不能|未知|不知道|未提供).{0,24}(?:省略|缺失|删减|被删|补|猜|恢复|核对)|(?:省略|缺失|删减|被删).{0,24}(?:不|无法|不能|未知|未提供)",
                reply,
            )
        ),
        current_state_not_asserted_from_old_observation=bool(
            re.search(
                r"(?:现状|当前|现在).{0,30}(?:无法|不能|不保证|不能保证|未核实|未核对|没有.{0,8}核对)|(?:无法|不能|不能保证|未核实|未核对).{0,30}(?:现状|当前|现在)",
                reply,
            )
        ),
    )
    proof["authority_guards"] = guards
    proof["source_completeness_checks"] = business
    return {**guards, **business}
