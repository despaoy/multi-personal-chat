"""Audit saved native quoted-source capture and its new cold comprehension read."""

import json
import re
from html import unescape


def audit_quoted_source(proof, fixture, cloud_calls):
    generation = proof["generation"][0]
    answers = [c for c in cloud_calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    done = proof["terminal_receipt"]
    receipt = done["context"]["memory_completion"]["semantic_receipt"]
    before = proof["owner_before"]
    after = proof["owner_after_ack"]
    sid = proof["primary_source_message_id"]
    sources = [r for r in after["sources"] if r["source_message_id"] == sid]
    checks = dict(
        normal_owner_auth=proof["auth_statuses"] == [200, 200],
        actual_signed_auth=proof["unauthorized_status"] == 401
        and proof["signed_auth_statuses"] == [200] * (5 if proof.get("retained_read_executed") else 3),
        full_actual_source=len(sources) == 1 and sources[0]["body"] == fixture["new_native_source"],
        real_current_pro=bool(cloud_calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls),
        actual_final_model=len(answers) == 1
        and fixture["new_native_source"] in unescape(answers[0]["request"]["messages"][-1]["content"]),
        genuine_prior_claims=len(before["claims"]) == 2
        and len(before["sources"]) == 1
        and before["sources"][0]["source_message_id"] == fixture["initial_native_source_message_id"],
        no_memory_mutation_before_ack=all(
            proof["owner_before_ack"][key] == before[key] for key in ["claims", "sources", "source_links"]
        ),
        physical_delivery=done["status"] == "delivered" and done["owner_matches_actual_token"],
        exact_source_only_completion=done["context"]["memory_completion"]["state"] == "completed"
        and receipt.get("status") == "source_only"
        and receipt.get("source_capture") == "recorded"
        and receipt.get("accepted") == receipt.get("persisted") == 0,
        original_claims_and_links_unpromoted=after["claims"] == before["claims"]
        and after["source_links"] == before["source_links"],
        original_reply_and_clock=done["original_reply_preserved"]
        and proof["generated_receipt"]["context"]["completion_snapshot"] == done["context"]["completion_snapshot"],
        duplicate_no_models=proof["acknowledgements"][-1]["response"]["changed"] is False
        and proof["acknowledgements"][-1]["cloud_call_range"][0]
        == proof["acknowledgements"][-1]["cloud_call_range"][1],
        duplicate_no_mutation=proof["owner_after_duplicate"] == after,
        parent_unchanged=proof["scope_sql"]["source_database_snapshot_before"]
        == proof["scope_sql"]["source_database_snapshot_after"],
        web_archive_unchanged=proof["scope_sql"]["owner_messages_sha256_before"]
        == proof["scope_sql"]["owner_original_messages_sha256_after"],
        no_old_calls=proof["reused_native_fixture"]["source_writing_generations_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == proof["document_imports_replayed"]
        == 0
        and proof["searches"] == [],
        all_handles_terminal=proof["delivery_worker_terminal"]
        and proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["lock_waiters_remaining"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == 0,
        original_qq_archive_preserved=proof.get("original_qq_archive_preserved") is True,
        current_input_not_erasure=proof["actual_intent_gate"] is False,
        retained_read_executed=proof.get("retained_read_executed") is True,
    )
    if proof.get("retained_read_executed"):
        read = proof["generation"][1]
        actual = [c for c in cloud_calls[slice(*read["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        wires = (
            [unescape(m["content"]) for m in actual[0]["request"]["messages"] if m.get("role") == "user"]
            if len(actual) == 1
            else []
        )
        # Parse the actual transported episodic JSON; no substring proxy for
        # completeness, scope or role. The full original quotation is unchanged.
        packets = []
        for wire in wires:
            for match in re.finditer(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S):
                packets.append(json.loads(match[1]))
        records = [r for packet in packets for r in packet.get("records", []) if r.get("source_id") == sid]
        reply = read["response"].get("replyText", "")
        checks.update(
            actual_read_model=len(actual) == 1 and fixture["retained_read_message"] in wires[-1],
            actual_read_history_empty=proof["generation_diagnostics"][-1]["history"] == [],
            full_new_source_in_read_packet=len(records) == 1 and records[0]["text"] == fixture["new_native_source"],
            source_role_unresolved=bool(packets)
            and all(
                p["speaker_role"] == "user" and p["described_subject"] == p["current_validity"] == "not_resolved"
                for p in packets
            ),
            source_not_system_rule=all(
                fixture["new_native_source"] not in m["content"]
                for m in actual[0]["request"]["messages"]
                if m.get("role") == "system"
            ),
            course_facts_understood=all(x in reply for x in ["MC-845-D", "2026", "12月5日", "周六", "海庭鹤林", "有效"])
            and bool(re.search("(?:尚未|没有|未).{0,5}参加", reply))
            and bool(re.search("(?:尚未|没有|未).{0,5}出发", reply)),
            historical_quote_not_new_preference=bool(
                re.search("(?:历史|以前|过去).{0,10}(?:引用|发言)|(?:引用|发言).{0,12}(?:历史|以前|过去)", reply)
            )
            and bool(
                re.search("(?:不是|并非|不构成|没有).{0,18}(?:重新|更新|新增|声明)|(?:不是|并非).{0,18}偏好", reply)
            ),
            new_recheck_understood=bool(re.search("(?:再次|重新|最近|又).{0,12}核对|核对.{0,12}(?:再次|最近)", reply)),
            read_delivery_completed=proof["read_terminal_receipt"]["context"]["memory_completion"]["state"]
            == "completed",
            existing_claims_still_unpromoted=proof["owner_after_read"]["claims"] == before["claims"],
        )
    proof["authority_guards"] = {
        k: v
        for k, v in checks.items()
        if k
        not in {
            "full_actual_source",
            "exact_source_only_completion",
            "retained_read_executed",
            "full_new_source_in_read_packet",
            "course_facts_understood",
            "historical_quote_not_new_preference",
            "new_recheck_understood",
        }
    }
    return checks
