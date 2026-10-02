"""Saved evidence audit for selective deletion of an actual source fragment."""

import json
import re
from html import unescape


def augment_fragment_lifecycle(proof, fixture, cloud_calls, checks):
    before = proof["owner_before"]
    after = proof["owner_after_ack"]
    old = [r for r in before["claims"] if r["memory_key"] == fixture["erased_memory_key"]]
    checks.update(
        real_prior_partial_source_observation=len(old) == 1
        and old[0]["id"] == 8
        and json.loads(old[0]["metadata_json"])["erasure_evidence_projection"]["complete_original_source"] is False,
        real_disjoint_quote_anchors=bool(old)
        and fixture["erase_anchor"] in str(old[0]["evidence_json"])
        and fixture["keep_anchor"] not in str(old[0]["evidence_json"]),
        actual_later_fragment_deleted=not any(r["id"] == 8 for r in after["claims"]),
        no_memory_added_by_erasure={r["id"] for r in after["claims"]} <= {r["id"] for r in before["claims"]},
    )
    writers = []
    for call in cloud_calls:
        if call["request"].get("max_tokens") != 768:
            continue
        for message in call["request"]["messages"]:
            if message.get("role") != "user":
                continue
            try:
                payload = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            if payload.get("current_user_message") == fixture["cases"][-1]["message"]:
                writers.append(payload)
    expected = {str(row["id"]): row for row in before["claims"]}
    if len(writers) == 1:
        selected = writers[0].get("existing_memories", [])
        exact = {row["memory_id"] for row in selected} == set(expected)
        for row in selected:
            original = expected.get(row["memory_id"], {})
            packet = row.get("source_observation", {})
            exact &= packet.get("evidence") == json.loads(original.get("evidence_json") or "[]")
            exact &= packet.get("observed_at") == original.get("observed_at")
            exact &= packet.get("source_message_ids") == json.loads(original.get("source_message_ids_json") or "[]")
            exact &= (
                packet.get("complete_original_source") is False and packet.get("described_subject") == "not_resolved"
            )
        checks["actual_writer_receives_original_partial_sources_and_clocks"] = exact
        checks["actual_writer_logical_identifiers_not_truncated"] = all(
            row.get("memory_key") == expected.get(row["memory_id"], {}).get("memory_key") for row in selected
        ) and bool(selected)
    else:
        checks["actual_writer_receives_original_partial_sources_and_clocks"] = False
        checks["actual_writer_logical_identifiers_not_truncated"] = False
    if proof.get("retained_read_executed"):
        read = proof["generation"][1]
        reply = read["response"]["replyText"]
        checks["later_fragment_absence_understood"] = bool(
            re.search(
                r"(?:没|未|不再|不存在|已删除|已忘|不包含).{0,35}(?:核对|后来|片段)|(?:核对|后来|片段).{0,35}(?:没|未|不再|不存在|已删除|已忘|不包含)",
                reply,
            )
        )
        actual = [c for c in cloud_calls[slice(*read["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        wire = unescape(actual[0]["request"]["messages"][-1]["content"]) if len(actual) == 1 else ""
        block = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
        packets = [json.loads(line[2:]) for line in block[1].splitlines() if line.startswith("- {")] if block else []
        checks["actual_read_history_empty"] = proof["generation_diagnostics"][-1]["history"] == []
        checks["deleted_later_original_prefix_absent_from_actual_read"] = fixture["erase_anchor"] not in json.dumps(
            packets, ensure_ascii=False
        )
        checks["retained_original_confirmation_prefix_in_actual_read"] = any(
            fixture["keep_anchor"] in str(x) for p in packets for x in p.get("evidence", [])
        )
    return checks
