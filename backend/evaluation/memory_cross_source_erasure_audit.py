"""Audit saved native evidence for retaining a separate historical source."""

import json
import re
from html import unescape


def augment_cross_source_erasure(proof, fixture, cloud_calls, checks):
    before = proof["owner_before"]
    after = proof["owner_after_ack"]
    sid = fixture["unlinked_historical_source_id"]
    originals = [x for x in before["sources"] if x["source_message_id"] == sid]
    projections = [x for x in after["claims"] if x["memory_key"].startswith("source_fragment_")]
    projection = projections[0] if len(projections) == 1 else {}
    metadata = json.loads(projection.get("metadata_json") or "{}")
    fragment_info = metadata.get("erasure_evidence_projection", {})
    original = originals[0] if len(originals) == 1 else {}
    text = original.get("body", "")
    quote = fixture["historical_erased_quote"]
    index = text.find(quote)
    expected_spans = [[0, index], [index + len(quote), len(text)]]
    expected_evidence = text.split(quote) if text.count(quote) == 1 else []
    checks.update(
        real_unlinked_historical_source=original.get("body") == fixture["unlinked_historical_source"]
        and not any(x["source_message_id"] == sid for x in before["source_links"]),
        all_prior_sources_unreadable_after_ack=not any(
            x["source_message_id"] in fixture["all_prior_source_ids"] for x in after["sources"]
        ),
        literal_historical_source_fragments_and_clock=bool(expected_evidence)
        and projection.get("source_message_id") == sid
        and json.loads(projection.get("source_message_ids_json") or "[]") == [sid]
        and projection.get("observed_at") == original.get("observed_at")
        and json.loads(projection.get("evidence_json") or "[]") == expected_evidence
        and fragment_info
        == dict(
            version=1,
            kind="original_source_fragments",
            sources=[dict(source_message_id=sid, spans=expected_spans)],
            complete_original_source=False,
        ),
        fragment_not_promoted_to_current_fact=projection.get("memory_type") == "shared_event"
        and metadata.get("content_semantics") == "quoted_source"
        and metadata.get("speaker_role") == "user"
        and metadata.get("described_subject") == "not_resolved"
        and metadata.get("temporal_provenance")
        == dict(version=1, producer="source_erasure_projection", validity_authority="unspecified"),
        real_projection_source_link=bool(projection)
        and any(x["source_message_id"] == sid and x["memory_id"] == projection["id"] for x in after["source_links"]),
    )
    if proof.get("retained_read_executed"):
        read = proof["generation"][1]
        actual = [c for c in cloud_calls[slice(*read["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        wire = unescape(actual[0]["request"]["messages"][-1]["content"]) if len(actual) == 1 else ""
        memory = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
        packets = [json.loads(line[2:]) for line in memory[1].splitlines() if line.startswith("- {")] if memory else []
        blocks = [
            json.loads(m[1]) for m in re.finditer(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
        ]
        records = [r for p in blocks for r in p.get("records", [])]
        preserved_text = json.dumps([packets, records], ensure_ascii=False)
        reply = read["response"].get("replyText", "")
        checks.update(
            actual_read_history_empty=proof["generation_diagnostics"][-1]["history"] == [],
            no_erased_preference_in_any_saved_reference="深蓝色" not in preserved_text,
            old_unlinked_quote_not_raw_recalled=not any(
                r["source_id"] in fixture["all_prior_source_ids"] for r in records
            ),
            retained_later_recheck_source_present=any(
                fixture["retained_recheck_prefix"] in str(x) for p in packets for x in p.get("evidence", [])
            ),
            retained_later_recheck_understood=bool(
                re.search(r"(?:再次|重新|最近|又).{0,12}核对|核对.{0,12}(?:再次|最近)", reply)
            )
            and not bool(
                re.search(
                    r"(?:没|未|无法|不能).{0,18}(?:核对记录|再次核对|再.{0,4}核对)|(?:核对记录|再次核对).{0,18}(?:没|未|无法|不能)",
                    reply,
                )
            ),
        )
    proof["cross_source_checks"] = {
        k: v
        for k, v in checks.items()
        if k not in proof.get("authority_guards", {})
        and k not in proof.get("business_checks", {})
        and k not in proof.get("read_checks", {})
    }
    return checks
