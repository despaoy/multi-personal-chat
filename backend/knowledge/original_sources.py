"""Fresh authoritative original bodies, independently budgeted alongside indexed context."""

import hashlib
import re

from inference.context_budget import estimated_tokens


def attach_original_sources(bundle, read_document, *, source_budget_tokens, authority_revision):
    if bundle.get("abstained") or not bundle.get("results"):
        return bundle
    if not isinstance(authority_revision, int) or isinstance(authority_revision, bool) or authority_revision < 0:
        raise ValueError("Invalid original source authority revision")
    if not isinstance(source_budget_tokens, int) or isinstance(source_budget_tokens, bool) or source_budget_tokens <= 0:
        raise ValueError("Invalid original source budget")
    results = bundle["results"]
    remaining = max(0, source_budget_tokens - sum(estimated_tokens(r["content"]) + 4 for r in results))
    packets = []
    coverage = []
    for record in bundle.get("source_coverage", ()):
        source_id = record.get("source_id")
        doc_match = re.fullmatch(r"doc_([1-9]\d*)", source_id or "")
        if not doc_match:
            raise ValueError("Invalid original source identity")
        identity = int(doc_match[1])
        members = [r for r in results if r.get("document_id") == identity]
        if not members:
            raise ValueError("Original source has no authorized retrieved identity")
        document = read_document(identity)
        if not isinstance(document, dict) or document.get("id") != identity:
            raise RuntimeError("Original source no longer exists")
        for member in members:
            if any(document.get(key) != member.get(key) for key in ["title", "category", "knowledge_base_id"]):
                raise RuntimeError("Original source scope changed during retrieval")
        body = document.get("content")
        title = document.get("title")
        if not isinstance(body, str) or not body.strip() or not isinstance(title, str):
            # An unavailable optional original is not authority for full text.
            # Keep independently authorized indexed material and its limits.
            coverage.append({**record, "original_unverified_reason": "original_body_unavailable"})
            continue
        packet_id = source_id + "_original"
        text = f"【核对原始正文: {title}】\n{body}"
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        packet_hash = hashlib.sha256(text.encode()).hexdigest()
        packet = {
            "kind": "evidence",
            "document_ids": [packet_id],
            "text": text,
            "original_source_id": source_id,
            "original_body": body,
        }
        cost = estimated_tokens(text) + 4
        included = cost <= remaining
        if included:
            packets.append(packet)
            remaining -= cost
        receipt = {
            "version": 1,
            "authority": "fresh_knowledge_document_read",
            "source_id": source_id,
            "document_id": identity,
            "authority_revision": authority_revision,
            "original_body_sha256": body_hash,
            "original_body_chars": len(body),
            "original_packet_id": packet_id,
            "original_packet_sha256": packet_hash,
            "candidate_included": included,
        }
        coverage.append({**record, "original_source_receipt": receipt})
    return {**bundle, "source_coverage": tuple(coverage), "original_source_packets": tuple(packets)}
