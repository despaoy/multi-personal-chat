"""Admission dependencies require actual referring evidence, not source IDs."""

import re

from inference.evidence_coverage import settle_source_coverage


def dependency_satisfied(packet, admitted_packets, source_coverage):
    if packet.get("kind") != "background":
        return True
    ids = {str(i) for p in admitted_packets for i in p.get("document_ids", ())}
    support = packet.get("supporting_document_ids")
    if not ids:
        return False
    if support is None:
        return True  # Legacy background still requires an actually admitted root.
    if not isinstance(support, (tuple, list)) or not support or not all(isinstance(i, str) for i in support):
        return False
    refs = packet.get("supporting_source_refs")
    if refs is None:
        return bool(ids.intersection(support))
    if not isinstance(refs, (tuple, list)) or not refs:
        raise ValueError("Invalid source dependency receipts")
    target = packet.get("source_id")
    kb = packet.get("knowledge_base_id")
    if not isinstance(target, str) or not re.fullmatch(r"doc_[1-9]\d*", target) or type(kb) is not int or kb <= 0:
        raise ValueError("Invalid dependent source scope")
    by_id = {}
    for ref in refs:
        keys = {"document_id", "source_id", "source_title", "knowledge_base_id", "relation", "target_title", "text"}
        if not isinstance(ref, dict) or set(ref) != keys:
            raise ValueError("Invalid source dependency receipt")
        identity, parent, text = ref["document_id"], ref["source_id"], ref["text"]
        if (
            not isinstance(parent, str)
            or not re.fullmatch(r"doc_[1-9]\d*", parent)
            or not isinstance(identity, str)
            or not re.fullmatch(re.escape(parent) + r"_chunk_(?:0|[1-9]\d*)", identity)
            or type(ref["knowledge_base_id"]) is not int
            or ref["knowledge_base_id"] != kb
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(ref["source_title"], str)
            or not ref["source_title"]
            or not isinstance(ref["target_title"], str)
            or not ref["target_title"]
        ):
            raise ValueError("Invalid referring source identity or scope")
        relation = ref["relation"]
        if relation == "same_source":
            if parent != target or ref["source_title"] != ref["target_title"]:
                raise ValueError("Invalid same-source dependency")
        elif relation == "title_reference":
            if parent == target or "《" + ref["target_title"] + "》" not in text:
                raise ValueError("Invalid literal title dependency")
        else:
            raise ValueError("Invalid source dependency relation")
        if identity in by_id and by_id[identity] != ref:
            raise ValueError("Conflicting source dependency receipts")
        by_id[identity] = ref
    if set(by_id) != set(support):
        raise ValueError("Dependency identities differ from support")
    originals = {
        r["source_id"]: r
        for r in settle_source_coverage(source_coverage, ids, admitted_packets=admitted_packets)
        if r.get("original_status") == "verified_original_body_admitted"
    }
    for identity, ref in by_id.items():
        if any(identity in p.get("document_ids", ()) and ref["text"] in p.get("text", "") for p in admitted_packets):
            return True
        original = originals.get(ref["source_id"])
        if (
            original is None
            or identity not in original["indexed_document_ids"]
            or original.get("source_title") != ref["source_title"]
        ):
            continue
        if any(
            p.get("original_source_id") == ref["source_id"]
            and p.get("knowledge_base_id") == kb
            and ref["text"] in p.get("original_body", "")
            for p in admitted_packets
        ):
            return True
    return False
