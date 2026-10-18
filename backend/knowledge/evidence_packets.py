"""Whole retrieved document chunks, independent of citation presentation."""

from collections.abc import Mapping, Sequence
from typing import Any


def evidence_source_id(value) -> str | None:
    """Normalize citable external IDs; absent or boolean IDs cannot bind sources."""
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() else None


def validate_evidence_packet(packet: Mapping[str, Any]) -> tuple[str, Sequence[str]]:
    """Read raw text and identities before annotation or budget admission."""
    text = packet.get("text")
    ids = packet.get("document_ids", ())
    if (not isinstance(text, str) or not text.strip() or not isinstance(ids, (list, tuple))
            or any(not isinstance(identity, str) or not identity.strip() for identity in ids)):
        raise ValueError("Invalid generation evidence packet: text and document identities must be valid")
    return text, ids


def document_evidence_packets(results: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    packets = []
    for document in results:
        body = document.get("content")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Retrieved document content must be non-empty text")
        identity = evidence_source_id(document.get("id") or document.get("chunk_id"))
        ids = [identity] if identity is not None else []
        title = document.get("title") or document.get("original_title") or "未命名资料"
        packet = {"kind": "evidence", "document_ids": ids, "text": f"【检索资料片段: {title}】\n{body}"}
        if document.get("retrieval_role") == "source_context":
            support = document.get("supporting_document_ids")
            if (not isinstance(support, (list, tuple)) or not support
                    or any(not isinstance(identity, str) or not identity.strip() for identity in support)):
                raise ValueError("Background evidence must identify supporting documents")
            packet.update(kind="background", supporting_document_ids=list(support))
            if document.get("supporting_source_refs") is not None:
                packet.update(
                    supporting_source_refs=document["supporting_source_refs"],
                    source_id=f"doc_{document['document_id']}",
                    knowledge_base_id=document["knowledge_base_id"],
                )
        packets.append(packet)
    return tuple(packets)
