"""Recheck the same normal knowledge documents at each model-send boundary.

Snapshots are internal, captured by the fresh retrieval read. A changed source
cannot substitute a new body into a review of the previous body.
"""

import asyncio
import re
from dataclasses import replace

from knowledge.original_sources import document_authority_snapshot


def _root(identity):
    match = re.fullmatch(r"(doc_[1-9]\d*)(?:_chunk_(?:0|[1-9]\d*)|_original)?", str(identity))
    return match[1] if match else None


def _recheck(records, read_document):
    granted = set()
    seen = set()
    for record in records:
        source_id = record.get("source_id")
        snapshot = record.get("source_authority_snapshot")
        if source_id in seen:
            raise ValueError("Duplicate public source authority record")
        seen.add(source_id)
        if not isinstance(snapshot, dict) or snapshot.get("source_id") != source_id:
            raise ValueError("Invalid public source authority snapshot")
        identity = snapshot["document_id"]
        if isinstance(identity, bool) or not isinstance(identity, int) or source_id != f"doc_{identity}":
            raise ValueError("Invalid public source authority identity")
        document = read_document(identity)
        # An absent or changed document revokes its old grant; a failed read
        # propagates so the caller cannot report a successful evidence check.
        if document is not None and document_authority_snapshot(document) == snapshot:
            granted.add(source_id)
    return granted


def _prune(retrieval, granted):
    candidates = retrieval.evidence_packets or retrieval.admitted_evidence_packets
    roots = {
        root
        for packet in candidates
        for identity in packet.get("document_ids", ())
        if (root := _root(identity)) is not None
    }
    stale = roots - granted
    if not stale:
        return retrieval
    removed = {
        identity for packet in candidates for identity in packet.get("document_ids", ()) if _root(identity) in stale
    }
    # A removed registry/referring packet also revokes dependent background,
    # including descendants belonging to otherwise unchanged documents.
    while True:
        expanded = removed | {
            identity
            for packet in candidates
            if packet.get("kind") == "background"
            and (
                removed.intersection(packet.get("supporting_document_ids", ()))
                or any(ref.get("referring_source_id") in stale for ref in packet.get("supporting_source_refs", ()))
            )
            for identity in packet.get("document_ids", ())
        }
        if expanded == removed:
            break
        removed = expanded

    def keep(packet):
        return not removed.intersection(packet.get("document_ids", ()))

    packets = tuple(packet for packet in candidates if keep(packet))
    admitted = tuple(packet for packet in retrieval.admitted_evidence_packets if keep(packet))
    evidence = "\n\n".join(packet["text"] for packet in packets)
    ids = {identity for packet in packets for identity in packet.get("document_ids", ())}

    def metadata_kept(row):
        identity = row.get("id") or row.get("chunk_id") or row.get("source_id")
        return _root(identity) not in stale and identity not in removed

    requests = []
    for row in retrieval.requested_sources:
        remaining = [sid for sid in row["source_ids"] if sid not in stale]
        if len(remaining) != len(row["source_ids"]):
            requests.append(
                dict(row, source_ids=remaining, lookup_status="matched_in_index_scope" if remaining else "not_resolved")
            )
        else:
            requests.append(row)
    references = tuple(
        row
        for row in retrieval.source_references
        if row.get("referring_source_id") not in stale and not stale.intersection(row.get("source_ids", ()))
    )
    return replace(
        retrieval,
        status=retrieval.status if evidence else "character_abstention",
        reason="public_source_authority_changed",
        evidence=evidence,
        evidence_packets=packets,
        admitted_evidence_packets=admitted,
        documents=tuple(
            row for row in retrieval.documents if metadata_kept(row) and (row.get("id") or row.get("chunk_id")) in ids
        ),
        citations=tuple(
            row for row in retrieval.citations if metadata_kept(row) and (row.get("id") or row.get("source_id")) in ids
        ),
        source_excerpts=tuple(row for row in retrieval.source_excerpts if metadata_kept(row)),
        requested_sources=tuple(requests),
        source_references=references,
    )


def make_public_context_revalidator(read_document):
    """Bind to the same database reader used by generic retrieval, not a caller DB."""

    async def revalidate(request):
        retrieval = request.retrieval
        if not retrieval.evidence_packets and not retrieval.admitted_evidence_packets:
            return request
        granted = await asyncio.to_thread(_recheck, retrieval.source_coverage, read_document)
        updated = _prune(retrieval, granted)
        return request if updated is retrieval else replace(request, retrieval=updated)

    return revalidate
