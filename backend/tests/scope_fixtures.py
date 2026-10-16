"""Explicit open-scope controls for tests that own other contracts."""

import json


def open_scope_response(messages, rows, **options):
    """Select the requested wire contract, never infer scope from query words."""
    payload = json.loads(messages[-1]["content"])
    if "scope_policy" in payload and isinstance(rows, list):
        rows = [
            dict(row, retrieval_scope="open") if isinstance(row, dict) and "retrieval_scope" not in row else row
            for row in rows
        ]
    return json.dumps(dict(scopes=rows), **options)


def literal_span_response(payload, value):
    """Select the actual same-source reference for an explicit synthetic quote."""
    from copy import deepcopy

    value = deepcopy(value)
    for row in value["decisions"]:
        for proof in row["object_evidence"]:
            quote = proof.pop("source_quote")
            matches = [
                ref
                for ref in payload["source_span_catalog"]
                if ref["source_id"] == row["source_id"] and quote in ref["source_quote"]
            ]
            assert len(matches) == 1, "Synthetic proof must select one actual source reference"
            proof["source_span_id"] = matches[0]["span_id"]
    return json.dumps(value, ensure_ascii=False)


def install_public_source_reader(monkeypatch, generate, bundle):
    """The send-time reader sees the same complete synthetic source snapshots."""
    from knowledge.original_sources import document_authority_snapshot

    documents = {}
    for row in bundle["results"]:
        identity = row["document_id"]
        packets = [packet for packet in bundle["original_source_packets"] if packet["original_source_id"] == f"doc_{identity}"]
        assert len(packets) == 1
        documents[identity] = dict(row, id=identity, content=packets[0]["original_body"])
    for record in bundle["source_coverage"]:
        snapshot = record["source_authority_snapshot"]
        assert document_authority_snapshot(documents[snapshot["document_id"]]) == snapshot
    monkeypatch.setattr(generate.db, "get_knowledge_document", lambda identity: documents[identity])
