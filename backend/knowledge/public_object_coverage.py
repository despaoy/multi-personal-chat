"""Settle each required object against the exact admitted evidence packets."""


def settle_public_object_coverage(retrieval, payload, scopes, scoped_decisions, decisions):
    sources = {s["source_id"]: s for s in payload["sources"]}
    objects = {o["object_id"]: o["query_text"] for o in scopes["objects"]}
    accepted = {d["source_id"]: set(d["task_ids"]) for d in decisions}
    proofs = {}
    for row in scoped_decisions:
        source = sources[row["source_id"]]
        chunks = source["indexed_chunks"]
        parts = [source.get("title"), source.get("original_body"), *[c["content"] for c in chunks]]
        allowed_ids = {c["id"] for c in chunks} | {row["source_id"] + "_original"}
        for proof in row["object_evidence"]:
            identity, quote = proof["object_id"], proof["source_quote"]
            if objects[identity] not in quote or not any(isinstance(part, str) and quote in part for part in parts):
                continue
            # A visible unrelated chunk from this source, or the same text from
            # another source, does not admit this source-object proof.
            admitted = retrieval.status == "ok" and any(
                isinstance(packet.get("text"), str)
                and quote in packet["text"]
                and bool(allowed_ids.intersection(packet.get("document_ids", ())))
                for packet in retrieval.admitted_evidence_packets or retrieval.evidence_packets
            )
            proofs[(row["source_id"], identity)] = admitted
    tasks = {}
    for task in scopes["task_scopes"]:
        rows = []
        for identity in task["object_ids"]:
            candidates = sorted(
                sid for sid, allowed in accepted.items() if task["task_id"] in allowed and (sid, identity) in proofs
            )
            visible = [sid for sid in candidates if proofs[(sid, identity)]]
            rows.append(
                dict(
                    object_id=identity,
                    query_text=objects[identity],
                    status="verified_object_evidence_admitted"
                    if visible
                    else "verified_object_evidence_not_admitted"
                    if candidates
                    else "no_verified_object_evidence",
                    source_ids=visible,
                    candidate_source_ids=candidates,
                )
            )
        tasks[task["task_id"]] = rows
    return tasks
