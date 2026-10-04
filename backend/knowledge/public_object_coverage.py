"""Settle each required object against the exact admitted evidence packets."""


def settle_public_object_coverage(retrieval, payload, scopes, scoped_decisions, decisions):
    from knowledge.public_identity_dependencies import quote_admitted, validate_identity_receipt, verified_object_proof

    identity_review = validate_identity_receipt(payload, scopes)
    bindings = {b["binding_id"]: b for b in identity_review["bindings"]}
    packets = retrieval.admitted_evidence_packets or retrieval.evidence_packets
    sources = {s["source_id"]: s for s in payload["sources"]}
    objects = {o["object_id"]: o["query_text"] for o in scopes["objects"]}
    accepted = {d["source_id"]: set(d["task_ids"]) for d in decisions}
    proofs = {}
    for row in scoped_decisions:
        source = sources[row["source_id"]]
        for proof in row["object_evidence"]:
            identity, quote = proof["object_id"], proof["source_quote"]
            if not verified_object_proof(proof, source, objects, bindings):
                continue
            # A visible unrelated chunk from this source, or the same text from
            # another source, does not admit this source-object proof.
            admitted = retrieval.status == "ok" and quote_admitted(source, quote, packets)
            if "identity_binding_id" in proof:
                binding = bindings[proof["identity_binding_id"]]
                admitted = admitted and quote_admitted(sources[binding["source_id"]], binding["source_quote"], packets)
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
