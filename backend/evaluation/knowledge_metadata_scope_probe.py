"""One new metadata-only document migration, real retrieval and pro answer."""

import json
from copy import deepcopy

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_metadata_scope_probe(
    client, args, proof, fixture, cloud_calls, *, database, runtime, output, capture_storage_proof
):
    from api import knowledge
    from knowledge.vector_db import get_vector_db

    vector = get_vector_db()
    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    proof["actual_account_role"] = me.json()["user"]["role"]
    assert proof["actual_account_role"] == "admin"
    bases = []
    for name in fixture["knowledge_bases"]:
        response = await client.post(
            "/api/knowledge/bases", json=dict(name=name, description="Complete synthetic metadata migration boundary")
        )
        assert response.status_code == 200
        bases.append(response.json()["base"]["id"])
    proof["actual_base_ids"] = bases
    for i, document in enumerate(fixture["documents"]):
        response = await client.post(
            "/api/knowledge/documents", json=dict(**document, knowledge_base_id=bases[0 if i == 0 else 1])
        )
        proof["documents"].append(dict(http_status=response.status_code, response=response.json()))
        assert response.status_code == 200
    doc_id = proof["documents"][0]["response"]["document"]["id"]
    proof["actual_target_document_id"] = doc_id
    query = fixture["search_query"]

    async def search(label, base):
        response = await client.post("/api/knowledge/search", json=dict(query=query, topK=3, knowledgeBaseName=base))
        row = dict(label=label, http_status=response.status_code, response=response.json())
        proof["searches"].append(row)
        assert response.status_code == 200
        return row

    warm = await search("original_scope_warm", fixture["knowledge_bases"][0])
    assert any(r["documentId"] == f"doc_{doc_id}_chunk_0" for r in warm["response"]["results"])
    assert knowledge._vector_index_built
    proof["before_update"] = dict(
        document=database.get_knowledge_document(doc_id),
        chunks=database.get_knowledge_chunks(doc_id),
        vector_metadata=deepcopy(vector.metadata),
        cache_generation=vector.cache_generation,
        revision=knowledge._get_rebuild_revision(),
        status=knowledge._read_rebuild_status(),
        index_built=knowledge._vector_index_built,
    )
    payload = dict(**fixture["metadata_update"], knowledge_base_id=bases[1])
    proof["actual_update_payload"] = payload
    updated = await client.put(f"/api/knowledge/documents/{doc_id}", json=payload)
    proof["metadata_update_response"] = dict(http_status=updated.status_code, response=updated.json())
    assert updated.status_code == 200
    proof["immediately_after_update"] = dict(
        document=database.get_knowledge_document(doc_id),
        chunks=database.get_knowledge_chunks(doc_id),
        vector_metadata=deepcopy(vector.metadata),
        cache_generation=vector.cache_generation,
        revision=knowledge._get_rebuild_revision(),
        status=knowledge._read_rebuild_status(),
        index_built=knowledge._vector_index_built,
    )
    # No search/index repair between the actual update and ordinary generation.
    before = len(cloud_calls)
    generated = await client.post(
        "/api/generate",
        json=dict(
            message=fixture["cases"][0]["message"],
            characterId="tsukiyashiro_kisaki",
            loraId="default",
            sessionId=args.run_label,
            sessionType="private",
        ),
    )
    proof["generation"].append(
        dict(
            id=fixture["cases"][0]["id"],
            http_status=generated.status_code,
            response=generated.json(),
            cloud_call_range=[before, len(cloud_calls)],
        )
    )
    (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    assert generated.status_code == 200
    proof["after_generation"] = dict(
        vector_metadata=deepcopy(vector.metadata),
        cache_generation=vector.cache_generation,
        revision=knowledge._get_rebuild_revision(),
        status=knowledge._read_rebuild_status(),
        index_built=knowledge._vector_index_built,
    )
    await search("old_scope_after_update", fixture["knowledge_bases"][0])
    await search("new_scope_after_update", fixture["knowledge_bases"][1])
    scheduler = get_memory_enrichment_scheduler()
    assert await scheduler.flush_memory(timeout=90)
    completion = get_turn_completion_runtime()
    await completion.shutdown(timeout=35)
    assert await scheduler.flush_memory(timeout=90)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
    proof["completion_runtime_terminal"] = dict(
        active=completion.active, reserved=completion.reserved, failed=completion.failed, cancelled=completion.cancelled
    )
    proof["queue_stats"] = runtime.stats()
    proof["sync_pending_final"] = len(database._pending)
    proof["final_document"] = database.get_knowledge_document(doc_id)
    proof["final_chunks"] = database.get_knowledge_chunks(doc_id)
    capture_storage_proof(proof, output)
    from evaluation.knowledge_metadata_scope_audit import audit_metadata_scope

    proof["checks"] = audit_metadata_scope(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            dict(native_requests=len(proof["generation"]), cloud_calls=len(cloud_calls), checks=proof["checks"]),
            ensure_ascii=False,
        )
    )
    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]
