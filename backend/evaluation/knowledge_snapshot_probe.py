"""A new ordinary generation on real, concurrently published complete sources."""

import hashlib
import json
import shutil
from copy import deepcopy

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_snapshot_probe(
    client, args, proof, fixture, cloud_calls, *, database, runtime, output, capture_storage_proof, source_proof
):
    import asyncpg

    from api import knowledge
    from knowledge import vector_db

    async def parent_snapshot():
        c = await asyncpg.connect(
            user="boot", database="stage3_stage50_container_fixed", host=str(output.parent / "socket"), port=25433
        )
        try:
            assert await c.fetchval("SHOW data_directory") == str(output.parent / "data")
            return {
                t: hashlib.sha256(
                    json.dumps(
                        sorted(
                            [dict(row) for row in await c.fetch("SELECT * FROM " + t)],
                            key=lambda row: json.dumps(row, sort_keys=True, default=str),
                        ),
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest()
                for t in ["knowledge_bases", "knowledge_folders", "knowledge_documents", "knowledge_chunks", "config"]
            }
        finally:
            await c.close()

    before = await parent_snapshot()
    assert all(source_proof["checks"].values()) and len(source_proof["checks"]) == 34
    proof["parent_verified_checks"] = 34
    docs = deepcopy(source_proof["final_documents"])
    proof["initial_parent_documents"] = docs
    proof["initial_chunks"] = deepcopy(source_proof["final_chunks"])
    assert [database.get_knowledge_document(d["id"]) for d in docs] == docs
    for d, f in zip(docs, fixture["documents"]):
        assert d["content"] == f["content"] and d["title"] == f["title"]
    vector = vector_db.get_vector_db()
    proof["legacy"] = {
        "stats": vector.get_stats(),
        "validated": vector.snapshot_validated,
        "may_skip": knowledge._loaded_metadata_matches_database(vector),
    }
    response = await client.post("/api/knowledge/search", json={"query": "LL-842-Q", "topK": 3})
    proof["migration"] = {
        "http_status": response.status_code,
        "validated": vector.snapshot_validated,
        "revision": knowledge._get_rebuild_revision(),
        "status": knowledge._read_rebuild_status(),
    }
    assert response.status_code == 200 and vector.snapshot_validated
    folder = output.parent / "stage51-storage-fixed"
    storage = json.loads((folder / "result.json").read_text())
    assert all(storage["checks"].values()) and storage["all_writer_handles_terminal"]
    proof["storage_concurrency"] = storage
    source = folder / "shared-vectors/index_snapshot.zip"
    destination = output / "vectors/index_snapshot.zip"
    proof["published_snapshot_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(source, destination)
    proof["copied_snapshot_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    # Start a genuinely cold vector reader on the saved real writer snapshot.
    vector = vector_db.VectorDatabase(str(output / "vectors"))
    vector_db._vector_db = vector
    knowledge._vector_index_built = False
    knowledge._vector_index_revision = None
    proof["cold_snapshot"] = {
        "validated": vector.snapshot_validated,
        "metadata_ids": [d["id"] for d in vector.metadata],
        "corpus_aligned": vector.bm25.corpus == [d["title"] + " " + d["content"] for d in vector.metadata],
    }
    case = fixture["cases"][0]
    assert case["message"] not in [c["message"] for c in source_proof["cases"]]
    start = len(cloud_calls)
    response = await client.post(
        "/api/generate",
        json={
            "message": case["message"],
            "characterId": "tsukiyashiro_kisaki",
            "loraId": "default",
            "sessionId": args.run_label,
            "sessionType": "private",
        },
    )
    assert response.status_code == 200
    proof["generation"].append(
        {
            "id": case["id"],
            "http_status": response.status_code,
            "response": response.json(),
            "cloud_call_range": [start, len(cloud_calls)],
        }
    )
    (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    scheduler = get_memory_enrichment_scheduler()
    assert await scheduler.flush_memory(timeout=90)
    completion = get_turn_completion_runtime()
    await completion.shutdown(timeout=35)
    assert await scheduler.flush_memory(timeout=90)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
    proof["completion_runtime_terminal"] = {"active": completion.active, "reserved": completion.reserved}
    proof["queue_stats"] = runtime.stats()
    proof["sync_pending_final"] = len(database._pending)
    proof["final_documents"] = [database.get_knowledge_document(d["id"]) for d in docs]
    proof["final_chunks"] = {str(d["id"]): database.get_knowledge_chunks(d["id"]) for d in docs}
    proof["parent_source_unchanged"] = before == await parent_snapshot()
    capture_storage_proof(proof, output)
    from evaluation.knowledge_snapshot_audit import audit_snapshot

    proof["checks"] = audit_snapshot(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values())
