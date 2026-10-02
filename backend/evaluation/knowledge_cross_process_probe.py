"""Warm reader, real separate writer rebuild, then one new pro answer."""

import asyncio
import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_cross_process_probe(
    client,
    args,
    proof,
    fixture,
    cloud_calls,
    *,
    database,
    runtime,
    output,
    capture_storage_proof,
    password,
    source_proof,
):
    import asyncpg

    from api import generate, knowledge
    from knowledge.rag_helper import get_rag_helper
    from knowledge.vector_db import get_vector_db

    async def parent_snapshot():
        connection = await asyncpg.connect(
            user="boot", database="stage3_stage45_scope_fixed", host=str(output.parent / "socket"), port=25433
        )
        try:
            assert await connection.fetchval("SHOW data_directory") == str(output.parent / "data")
            hashes = {}
            for table in ["knowledge_documents", "knowledge_chunks", "config"]:
                rows = [dict(r) for r in await connection.fetch("SELECT * FROM " + table)]
                hashes[table] = hashlib.sha256(
                    json.dumps(
                        sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str)),
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest()
            return hashes
        finally:
            await connection.close()

    source_before = await parent_snapshot()
    proof["reader_pid"] = os.getpid()
    proof["reader_database_name"] = database.execute_sql("SELECT current_database() AS name", {})[0]["name"]
    proof["reader_vector_path"] = str(output / "vectors")
    proof["writer_vector_path"] = str(output / "writer-vectors")
    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    proof["actual_account_role"] = me.json()["user"]["role"]
    assert proof["actual_account_role"] == "admin"
    assert len(source_proof["checks"]) == 25 and all(source_proof["checks"].values())
    assert source_proof["final_document"]["content"] == fixture["documents"][0]["content"]
    bases = database.get_knowledge_bases()
    proof["actual_base_ids"] = [
        next(b["id"] for b in bases if b["name"] == name) for name in fixture["knowledge_bases"]
    ]
    proof["documents"] = deepcopy(source_proof["documents"])
    proof["document_imports_replayed"] = 0
    doc_id = source_proof["actual_target_document_id"]
    proof["actual_target_document_id"] = doc_id
    assert database.get_knowledge_document(doc_id) == source_proof["final_document"]
    assert database.get_knowledge_chunks(doc_id) == source_proof["final_chunks"]
    for row in proof["documents"]:
        saved = row["response"]["document"]
        actual = database.get_knowledge_document(saved["id"])
        assert actual["content"] == saved["content"]
        assert "".join(c["content"] for c in database.get_knowledge_chunks(saved["id"])) == saved["content"]
    proof["parent_complete_fixture_verified"] = True
    proof["new_query_distinct_from_parent"] = fixture["cases"][0]["message"] not in [
        c["message"] for c in source_proof["cases"]
    ]
    assert proof["new_query_distinct_from_parent"]

    async def search(label, base):
        response = await client.post(
            "/api/knowledge/search", json=dict(query=fixture["search_query"], topK=3, knowledgeBaseName=base)
        )
        assert response.status_code == 200
        row = dict(label=label, http_status=response.status_code, response=response.json())
        proof["searches"].append(row)
        return row

    warm = await search("original_scope_warm", fixture["knowledge_bases"][0])
    assert any(r["documentId"] == f"doc_{doc_id}_chunk_0" for r in warm["response"]["results"])
    # Populate the actual RAG helper cache with the new query, without a model answer.
    await generate._retrieve_rag_bundle(fixture["cases"][0]["message"], 3, None)
    vector = get_vector_db()

    def snapshot():
        return dict(
            document=database.get_knowledge_document(doc_id),
            chunks=database.get_knowledge_chunks(doc_id),
            vector_metadata=deepcopy(vector.metadata),
            cache_generation=vector.cache_generation,
            revision=knowledge._get_rebuild_revision(),
            status=knowledge._read_rebuild_status(),
            index_built=knowledge._vector_index_built,
        )

    proof["before_update"] = snapshot()
    proof["rag_cache_warmed_entries"] = len(get_rag_helper()._query_cache)
    shutil.copytree(output / "vectors", output / "writer-vectors")
    payload = dict(**fixture["metadata_update"], knowledge_base_id=proof["actual_base_ids"][1])
    proof["actual_update_payload"] = payload
    with (output / "writer.log").open("xb") as log:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "evaluation.knowledge_cross_process_writer",
            stdin=asyncio.subprocess.PIPE,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        (output / "writer-job.json").write_text(json.dumps(dict(pid=child.pid)))
        await child.communicate(
            json.dumps(
                dict(
                    output=str(output),
                    password=password,
                    doc_id=doc_id,
                    payload=payload,
                    query=fixture["search_query"],
                    new_base=fixture["knowledge_bases"][1],
                )
            ).encode()
        )
    proof["writer_handle"] = dict(pid=child.pid, returncode=child.returncode, terminal=child.returncode is not None)
    assert child.returncode == 0, "Separate authenticated writer failed; inspect its private log"
    writer = json.loads((output / "writer-result.json").read_text())
    proof["writer"] = writer
    proof["metadata_update_response"] = writer["update_response"]
    proof["immediately_after_update"] = snapshot()
    # No reader-side search or manual index repair between writer completion and generation.
    before = len(cloud_calls)
    response = await client.post(
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
            http_status=response.status_code,
            response=response.json(),
            cloud_call_range=[before, len(cloud_calls)],
        )
    )
    (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    assert response.status_code == 200
    proof["after_generation"] = snapshot()
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
    proof["parent_source_unchanged"] = source_before == await parent_snapshot()
    capture_storage_proof(proof, output)
    from evaluation.knowledge_cross_process_audit import audit_cross_process

    proof["checks"] = audit_cross_process(proof, fixture, cloud_calls)
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
