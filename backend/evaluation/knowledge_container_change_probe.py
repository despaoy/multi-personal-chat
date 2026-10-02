"""Retain full parent rules while real administrative paths change."""

import asyncio
import hashlib
import json
import os
import sys
from copy import deepcopy

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_container_probe(
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
    from knowledge.vector_db import get_vector_db

    async def parent_snapshot():
        c = await asyncpg.connect(
            user="boot", database="stage3_stage49_mutation_fixed", host=str(output.parent / "socket"), port=25433
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

    parent_before = await parent_snapshot()
    assert all(source_proof["checks"].values()) and len(source_proof["checks"]) == 22
    proof["parent_verified_checks"] = 22
    proof["document_imports_replayed"] = 0
    proof["reader_pid"] = os.getpid()
    proof["reader_database"] = database.execute_sql("SELECT current_database() AS name", {})[0]["name"]
    me = await client.get("/api/auth/me")
    proof["actual_account_role"] = me.json()["user"]["role"]
    assert me.status_code == 200 and proof["actual_account_role"] == "admin"
    initial = deepcopy(source_proof["final_documents"])
    proof["initial_parent_documents"] = initial
    doc_id = initial[0]["id"]
    kb_id = initial[0]["knowledge_base_id"]
    proof["actual_target_document_id"] = doc_id
    proof["actual_target_kb_id"] = kb_id
    proof["initial_base"] = database.get_knowledge_base(kb_id)
    for d, f in zip(initial, fixture["documents"]):
        assert (
            database.get_knowledge_document(d["id"]) == d and d["content"] == f["content"] and d["title"] == f["title"]
        )
    proof["initial_chunks"] = deepcopy(source_proof["final_chunks"])
    for d in initial:
        assert database.get_knowledge_chunks(d["id"]) == proof["initial_chunks"][str(d["id"])]
    folder = await client.post(f"/api/knowledge/bases/{kb_id}/folders", json={"name": fixture["new_folder_name"]})
    assert folder.status_code == 200
    folder_id = folder.json()["folder"]["id"]
    proof["actual_folder_id"] = folder_id
    attached = await client.put(
        f"/api/knowledge/documents/{doc_id}", json={"folder_id": folder_id, "category": fixture["new_folder_name"]}
    )
    assert attached.status_code == 200
    proof["new_setup_api_requests"] = 2
    vector = get_vector_db()

    def snapshot():
        return {
            "base": database.get_knowledge_base(kb_id),
            "folder": database.get_knowledge_folder(folder_id),
            "document": database.get_knowledge_document(doc_id),
            "chunks": database.get_knowledge_chunks(doc_id),
            "revision": knowledge._get_rebuild_revision(),
            "status": knowledge._read_rebuild_status(),
            "vector_metadata": deepcopy(vector.metadata),
            "cache_generation": vector.cache_generation,
        }

    warm = await client.post("/api/knowledge/search", json={"query": fixture["search_query"], "topK": 3})
    assert warm.status_code == 200
    for case in fixture["cases"]:
        await generate._retrieve_rag_bundle(case["message"], 3, None)
    proof["initial_state"] = snapshot()
    proof["writers"] = {}
    proof["writer_handles"] = {}
    proof["observations"] = []
    for label, case in zip(["rename", "detach"], fixture["cases"]):
        assert case["message"] not in [old["message"] for old in source_proof["cases"]]
        with (output / (label + ".log")).open("xb") as log:
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "evaluation.knowledge_container_change_writer",
                stdin=asyncio.subprocess.PIPE,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            (output / (label + "-job.json")).write_text(json.dumps({"pid": child.pid, "label": label}))
            child.stdin.write(
                json.dumps(
                    {
                        "output": str(output),
                        "label": label,
                        "password": password,
                        "kb_id": kb_id,
                        "folder_id": folder_id,
                        "doc_id": doc_id,
                        "new_base_name": fixture["new_base_name"],
                    }
                ).encode()
            )
            await child.stdin.drain()
            child.stdin.close()
            await child.wait()
        proof["writer_handles"][label] = {
            "pid": child.pid,
            "returncode": child.returncode,
            "terminal": child.returncode is not None,
        }
        assert child.returncode == 0, "Inspect private writer log"
        proof["writers"][label] = json.loads((output / (label + "-result.json")).read_text())
        before = snapshot()
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
        proof["observations"].append({"label": label, "before_generation": before, "after_generation": snapshot()})
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
    proof["final_documents"] = [database.get_knowledge_document(d["id"]) for d in initial]
    proof["final_chunks"] = {str(d["id"]): database.get_knowledge_chunks(d["id"]) for d in initial}
    proof["parent_source_unchanged"] = parent_before == await parent_snapshot()
    capture_storage_proof(proof, output)
    from evaluation.knowledge_container_change_audit import audit_container

    proof["checks"] = audit_container(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]
