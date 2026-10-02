"""Real dirty-write failure in a clone, independent API writer and warm reader."""

import asyncio
import hashlib
import json
import os
import sys
from copy import deepcopy

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_mutation_probe(
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

    async def connection(name):
        c = await asyncpg.connect(user="boot", database=name, host=str(output.parent / "socket"), port=25433)
        assert await c.fetchval("SHOW data_directory") == str(output.parent / "data")
        return c

    async def parent_snapshot():
        c = await connection("stage3_stage48_policy_fixed")
        try:
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
                for t in ["knowledge_documents", "knowledge_chunks", "config"]
            }
        finally:
            await c.close()

    before_parent = await parent_snapshot()
    assert all(source_proof["checks"].values()) and len(source_proof["checks"]) == 32
    proof["parent_verified_checks"] = 32
    proof["document_imports_replayed"] = 0
    proof["reader_pid"] = os.getpid()
    proof["reader_database"] = database.execute_sql("SELECT current_database() AS name", {})[0]["name"]
    proof["atomic_mutation_available"] = callable(getattr(database, "save_knowledge_document", None))
    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    proof["actual_account_role"] = me.json()["user"]["role"]
    assert proof["actual_account_role"] == "admin"
    proof["initial_documents"] = deepcopy(source_proof["final_documents"])
    proof["initial_chunks"] = deepcopy(source_proof["final_chunks"])
    doc_id = proof["initial_documents"][0]["id"]
    proof["actual_target_document_id"] = doc_id
    for d, f in zip(proof["initial_documents"], fixture["documents"]):
        assert (
            database.get_knowledge_document(d["id"]) == d and d["content"] == f["content"] and d["title"] == f["title"]
        )
        assert database.get_knowledge_chunks(d["id"]) == proof["initial_chunks"][str(d["id"])]
    vector = get_vector_db()

    def snapshot():
        return {
            "document": database.get_knowledge_document(doc_id),
            "chunks": database.get_knowledge_chunks(doc_id),
            "revision": knowledge._get_rebuild_revision(),
            "status": knowledge._read_rebuild_status(),
            "vector_metadata": deepcopy(vector.metadata),
            "cache_generation": vector.cache_generation,
            "built": knowledge._vector_index_built,
        }

    warm = await client.post("/api/knowledge/search", json={"query": fixture["search_query"], "topK": 3})
    assert warm.status_code == 200
    await generate._retrieve_rag_bundle(fixture["cases"][0]["message"], 3, None)
    proof["initial_state"] = snapshot()
    proof["writers"] = {}
    proof["writer_handles"] = {}
    c = await connection(proof["reader_database"])
    await c.execute(
        "CREATE FUNCTION stage49_reject_dirty() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.key='vector_index_rebuild_status' AND NEW.value='dirty' THEN RAISE EXCEPTION 'stage49 actual dirty-write failure'; END IF; RETURN NEW; END $$"
    )
    await c.execute(
        "CREATE TRIGGER stage49_dirty_failure BEFORE INSERT OR UPDATE ON config FOR EACH ROW EXECUTE FUNCTION stage49_reject_dirty()"
    )
    proof["actual_database_failure_installed"] = bool(
        await c.fetchval("SELECT EXISTS(SELECT 1 FROM pg_trigger WHERE tgname='stage49_dirty_failure')")
    )

    async def writer(label):
        with (output / (label + ".log")).open("xb") as log:
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "evaluation.knowledge_mutation_atomicity_writer",
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
                        "doc_id": doc_id,
                        "payload": fixture["update"],
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

    try:
        await writer("failure")
        proof["after_failure"] = snapshot()
    finally:
        await c.execute("DROP TRIGGER stage49_dirty_failure ON config")
        await c.execute("DROP FUNCTION stage49_reject_dirty()")
        proof["actual_failure_trigger_removed"] = not await c.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_trigger WHERE tgname='stage49_dirty_failure')"
        )
        await c.close()
    if proof["atomic_mutation_available"]:
        await writer("retry")
    proof["after_writer"] = snapshot()
    case = fixture["cases"][0]
    assert case["message"] not in [old["message"] for old in source_proof["cases"]]
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
    proof["generation"].append(
        {
            "id": case["id"],
            "http_status": response.status_code,
            "response": response.json(),
            "cloud_call_range": [start, len(cloud_calls)],
        }
    )
    assert response.status_code == 200
    (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    proof["after_generation"] = snapshot()
    scheduler = get_memory_enrichment_scheduler()
    assert await scheduler.flush_memory(timeout=90)
    completion = get_turn_completion_runtime()
    await completion.shutdown(timeout=35)
    assert await scheduler.flush_memory(timeout=90)
    proof["jobs_terminal"] = scheduler._processing == scheduler._inflight == 0
    proof["completion_runtime_terminal"] = {"active": completion.active, "reserved": completion.reserved}
    proof["queue_stats"] = runtime.stats()
    proof["sync_pending_final"] = len(database._pending)
    proof["final_documents"] = [database.get_knowledge_document(d["id"]) for d in proof["initial_documents"]]
    proof["final_chunks"] = {str(d["id"]): database.get_knowledge_chunks(d["id"]) for d in proof["initial_documents"]}
    proof["parent_source_unchanged"] = before_parent == await parent_snapshot()
    capture_storage_proof(proof, output)
    from evaluation.knowledge_mutation_atomicity_audit import audit_mutation

    proof["checks"] = audit_mutation(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]
