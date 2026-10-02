"""A delayed real writer read, intervening commits and one new pro answer."""

import asyncio
import hashlib
import json
import os
import sys
import time
from copy import deepcopy

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_concurrency_probe(
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

    async def connect(name):
        connection = await asyncpg.connect(user="boot", database=name, host=str(output.parent / "socket"), port=25433)
        assert await connection.fetchval("SHOW data_directory") == str(output.parent / "data")
        return connection

    async def parent_snapshot():
        connection = await connect("stage3_stage46_workers_fixed")
        try:
            result = {}
            for table in ["knowledge_documents", "knowledge_chunks", "config"]:
                rows = [dict(r) for r in await connection.fetch("SELECT * FROM " + table)]
                result[table] = hashlib.sha256(
                    json.dumps(
                        sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str)),
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest()
            return result
        finally:
            await connection.close()

    parent_before = await parent_snapshot()
    assert len(source_proof["checks"]) == 32 and all(source_proof["checks"].values())
    proof["reader_pid"] = os.getpid()
    proof["reader_database_name"] = database.execute_sql("SELECT current_database() AS name", {})[0]["name"]
    proof["parent_verified_checks"] = 32
    proof["document_imports_replayed"] = 0
    proof["atomic_invalidation_available"] = callable(getattr(database, "mark_knowledge_index_dirty", None))
    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    proof["actual_account_role"] = me.json()["user"]["role"]
    assert proof["actual_account_role"] == "admin"
    doc_id = source_proof["actual_target_document_id"]
    proof["actual_target_document_id"] = doc_id
    proof["documents"] = deepcopy(source_proof["documents"])
    proof["initial_documents"] = [
        database.get_knowledge_document(d["response"]["document"]["id"]) for d in proof["documents"]
    ]
    assert database.get_knowledge_document(doc_id) == source_proof["final_document"]
    assert database.get_knowledge_chunks(doc_id) == source_proof["final_chunks"]
    for document, fixture_document in zip(proof["initial_documents"], fixture["documents"]):
        assert document["content"] == fixture_document["content"] and document["title"] == fixture_document["title"]
    proof["initial_chunks"] = {str(d["id"]): database.get_knowledge_chunks(d["id"]) for d in proof["initial_documents"]}
    assert all(
        "".join(c["content"] for c in proof["initial_chunks"][str(d["id"])]) == d["content"]
        for d in proof["initial_documents"]
    )
    proof["new_query_distinct_from_parent"] = fixture["cases"][0]["message"] not in [
        c["message"] for c in source_proof["cases"]
    ]
    assert proof["new_query_distinct_from_parent"]
    vector = get_vector_db()

    def snapshot():
        return dict(
            document=database.get_knowledge_document(doc_id),
            vector_metadata=deepcopy(vector.metadata),
            cache_generation=vector.cache_generation,
            revision=knowledge._get_rebuild_revision(),
            status=knowledge._read_rebuild_status(),
            index_built=knowledge._vector_index_built,
        )

    async def search(label):
        response = await client.post(
            "/api/knowledge/search",
            json=dict(query=fixture["search_query"], topK=3, knowledgeBaseName=fixture["knowledge_bases"][1]),
        )
        assert response.status_code == 200
        proof["searches"].append(dict(label=label, http_status=response.status_code, response=response.json()))

    await search("initial_source_warm")
    proof["initial_state"] = snapshot()
    children = {}
    logs = {}
    proof["writers"] = {}
    proof["writer_handles"] = {}

    async def launch(label, selected):
        logs[label] = (output / f"{label}.log").open("xb")
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "evaluation.knowledge_concurrency_writer",
            stdin=asyncio.subprocess.PIPE,
            stdout=logs[label],
            stderr=asyncio.subprocess.STDOUT,
        )
        children[label] = child
        (output / f"{label}-job.json").write_text(json.dumps(dict(pid=child.pid, label=label)))
        child.stdin.write(
            json.dumps(
                dict(
                    output=str(output),
                    label=label,
                    password=password,
                    doc_id=selected,
                    payload=fixture["metadata_updates"][label],
                )
            ).encode()
        )
        await child.stdin.drain()
        child.stdin.close()
        return child

    async def finish(label):
        child = children[label]
        await child.wait()
        proof["writer_handles"][label] = dict(
            pid=child.pid, returncode=child.returncode, terminal=child.returncode is not None
        )
        assert child.returncode == 0, f"Writer {label} failed; inspect its private log"
        proof["writers"][label] = json.loads((output / f"{label}-result.json").read_text())

    async def wait_gate():
        deadline = time.monotonic() + 18
        while not (output / "B-observed.json").exists():
            assert children["B"].returncode is None, "Writer B ended before its observed read"
            if time.monotonic() > deadline:
                raise TimeoutError("No actual B revision read observed")
            await asyncio.sleep(0.025)
        proof["B_observed"] = json.loads((output / "B-observed.json").read_text())

    try:
        await launch("B", proof["initial_documents"][1]["id"])
        await wait_gate()
        await launch("A", doc_id)
        if proof["atomic_invalidation_available"]:
            connection = await connect(proof["reader_database_name"])
            try:
                deadline = time.monotonic() + 12
                while True:
                    locks = await connection.fetch(
                        "SELECT pid,wait_event_type,wait_event,pg_blocking_pids(pid) AS blocking_pids FROM pg_stat_activity WHERE datname=$1 AND wait_event_type='Lock' AND query ILIKE '%config%'",
                        proof["reader_database_name"],
                    )
                    if locks:
                        proof["actual_database_lock_wait"] = [dict(row) for row in locks]
                        break
                    assert children["A"].returncode is None, (
                        "Writer A unexpectedly ended while B held revision authority"
                    )
                    if time.monotonic() > deadline:
                        raise TimeoutError("No actual atomic writer lock wait")
                    await asyncio.sleep(0.025)
            finally:
                await connection.close()
            (output / "B-release").write_text("release after actual database lock wait")
            await finish("B")
            await finish("A")
        else:
            await finish("A")
            proof["actual_database_lock_wait"] = []
        await search("after_A_before_C")
        await generate._retrieve_rag_bundle(fixture["cases"][0]["message"], 3, None)
        proof["reader_warmed_after_A"] = snapshot()
        await launch("C", doc_id)
        await finish("C")
        proof["after_C"] = snapshot()
        if not proof["atomic_invalidation_available"]:
            (output / "B-release").write_text("release after actual later C commit")
            await finish("B")
        proof["after_all_writers"] = snapshot()
    finally:
        if "B" in children and not (output / "B-release").exists():
            (output / "B-release").write_text("release for cleanup")
        for child in children.values():
            await child.wait()
        for log in logs.values():
            log.close()
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
    await search("after_generation_current_scope")
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
    proof["final_documents"] = [database.get_knowledge_document(d["id"]) for d in proof["initial_documents"]]
    proof["final_chunks"] = {str(d["id"]): database.get_knowledge_chunks(d["id"]) for d in proof["initial_documents"]}
    proof["parent_source_unchanged"] = parent_before == await parent_snapshot()
    capture_storage_proof(proof, output)
    from evaluation.knowledge_concurrency_audit import audit_concurrency

    proof["checks"] = audit_concurrency(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps(dict(checks=proof["checks"], cloud_calls=len(cloud_calls)), ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]
