"""Real keyword fallback crossing its unchanged 500-row page boundary."""

import asyncio
import hashlib
import json
import os
import stat
import sys
import time
from copy import deepcopy

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_read_snapshot_probe(
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

    from api import knowledge
    from knowledge.vector_db import get_vector_db

    async def parent_snapshot():
        c = await asyncpg.connect(
            user="boot", database="stage3_stage51_snapshot_fixed", host=str(output.parent / "socket"), port=25433
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
    assert len(source_proof["checks"]) == 23 and all(source_proof["checks"].values())
    proof["parent_verified_checks"] = 23
    docs = deepcopy(source_proof["final_documents"])
    proof["initial_parent_documents"] = docs
    assert [database.get_knowledge_document(d["id"]) for d in docs] == docs
    kb_id = docs[0]["knowledge_base_id"]

    def seed():
        a = fixture["initial_arrangement"]
        assert database.save_knowledge_document(a, doc_id=docs[0]["id"], chunks=[a["content"]])
        distractors = fixture["distractors"]
        noise = []
        for index in range(distractors["count"]):
            record = {
                "title": distractors["title_template"].format(index=index),
                "content": distractors["content_template"].format(index=index),
                "category": "Inventory",
                "knowledge_base_id": kb_id,
                "sourceType": "text",
                "fileType": "txt",
            }
            d = database.save_knowledge_document(record, chunks=[record["content"]])
            noise.append(d)
        record = {
            **fixture["initial_confirmation"],
            "category": "未分类",
            "knowledge_base_id": kb_id,
            "sourceType": "text",
            "fileType": "txt",
        }
        b = database.save_knowledge_document(record, chunks=[record["content"]])
        return b, noise

    bdoc, noise = await asyncio.to_thread(seed)
    ids = [docs[0]["id"], bdoc["id"]]
    proof["target_ids"] = ids
    proof["initial_paired_documents"] = [database.get_knowledge_document(i) for i in ids]
    proof["noise_count"] = len(noise)
    proof["noise_digest"] = hashlib.sha256(json.dumps(noise, sort_keys=True, default=str).encode()).hexdigest()
    proof["expected_count"] = knowledge._get_expected_chunk_count()
    assert proof["expected_count"] == 503
    vector = get_vector_db()
    folder = vector.db_path
    mode = stat.S_IMODE(folder.stat().st_mode)
    old_hash = hashlib.sha256(vector.snapshot_path.read_bytes()).hexdigest()
    original_iterator = database.iter_chunks_with_document
    gate = output / "reader-after-500.json"
    release = output / "reader-release"
    observed = []

    def iterator(batch_size=500):
        assert batch_size == 500
        for count, row in enumerate(original_iterator(batch_size=batch_size), 1):
            observed.append(deepcopy(row))
            yield row
            if count == 500:
                gate.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "actual_rows_yielded": count,
                            "actual_first_document": observed[0],
                            "batch_size": batch_size,
                        },
                        ensure_ascii=False,
                    )
                )
                deadline = time.monotonic() + 240
                while not release.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Poll the existing live writer, do not restart")
                    time.sleep(0.025)

    database.iter_chunks_with_document = iterator
    folder.chmod(0o500)  # Real mkdtemp/write failure; reads of the previous snapshot still work.
    try:
        task = asyncio.create_task(
            client.post("/api/knowledge/search", json={"query": fixture["search_query"], "topK": 3})
        )
        deadline = time.monotonic() + 55
        while not gate.exists():
            if task.done():
                raise RuntimeError("Keyword page gate not reached; inspect existing request result")
            if time.monotonic() > deadline:
                raise TimeoutError("Inspect same search handle")
            await asyncio.sleep(0.025)
        with (output / "writer.log").open("xb") as log:
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "evaluation.knowledge_read_snapshot_writer",
                stdin=asyncio.subprocess.PIPE,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            (output / "writer-job.json").write_text(json.dumps({"pid": child.pid, "label": "writer"}))
            child.stdin.write(
                json.dumps(
                    {
                        "output": str(output),
                        "password": password,
                        "ids": ids,
                        "updates": [fixture["final_arrangement"], fixture["final_confirmation"]],
                    }
                ).encode()
            )
            await child.stdin.drain()
            child.stdin.close()
            await child.wait()
        assert child.returncode == 0, "Inspect private writer log"
        proof["writer"] = json.loads((output / "writer-result.json").read_text())
        proof["writer_handle"] = {
            "pid": child.pid,
            "returncode": child.returncode,
            "terminal": child.returncode is not None,
        }
        release.write_text("actual writer completed both commits")
        response = await task
        assert response.status_code == 200
        proof["fallback_response"] = {"http_status": response.status_code, "response": response.json()}
        proof["reader_rows"] = observed
        proof["actual_gate"] = json.loads(gate.read_text())
        proof["physical_write_failure_kept_snapshot"] = (
            hashlib.sha256(vector.snapshot_path.read_bytes()).hexdigest() == old_hash
        )
    finally:
        folder.chmod(mode)
        database.iter_chunks_with_document = original_iterator
    proof["reader_pid"] = os.getpid()
    proof["reader_database"] = database.execute_sql("SELECT current_database() AS name", {})[0]["name"]
    proof["final_paired_documents"] = [database.get_knowledge_document(i) for i in ids]
    proof["other_parent_documents_preserved"] = [database.get_knowledge_document(d["id"]) for d in docs[1:]] == docs[1:]
    proof["parent_source_unchanged"] = before == await parent_snapshot()
    proof["new_questions_replayed"] = 0
    # The failed baseline is fully observable without paying for a model answer on missing/mixed evidence.
    if not args.run_label.endswith("-base"):
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
    capture_storage_proof(proof, output)
    from evaluation.knowledge_read_snapshot_audit import audit_read_snapshot

    proof["checks"] = audit_read_snapshot(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values())
