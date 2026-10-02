"""Two new citation questions against unchanged, verified parent documents."""

import hashlib
import json

from character.memory_llm import get_memory_enrichment_scheduler
from services.turn_completion import get_turn_completion_runtime


async def run_citation_policy_probe(
    client, args, proof, fixture, cloud_calls, *, database, runtime, output, capture_storage_proof, source_proof
):
    import asyncpg

    from api import knowledge

    async def parent_snapshot():
        connection = await asyncpg.connect(
            user="boot", database="stage3_stage47_revision_fixed", host=str(output.parent / "socket"), port=25433
        )
        try:
            assert await connection.fetchval("SHOW data_directory") == str(output.parent / "data")
            result = {}
            for table in ["knowledge_documents", "knowledge_chunks", "config"]:
                rows = [dict(row) for row in await connection.fetch("SELECT * FROM " + table)]
                result[table] = hashlib.sha256(
                    json.dumps(
                        sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str)),
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest()
            return result
        finally:
            await connection.close()

    parent_before = await parent_snapshot()
    assert [key for key, value in source_proof["checks"].items() if not value] == ["current_citation_title"]
    proof["parent_checks_passed"] = sum(source_proof["checks"].values())
    proof["parent_checks_total"] = len(source_proof["checks"])
    proof["parent_known_unresolved"] = ["current_citation_title"]
    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    proof["actual_account_role"] = me.json()["user"]["role"]
    assert proof["actual_account_role"] == "admin"
    proof["initial_documents"] = source_proof["final_documents"]
    for d, f in zip(proof["initial_documents"], fixture["documents"]):
        assert (
            database.get_knowledge_document(d["id"]) == d and d["title"] == f["title"] and d["content"] == f["content"]
        )
        assert database.get_knowledge_chunks(d["id"]) == source_proof["final_chunks"][str(d["id"])]
    proof["initial_chunks"] = source_proof["final_chunks"]
    proof["document_imports_replayed"] = 0
    proof["document_mutations"] = 0
    proof["initial_revision"] = knowledge._get_rebuild_revision()
    proof["new_queries_distinct_from_parent"] = all(
        c["message"] not in [old["message"] for old in source_proof["cases"]] for c in fixture["cases"]
    )
    assert proof["new_queries_distinct_from_parent"]
    for case in fixture["cases"]:
        before = len(cloud_calls)
        response = await client.post(
            "/api/generate",
            json=dict(
                message=case["message"],
                characterId="tsukiyashiro_kisaki",
                loraId="default",
                sessionId=args.run_label,
                sessionType="private",
            ),
        )
        proof["generation"].append(
            dict(
                id=case["id"],
                http_status=response.status_code,
                response=response.json(),
                cloud_call_range=[before, len(cloud_calls)],
            )
        )
        (output / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
        assert response.status_code == 200
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
    proof["final_revision"] = knowledge._get_rebuild_revision()
    proof["parent_source_unchanged"] = parent_before == await parent_snapshot()
    capture_storage_proof(proof, output)
    from evaluation.citation_policy_consistency_audit import audit_citation_policy

    proof["checks"] = audit_citation_policy(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps(dict(checks=proof["checks"], cloud_calls=len(cloud_calls)), ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values()), proof["checks"]
