"""Repair only the saved failed citation portion, never regenerate course facts."""

import hashlib
import json
import re
from dataclasses import replace
from html import unescape


async def run_citation_recovery_probe(
    client, args, proof, fixture, cloud_calls, *, database, output, capture_storage_proof, source_proof
):
    import asyncpg

    from inference.answer_citations import citation_marker, finalize_answer_citations, prepare_answer_citations
    from inference.citation_recovery import _body, recover_missing_citations
    from inference.generation_request import GenerationPlan, GenerationResult, RetrievalResult
    from inference.model_manager import OpenAICompatProvider

    known = ["current_two_sources_actually_cited", "current_citation_excerpts"]
    assert len(source_proof["checks"]) == 26 and [k for k, v in source_proof["checks"].items() if not v] == known
    proof["parent_verified_checks"] = 24
    proof["parent_known_unresolved"] = known
    proof["primary_generation_replays"] = 0

    async def snapshot(database_name):
        c = await asyncpg.connect(user="boot", database=database_name, host=str(output.parent / "socket"), port=25433)
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

    parent_name = "stage3_stage52_pages_fixed"
    clone_name = database.execute_sql("SELECT current_database() AS name", {})[0]["name"]
    parent_before = await snapshot(parent_name)
    clone_before = await snapshot(clone_name)
    parent_calls = json.loads((output.parent / "stage52-pages-fixed/cloud-calls.json").read_text())
    original = next(c for c in parent_calls if c["request"].get("max_tokens") == 1024)
    raw = original["response"]["choices"][0]["message"]["content"]
    assert (
        raw == source_proof["generation"][0]["response"]["reply"]
        and not source_proof["generation"][0]["response"]["citations"]
    )
    bundle = source_proof["retrieval_diagnostics"][0]["bundle"]
    documents = tuple(bundle["results"])
    citations = tuple(dict(c, id=c["source_id"]) for c in bundle["citations"])
    from api.knowledge import _vector_chunk_document

    kb_names = {kb["id"]: kb["name"] for kb in database.get_knowledge_bases()}
    actual = {
        d["id"]: d for d in (_vector_chunk_document(row, kb_names) for row in database.iter_chunks_with_document())
    }
    proof["authoritative_source_count"] = len(actual)
    assert len(actual) == 503
    for d in documents:
        assert all(d.get(k) == value for k, value in actual[d["id"]].items())
    assert sorted(_body(d) for d in documents) == sorted(fixture["synthetic_source_bodies"])
    proof["payload_provenance"] = (
        "exact repository synthetic source bodies; saved synthetic answer; no user history or credentials"
    )
    proof["current_source_documents"] = list(documents)
    proof["actual_target_documents"] = [database.get_knowledge_document(i) for i in source_proof["target_ids"]]
    retrieval = prepare_answer_citations(
        RetrievalResult(
            status="ok", evidence="\n\n".join(d["content"] for d in documents), documents=documents, citations=citations
        )
    )
    namespace = source_proof["generation_diagnostics"][0]["citation_namespace"]
    wire = unescape(original["request"]["messages"][-1]["content"])
    evidence = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", wire, re.S).group(1)
    for c in retrieval.citations:
        assert citation_marker(namespace, c["key"]) + " " + c["source_title"] in evidence
    retrieval = replace(retrieval, citation_namespace=namespace, evidence=evidence)
    plan = GenerationPlan(
        messages=tuple(original["request"]["messages"]),
        generation=original["request"],
        prompt_policy_version="saved_actual_parent",
        lora_name=None,
        retrieval=retrieval,
    )
    baseline = finalize_answer_citations(GenerationResult(reply=raw, plan=plan))
    proof["baseline"] = {
        "reply": baseline.reply,
        "citations": list(baseline.response_citations),
        "reply_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "namespace": namespace,
    }
    provider = OpenAICompatProvider()
    proof["annotation_adapter_requests"] = []

    async def annotate(**kwargs):
        proof["annotation_adapter_requests"].append(
            {"messages": kwargs["messages"], "max_tokens": kwargs["max_tokens"], "temperature": kwargs["temperature"]}
        )
        reply, cost = await provider.async_complete(
            kwargs["messages"],
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"],
            top_p=kwargs["top_p"],
        )
        return reply

    recovered = await recover_missing_citations(baseline, annotate, context_window_tokens=65536)
    proof["recovered"] = {
        "reply": recovered.reply,
        "citations": list(recovered.response_citations),
        "attempted": recovered.citation_repair_attempted,
        "status": recovered.citation_repair_status,
        "reply_sha256": hashlib.sha256(recovered.reply.encode()).hexdigest(),
    }
    proof["admitted_source_bodies"] = [
        {
            "key": c["key"],
            "source_id": c["source_id"],
            "body": _body(next(d for d in documents if d["id"] == c["source_id"])),
        }
        for c in retrieval.citations
    ]
    proof["parent_source_unchanged"] = parent_before == await snapshot(parent_name)
    proof["cloned_sources_unchanged"] = clone_before == await snapshot(clone_name)
    proof["sync_pending_final"] = len(database._pending)
    capture_storage_proof(proof, output)
    from evaluation.citation_recovery_audit import audit_recovery

    proof["checks"] = audit_recovery(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values())
