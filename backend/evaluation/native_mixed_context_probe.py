"""Complete private constraints with long native history and knowledge retrieval."""

import argparse
import asyncio
import base64
import json
import os
import re
import secrets
from dataclasses import asdict
from html import unescape
from pathlib import Path


def audit_mixed_wire(proof, calls, fixture):
    source = fixture["cases"][0]["message"]
    last = proof["generation"][-1]
    answer = [c for c in calls[slice(*last["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    if len(answer) != 1:
        return {"one_actual_mixed_answer": False}
    messages = answer[0]["request"]["messages"]
    wire = unescape(messages[-1]["content"])
    all_wire = "\n".join(unescape(m["content"]) for m in messages)
    memory = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
    knowledge = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", wire, re.S)
    packets = [json.loads(line[2:]) for line in memory[1].splitlines() if line.startswith("- {")] if memory else []
    related = [p for p in packets if source in p.get("evidence", [])]
    reply = last["response"].get("reply", "")
    doc = fixture["documents"][0]
    diag = proof["prepared_diagnostics"][-1]
    return dict(
        one_actual_mixed_answer=True,
        complete_private_source_persisted=any(json.loads(row.get("evidence_json") or "[]") == [source] for row in proof["claims"]),
        complete_private_source_in_selected_memory=bool(related),
        complete_private_source_in_actual_history=any(m.get("role") == "user" and m.get("content") == source for m in messages),
        complete_knowledge_source_in_final_evidence=bool(knowledge) and doc["content"] in knowledge[1],
        private_source_outside_knowledge=bool(knowledge) and source not in knowledge[1],
        knowledge_source_outside_private_memory=bool(memory) and doc["content"] not in memory[1],
        materials_not_system_rules=all(source not in m["content"] and doc["content"] not in m["content"] for m in messages if m["role"] == "system"),
        all_twelve_complete_history_sources_in_actual_wire=all(turn["message"] in all_wire for turn in fixture["history_turns"]),
        selection_succeeded=diag["selection_status"] == "selected",
        semantic_review_succeeded=diag["semantic_status"] == "applied",
        policy_succeeded=diag["policy_status"] == "applied",
        current_rain_and_missing_confirmation_explained=bool(re.search(r"雨", reply)) and bool(re.search(r"(?:没|未|不).{0,12}(?:书面|确认)|(?:书面|确认).{0,12}(?:没|未|不)", reply)),
        personal_trip_not_authorized=bool(re.search(r"不.{0,8}(?:出门|出行|参加)|取消.{0,8}(?:出行|计划)|(?:不能|无法|不满足).{0,12}(?:条件|出门|出行)", reply)),
        public_class_still_runs=bool(re.search(r"(?:课程|工坊|开课|纸雕课).{0,30}(?:照常|不停|不取消|正常)|(?:照常|正常).{0,12}(?:开课|举办|上课)", reply)),
        no_private_attendance_invented=bool(re.search(r"(?:尚未|没有|还没|未曾).{0,10}(?:预约|报名|参加|上课)", reply)),
        course_citation_provenance=any(c.get("source_id") == "doc_1_chunk_0" and c.get("source_title") == doc["title"] for c in last["response"].get("citations", [])),
    )


def audit_window_boundary_wire(proof, calls, fixture):
    """At the serving boundary, full memory must survive whole-turn pruning."""
    from inference.context_budget import ReviewContextBudget

    checks = audit_mixed_wire(proof, calls, fixture)
    # This case deliberately reaches the budget boundary: older history may
    # yield, but the full authorized private source must remain in memory.
    checks.pop("complete_private_source_in_actual_history", None)
    checks.pop("all_twelve_complete_history_sources_in_actual_wire", None)
    source = fixture["cases"][0]["message"]
    final = [c for c in calls[slice(*proof["generation"][-1]["cloud_call_range"])]
             if c["request"].get("max_tokens") == 1024]
    checks["actual_answer_fits_serving_budget"] = len(final) == 1 and ReviewContextBudget(65536).fits(final[0]["request"]["messages"], 1024)
    checks["history_really_trimmed_at_boundary"] = len(final) == 1 and not any(m.get("content") == source for m in final[0]["request"]["messages"])
    reply = proof["generation"][-1]["response"].get("reply", "")
    checks["private_no_rain_condition_recovered"] = bool(re.search(r"不下雨|无雨|没有下雨|未下雨", reply))
    checks["historical_no_appointment_recovered"] = bool(re.search(r"(?:尚未|没有|还没|未曾|未).{0,10}(?:提交预约|预约|报名)", reply))
    checks["historical_no_attendance_recovered"] = bool(re.search(r"(?:尚未|没有|还没|未曾|未).{0,10}(?:参加|上课)", reply))
    return checks


def audit_citation_precision_wire(proof, calls, fixture):
    """Every returned source must be bound from a real answer marker."""
    checks = {}
    for case, generation in zip(fixture["cases"], proof["generation"]):
        citations = generation["response"].get("citations") or []
        actual = [c.get("source_id") for c in citations]
        checks[case["id"] + "_only_answer_used_sources"] = set(actual) == set(case["expected_citation_ids"]) and len(actual) == len(set(actual))
        answer = [c for c in calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        raw = answer[-1]["response"]["choices"][0]["message"]["content"] if answer else ""
        keys = re.findall(r"\[S(\d{1,2})\]", raw)
        checks[case["id"] + "_returned_keys_from_actual_model"] = bool(citations) and all(c.get("key", "")[1:] in keys for c in citations)
        wire = unescape(answer[-1]["request"]["messages"][-1]["content"]) if answer else ""
        checks[case["id"] + "_all_complete_sources_still_offered"] = all(d["content"] in wire for d in fixture["documents"])
    return checks


def expand_history(fixture):
    """Expand fully specified synthetic history without storing repeated text."""
    import hashlib

    template = fixture["history_template"]
    turns = []
    for turn in range(1, template["turn_count"] + 1):
        message = template["heading"].format(turn=turn) + "\n" + "\n".join(
            template["line"].format(turn=turn, record=record)
            for record in range(template["records_per_turn"]))
        turns.append(dict(message=message, reply=template["reply"].format(turn=turn)))
    encoded = json.dumps(turns, ensure_ascii=False, sort_keys=True).encode()
    if hashlib.sha256(encoded).hexdigest() != template["expanded_sha256"]:
        raise ValueError("Synthetic history differs from the complete source manifest")
    return turns


def capture_storage_proof(proof, output):
    """Snapshot adapter-owned facts while application resources remain open."""
    from api.knowledge import _get_expected_chunk_count
    from knowledge.vector_db import VectorDatabase

    proof["persisted_vector_stats"] = VectorDatabase(db_path=str(output / "vectors")).get_stats()
    proof["persisted_valid_chunk_count"] = _get_expected_chunk_count()
    (output / "checkpoint.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))


def isolated_probe_paths(root, run_label, api_key_file):
    root = Path(root).resolve()
    allowed = Path("/home/boot/lhm/multipersonal-runtime/evaluations").resolve()
    if root.parent != allowed or not re.fullmatch(r"r148pg\.[a-zA-Z0-9_.-]+", root.name):
        raise ValueError("This probe only accepts a stage-three disposable cluster")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", run_label):
        raise ValueError("Invalid isolated run label")
    key_path = Path(api_key_file).resolve()
    if key_path.parent != Path("/home/boot/lhm/multipersonal-runtime/config").resolve():
        raise ValueError("API key must remain in the private runtime config directory")
    if not (root / "data").is_dir() or not (root / "socket").is_dir():
        raise ValueError("Create the disposable cluster before running this probe")
    return root, root / run_label, key_path


async def main(args):
    from evaluation.conversation_source_probe import verify_cluster

    if args.window_boundary and args.citation_precision:
        raise ValueError("Choose one specific probe scenario")

    ROOT, OUT, key_path = isolated_probe_paths(args.root, args.run_label, args.api_key_file)
    bootstrap_url = "postgresql+asyncpg://boot@/postgres?host=" + str(ROOT / "socket") + "&port=25433"
    await verify_cluster(bootstrap_url, ROOT / "data")
    database_name = "stage3_" + args.run_label.replace("-", "_")
    OUT.mkdir(exist_ok=False)
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures" / ("deepseek_citation_precision_cases.json" if args.citation_precision else "deepseek_mixed_window_boundary.json" if args.window_boundary else "deepseek_mixed_long_context_cases.json")).read_text()
    )
    fixture["history_turns"] = expand_history(fixture)
    cases = fixture["cases"]
    key = key_path.read_text().strip()
    os.environ.update(
        DATABASE_URL="postgresql+asyncpg://boot@/" + database_name + "?host=" + str(ROOT / "socket") + "&port=25433",
        USE_POSTGRESQL="true",
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        ASTRBOT_INTEGRATION_TOKEN=secrets.token_urlsafe(48),
        MULTIPERSONAL_BACKEND_URL="https://stage3-evaluation.invalid",
        ALLOWED_ORIGINS="https://stage3-evaluation.invalid",
        ALLOW_PUBLIC_REGISTRATION="false",
        SECURITY_MIDDLEWARE_ENABLED="true",
        LOG_LEVEL="INFO",
        BACKEND_WORKERS="1",
        AUDIT_LOG_DIR=str(OUT / "audit"),
        BACKUP_DIR=str(OUT / "backups"),
        MODEL_PROVIDER="openai_compat",
        VLLM_MAX_MODEL_LEN="8192",
        OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS="65536",
        VLLM_ENABLED="false",
        VLLM_BASE_URL="http://127.0.0.1:1",
        VLLM_BASE_URLS="http://127.0.0.1:1",
        OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
        OPENAI_COMPAT_API_KEY=key,
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        MEMORY_LLM_ENABLED="true",
        MEMORY_LLM_BASE_URL="https://api.deepseek.com",
        MEMORY_LLM_MODEL="deepseek-v4-pro",
        MEMORY_LLM_API_KEY=key,
        MEMORY_LLM_CONTEXT_WINDOW_TOKENS="65536",
        MEMORY_SOURCE_RECALL_ENABLED="false",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED="true",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS="30",
        CONTEXTUAL_MEMORY_SELECTION_ENABLED="true",
        CONTEXTUAL_DECISION_POLICY_ENABLED="true",
        REDIS_URL="redis://127.0.0.1:1/0",
        EMBEDDING_MODEL_PATH="/home/boot/lhm/multipersonal-runtime/models/paraphrase-multilingual-MiniLM-L12-v2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        VECTOR_DB_PATH=str(OUT / "vectors"),
        INTENT_MODEL_PATH=str(OUT / "no-intent-model"),
        CHARACTER_RAG_INDEX_ROOT=str(OUT / "no-character-index"),
        CORRECTIVE_RAG_ENABLED="false",
        RERANKER_ENABLED="false",
        CHAT_CONVERSATION_BURST="10",
        CHAT_SENDER_BURST="10",
    )

    await verify_cluster(bootstrap_url, ROOT / "data")
    import asyncpg

    connection = await asyncpg.connect(user="boot", database="postgres", host=str(ROOT / "socket"), port=25433)
    try:
        assert await connection.fetchval("SHOW data_directory") == str(ROOT / "data")
        await connection.execute("CREATE DATABASE " + database_name)
    finally:
        await connection.close()
    await verify_cluster(os.environ["DATABASE_URL"], ROOT / "data")
    import httpx

    cloud_calls = []
    local_review_calls = []
    original_send = httpx.AsyncClient.send

    async def observed_send(client, request, **kwargs):
        if request.url.host != "api.deepseek.com":
            if request.url.host == "127.0.0.1" and request.url.path.endswith("/chat/completions"):
                local_body = json.loads(request.content)
                local_review_calls.append(
                    dict(url=str(request.url), model=local_body.get("model"), max_tokens=local_body.get("max_tokens"))
                )
                (OUT / "local-review-calls.json").write_text(json.dumps(local_review_calls, indent=2))
            return await original_send(client, request, **kwargs)
        body = json.loads(request.content)
        response = await original_send(client, request, **kwargs)
        await response.aread()
        data = response.json()
        cloud_calls.append(dict(url=str(request.url), request=body, http_status=response.status_code, response=data))
        (OUT / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
        return response

    httpx.AsyncClient.send = observed_send
    from app.main import create_app
    from character.memory_llm import get_memory_enrichment_scheduler, shutdown_memory_enrichment
    from db.adapter import db, is_pg_mode
    from infra.concurrency_control import inference_runtime
    from services.character_context import CharacterContextService

    prepared_diagnostics = []
    generation_diagnostics = []
    original_prepare = CharacterContextService.prepare_turn
    from api import generate as generation_api

    retrieval_diagnostics = []
    original_retrieval = generation_api._retrieve_rag_bundle

    async def observed_retrieval(query, top_k, filters):
        bundle = await original_retrieval(query, top_k, filters)
        retrieval_diagnostics.append(dict(query=query, filters=filters, bundle=bundle))
        return bundle

    generation_api._retrieve_rag_bundle = observed_retrieval
    original_generation = generation_api.generate_character_response

    async def observed_prepare(service, turn, character_id):
        prepared = await original_prepare(service, turn, character_id)
        prepared_diagnostics.append(
            dict(
                query=turn.message,
                history=list(prepared.history),
                selection_status=prepared.memory_selection_status,
                selection_reason=prepared.memory_selection_reason,
                semantic_status=prepared.semantic_review_status,
                semantic_reason=prepared.semantic_review_fallback_reason,
                policy_status=prepared.contextual_policy_status,
                policy_reason=prepared.contextual_policy_reason,
                used_memory_ids=list(prepared.compiled.used_memory_ids),
            )
        )
        return prepared

    async def observed_generation(request, generate):
        row = dict(
            context_window_tokens=request.context_window_tokens,
            input_chars=len(request.message),
            history=list(request.history),
        )
        generation_diagnostics.append(row)
        try:
            result = await original_generation(request, generate)
            row["model_messages"] = list(result.plan.messages)
            return result
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
            raise

    CharacterContextService.prepare_turn = observed_prepare
    generation_api.generate_character_response = observed_generation
    assert is_pg_mode()
    db.update_config(dict(useKnowledgeBase=True, temperature=0.2, maxTokens=1024, topP=0.9))
    app = create_app()
    proof = dict(
        cases=cases,
        generation=[],
        prepared_diagnostics=prepared_diagnostics,
        generation_diagnostics=generation_diagnostics,
        provider="native_openai_compat",
        transport="authenticated_ASGI",
        source_ablation=False,
        rag_enabled=True,
        documents=[],
        searches=[],
        retrieval_diagnostics=retrieval_diagnostics,
    )
    password = secrets.token_urlsafe(24)
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="https://stage3-evaluation.invalid",
                timeout=180,
            ) as client,
        ):
            register = await client.post("/api/auth/register", json=dict(username=args.run_label, password=password))
            client.cookies.clear()
            login = await client.post("/api/auth/login", json=dict(username=args.run_label, password=password))
            me = await client.get("/api/auth/me")
            proof["auth_statuses"] = [register.status_code, login.status_code, me.status_code]
            assert proof["auth_statuses"] == [200, 200, 200]
            identity = str(me.json()["user"]["id"])
            from knowledge.vector_db import get_vector_db

            base = await client.post(
                "/api/knowledge/bases",
                json=dict(name=fixture["knowledge_base"], description="Complete synthetic chain fixtures"),
            )
            assert base.status_code == 200, base.text
            proof["base"] = base.json()
            base_id = base.json()["base"]["id"]
            for document in fixture["documents"]:
                response = await client.post(
                    "/api/knowledge/documents", json=dict(**document, knowledge_base_id=base_id)
                )
                proof["documents"].append(dict(http_status=response.status_code, response=response.json()))
                assert response.status_code == 200, response.text
            vector_db = get_vector_db()
            proof["indexed_count"] = len(vector_db.metadata)
            for query in fixture["search_queries"]:
                response = await client.post(
                    "/api/knowledge/search", json=dict(query=query, topK=3, knowledgeBaseName=fixture["knowledge_base"])
                )
                proof["searches"].append(dict(query=query, http_status=response.status_code, response=response.json()))
            for case in cases:
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
                assert await get_memory_enrichment_scheduler().flush_memory(timeout=90)
                proof["generation"].append(
                    dict(
                        id=case["id"],
                        http_status=response.status_code,
                        response=response.json(),
                        cloud_call_range=[before, len(cloud_calls)],
                    )
                )
                if case["id"] == "personal_schedule":
                    for index, turn in enumerate(fixture["history_turns"]):
                        await asyncio.to_thread(db.add_message, dict(
                            platform="web", adapter="web-character", senderId=identity, userId=identity,
                            conversationType="private", sessionType="private", conversationId=identity,
                            sessionId=args.run_label, characterId="tsukiyashiro_kisaki", loraName="default",
                            sourceMessageId=args.run_label + "-history-" + str(index),
                            message=turn["message"], reply=turn["reply"], modelName="synthetic-fixture-history"))
                (OUT / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
            proof["claims"] = db.list_character_memory_claims(
                "tsukiyashiro_kisaki",
                "web",
                "web-character",
                identity,
                "private",
                identity,
                limit=None,
                include_inactive=True,
            )
            proof["queue_stats"] = inference_runtime.stats()
            proof["memory_status"] = asdict(get_memory_enrichment_scheduler().status)
            capture_storage_proof(proof, OUT)
        checks = dict(
            all_generations_success=all(g["http_status"] == 200 for g in proof["generation"]),
            all_cloud_calls_completed=bool(cloud_calls)
            and all(
                c["http_status"] == 200
                and c["response"].get("model") == "deepseek-v4-pro"
                and all(x.get("finish_reason") == "stop" for x in c["response"].get("choices", []))
                for c in cloud_calls
            ),
            native_cloud_window_used=bool(generation_diagnostics)
            and all(g["context_window_tokens"] == 65536 for g in generation_diagnostics),
            documents_really_indexed=proof["indexed_count"] >= len(fixture["documents"]),
            native_searches_success=all(
                s["http_status"] == 200
                and s["response"].get("retrievalMode") == "evidence"
                and s["response"].get("results")
                for s in proof["searches"]
            ),
            no_wrong_local_completions=not local_review_calls,
            memory_writer_no_errors=proof["memory_status"]["failed"] == 0,
            long_history_loaded=len(prepared_diagnostics[-1]["history"]) >= 26
            and sum(len(h["content"]) for h in prepared_diagnostics[-1]["history"]) >= 18000,
        )
        if args.citation_precision:
            checks.pop("long_history_loaded")
            checks["no_long_history_replay"] = all(len(d["history"]) < 10 for d in prepared_diagnostics)
        for case, generation in zip(cases, proof["generation"]):
            if not case.get("knowledge_expected"):
                continue
            reply = generation["response"].get("reply", "")
            checks[case["id"] + "_answer_fields"] = all(v in reply for v in case["expected"])
            checks[case["id"] + "_citations"] = bool(generation["response"].get("citations"))
            selected = cloud_calls[slice(*generation["cloud_call_range"])]
            wires = [
                unescape(c["request"]["messages"][-1]["content"])
                for c in selected
                if c["request"].get("max_tokens") == 1024
            ]
            # These source fields do not occur in the user question. Check the
            # actual last-turn compiled evidence, not earlier answers/history.
            checks[case["id"] + "_knowledge_on_wire"] = bool(wires) and any(
                all(v in wire for v in case["knowledge_expected"]) for wire in wires
            )
        checks.update(audit_citation_precision_wire(proof, cloud_calls, fixture) if args.citation_precision
                      else audit_window_boundary_wire(proof, cloud_calls, fixture) if args.window_boundary
                      else audit_mixed_wire(proof, cloud_calls, fixture))
        reopened = proof["persisted_vector_stats"]
        checks.update(
            persisted_index_reopens_with_all_three_documents=reopened["total_documents"]
            == reopened["index_size"]
            == reopened["bm25_corpus_size"]
            == 3,
            real_postgresql_valid_chunk_count_matches=proof["persisted_valid_chunk_count"] == 3,
        )
        proof["persisted_vector_stats"] = reopened
        proof["checks"] = checks
        proof["controlled_ablation"] = dict(
            history_ablation_enabled=False,
            raw_source_recall_enabled=False,
            loaded_history_count=len(prepared_diagnostics[-1]["history"]),
        )
        proof["cloud_summary"] = [
            dict(
                http_status=c["http_status"],
                model=c["response"].get("model"),
                finish_reasons=[x.get("finish_reason") for x in c["response"].get("choices", [])],
                output_budget=c["request"].get("max_tokens"),
            )
            for c in cloud_calls
        ]
        (OUT / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                dict(result=str(OUT / "result.json"), checks=proof["checks"], cloud_calls=len(cloud_calls)),
                ensure_ascii=False,
            )
        )
        if args.require_success:
            assert all(proof["checks"].values()), proof["checks"]
    finally:
        await shutdown_memory_enrichment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--window-boundary", action="store_true")
    parser.add_argument("--citation-precision", action="store_true")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    asyncio.run(main(parser.parse_args()))
