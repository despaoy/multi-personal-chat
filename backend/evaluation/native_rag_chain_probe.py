"""Native knowledge import, embedding, retrieval, memory and DeepSeek probe."""

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


def audit_rag_wire(proof, calls, fixture):
    """Audit source ownership against real recorded provider messages."""
    source = proof["cases"][1]["message"]
    last = proof["generation"][-1]
    selected = calls[slice(*last["cloud_call_range"])]
    answer = [c for c in selected if c["request"].get("max_tokens") == 1024]
    if len(answer) != 1:
        raise ValueError("Expected exactly one native answer for the mixed request")
    messages = answer[0]["request"]["messages"]
    wire = unescape(messages[-1]["content"])
    memory = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
    knowledge = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", wire, re.S)
    packets = [json.loads(line[2:]) for line in memory[1].splitlines() if line.startswith("- {")] if memory else []
    plan = [p for p in packets if p.get("evidence") == [source]]
    course = fixture["documents"][0]
    return dict(
        cold_wire_has_no_history=len(messages) == 2
        and messages[0]["role"] == "system"
        and messages[-1]["role"] == "user",
        cold_wire_contains_original_personal_plan=bool(plan),
        personal_plan_packet_keeps_source_observation=bool(plan)
        and all(
            p.get("content_semantics") == "quoted_source" and p.get("temporal_mode") == "observation" for p in plan
        ),
        complete_course_source_is_in_retrieved_evidence=bool(knowledge) and course["content"] in knowledge[1],
        personal_source_is_outside_knowledge_evidence=bool(knowledge) and source not in knowledge[1],
        knowledge_source_is_outside_personal_memory=bool(memory) and course["content"] not in memory[1],
        source_materials_never_become_system_rules=all(
            source not in m["content"] and course["content"] not in m["content"]
            for m in messages
            if m["role"] == "system"
        ),
        cold_course_citation_has_correct_provenance=any(
            c.get("source_id") == "doc_1_chunk_0" and c.get("source_title") == course["title"]
            for c in last["response"].get("citations", [])
        ),
    )


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

    ROOT, OUT, key_path = isolated_probe_paths(args.root, args.run_label, args.api_key_file)
    bootstrap_url = "postgresql+asyncpg://boot@/postgres?host=" + str(ROOT / "socket") + "&port=25433"
    await verify_cluster(bootstrap_url, ROOT / "data")
    database_name = "stage3_" + args.run_label.replace("-", "_")
    OUT.mkdir(exist_ok=False)
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures/deepseek_native_rag_cases.json").read_text()
    )
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
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS="5",
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
    original_history = CharacterContextService._load_history
    cold_state = {"active": False, "underlying_history_count": None}

    async def controlled_history(service, turn, scope, character_id):
        actual = await original_history(service, turn, scope, character_id)
        if cold_state["active"]:
            cold_state["underlying_history_count"] = len(actual)
            return []
        return actual

    CharacterContextService._load_history = controlled_history

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
                cold_state["active"] = case["id"] == "mixed_cold_memory_knowledge"
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
            mixed_read_has_no_history=not prepared_diagnostics[-1]["history"]
            and (cold_state["underlying_history_count"] or 0) > 0,
        )
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
        checks.update(audit_rag_wire(proof, cloud_calls, fixture))
        from api.knowledge import _get_expected_chunk_count
        from knowledge.vector_db import VectorDatabase

        reopened = VectorDatabase(db_path=str(OUT / "vectors")).get_stats()
        checks.update(
            persisted_index_reopens_with_all_three_documents=reopened["total_documents"]
            == reopened["index_size"]
            == reopened["bm25_corpus_size"]
            == 3,
            real_postgresql_valid_chunk_count_matches=_get_expected_chunk_count() == 3,
        )
        proof["persisted_vector_stats"] = reopened
        proof["checks"] = checks
        proof["controlled_ablation"] = dict(
            history_only_on_last_mixed_read=True,
            raw_source_recall_enabled=False,
            underlying_history_count=cold_state["underlying_history_count"],
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
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    asyncio.run(main(parser.parse_args()))
