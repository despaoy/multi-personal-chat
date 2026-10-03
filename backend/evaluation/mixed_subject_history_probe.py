"""Observe a real conditional-preference writer followed by an authenticated historical lookup."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path


async def run(args):
    phase = Path(args.phase).resolve()
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    assert phase.parent == runtime / "backups/backend-chain-20261001" and re.fullmatch(r"stage[1-9]\d*", phase.name)
    assert re.fullmatch(r"native-pg(?:-[a-z]{1,12})*", args.variant)
    if args.resume_provider_block and not args.provider_access_restored:
        raise ValueError("Provider access must be restored before any resume artifact is created")
    root = phase / args.variant
    root.mkdir(mode=0o700, exist_ok=False)
    cluster = runtime / "evaluations/r148pg.s3"
    label = "stage3_" + phase.name + "_" + args.variant.replace("-", "_")
    source = "stage3_stage57_budget_fixed"
    assert re.fullmatch(r"native-pg(?:-[a-z]{1,12})*", args.seed_variant)
    seed_root = phase / args.seed_variant
    resume_provider_block = args.resume_provider_block
    if resume_provider_block:
        assert args.provider_access_restored, "Provider returned402; do not start until access is actually restored"
        assert not args.reuse_verified_seed and args.seed_variant == "native-pg-baseline"
        args.reuse_verified_seed = True
    seed_proof = None
    if args.reuse_verified_seed:
        source = "template0"
        if resume_provider_block:
            seed_proof = json.loads((phase / "partial-resume.json").read_text())
            diagnostic = json.loads((phase / "offline-pg-diagnostic.json").read_text())
            assert seed_proof["baseline_job_terminal"] and seed_proof["remaining_bridge_indices"] == [7, 8]
            assert not seed_proof["final_question_attempted"] and diagnostic["cloud_requests"] == 0
            seed_proof["seed_records"] = diagnostic["records"]
            seed_proof["before_question_backup"] = seed_proof["resume_backup"]
            seed_proof["durable_seed_verified_before_question"] = True
        else:
            seed_proof = json.loads((seed_root / "before-question.json").read_text())
        assert seed_proof["durable_seed_verified_before_question"] and len(seed_proof["seed_records"]) == 3
        assert seed_proof["before_question_backup"]["prior_same_task_answers"] == 0
        assert len(seed_proof["documents_imported"]) == 4 and seed_proof["seed_http_status"] == 200

    fixture = json.loads((phase / "fixture.json").read_text())
    parent = json.loads((runtime / "backups/backend-chain-20261001/stage57/saved-audited-result.json").read_text())
    assert fixture["synthetic"] and all(parent["checks"].values())
    key = Path(args.api_key_file).read_text().strip()
    assert key
    os.environ.update(
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        DATABASE_PATH=str(root / "import-only.sqlite"),
        DATABASE_URL="postgresql+asyncpg://boot@/" + label + "?host=" + str(cluster / "socket") + "&port=25433",
        USE_POSTGRESQL="true",
        MODEL_PROVIDER="openai_compat",
        OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS="65536",
        OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
        OPENAI_COMPAT_API_KEY=key,
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        MEMORY_LLM_ENABLED="true",
        MEMORY_LLM_BASE_URL="https://api.deepseek.com",
        MEMORY_LLM_API_KEY=key,
        MEMORY_LLM_MODEL="deepseek-v4-pro",
        MEMORY_LLM_CONTEXT_WINDOW_TOKENS="65536",
        MEMORY_SOURCE_RECALL_ENABLED="true",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED="true",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS="30",
        CONTEXTUAL_MEMORY_SELECTION_ENABLED="true",
        CONTEXTUAL_DECISION_POLICY_ENABLED="true",
        REDIS_URL="redis://127.0.0.1:1/0",
        AUDIT_LOG_DIR=str(root / "audit"),
        BACKUP_DIR=str(root / "backups"),
        EMBEDDING_MODEL_PATH=str(runtime / "models/paraphrase-multilingual-MiniLM-L12-v2"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        INTENT_MODEL_PATH=str(root / "no-intent-model"),
        VECTOR_DB_PATH=str(root / "vectors"),
        CHARACTER_RAG_INDEX_ROOT=str(root / "no-character-index"),
        RAG_CITATIONS_ENABLED="false",
        CORRECTIVE_RAG_ENABLED="false",
        RERANKER_ENABLED="false",
        VLLM_ENABLED="false",
        VLLM_BASE_URL="http://127.0.0.1:1",
        VLLM_BASE_URLS="http://127.0.0.1:1",
        MULTIPERSONAL_BACKEND_URL="https://stage3-evaluation.invalid",
        ALLOWED_ORIGINS="https://stage3-evaluation.invalid",
        ALLOW_PUBLIC_REGISTRATION="false",
        SECURITY_MIDDLEWARE_ENABLED="true",
        BACKEND_WORKERS="1",
        CHAT_CONVERSATION_BURST="10",
        CHAT_SENDER_BURST="10",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    import asyncpg

    connection = await asyncpg.connect(user="boot", database="postgres", host=str(cluster / "socket"), port=25433)
    try:
        assert await connection.fetchval("SHOW data_directory") == str(cluster / "data")
        assert await connection.fetchval("SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)", source)
        assert not await connection.fetchval("SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)", label)
        await connection.execute("CREATE DATABASE " + label + " TEMPLATE " + source)
    finally:
        await connection.close()
    pg_tools = runtime / "tools/pgsql/usr/lib/postgresql/14/bin"
    pg_env = os.environ.copy()
    pg_env["LD_LIBRARY_PATH"] = str(runtime / "tools/pgsql/usr/lib/x86_64-linux-gnu")
    assert (pg_tools / "pg_dump").is_file() and (pg_tools / "pg_restore").is_file()
    if args.reuse_verified_seed:
        backup = seed_root / ("provider-block.dump" if resume_provider_block else "before-question.dump")
        assert (
            hashlib.sha256(backup.read_bytes()).hexdigest() == seed_proof["before_question_backup"]["database_sha256"]
        )
        await asyncio.to_thread(
            subprocess.run,
            [
                str(pg_tools / "pg_restore"),
                "-h",
                str(cluster / "socket"),
                "-p",
                "25433",
                "-U",
                "boot",
                "-d",
                label,
                "--exit-on-error",
                str(backup),
            ],
            env=pg_env,
            check=True,
            timeout=45,
            stdout=subprocess.DEVNULL,
        )
    vectors = (
        seed_root / ("provider-block-vectors" if resume_provider_block else "before-question-vectors")
        if args.reuse_verified_seed
        else cluster / "stage57-budget-fixed/vectors"
    )
    assert vectors.is_dir() and not vectors.is_symlink()
    shutil.copytree(vectors, root / "vectors")
    import httpx

    calls = []
    original_send = httpx.AsyncClient.send

    async def observed_send(client, request, **kwargs):
        if request.url.host != "api.deepseek.com":
            return await original_send(client, request, **kwargs)
        payload = json.loads(request.content)
        assert payload.get("model") == "deepseek-v4-pro", "Isolated provider configuration must use pro"
        (root / "last-sent-request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10))
        if any("<user_query>" in m.get("content", "") for m in payload.get("messages", [])):
            (root / ("primary-request-" + str(len(calls)) + ".json")).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + chr(10)
            )
        if any(
            "<user_query>" in m.get("content", "") and fixture["question"] in m["content"]
            for m in payload.get("messages", [])
        ):
            from html import unescape

            whole = chr(10).join(unescape(m["content"]) for m in payload["messages"])
            seen = {str(i) + ":" + doc["title"]: doc["content"] in whole for i, doc in enumerate(fixture["documents"])}
            (root / "primary-input-observed.json").write_text(
                json.dumps(
                    {
                        "request": payload,
                        "document_bodies_present": seen,
                        "private_source_present": fixture["source_message"] in whole,
                        "prepared": proof["prepared"],
                        "retrieval": proof["retrieval"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + chr(10)
            )
            assert (
                all(
                    seen[str(i) + ":" + fixture["documents"][i]["title"]]
                    for i in fixture["required_gate_document_indices"]
                )
                and fixture["source_message"] in whole
            ), "Complete requested public/private source missing before actual model"
        from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
        from inference.token_counting import token_counter_info

        input_bound = sum(estimated_tokens(m["content"]) + 4 for m in payload["messages"])
        output_reserved = payload["max_tokens"]
        budget = dict(
            counter=token_counter_info(),
            input_bound=input_bound,
            output_reserved=output_reserved,
            safety_margin=CONTEXT_SAFETY_MARGIN_TOKENS,
            configured_window=65536,
        )
        assert budget["counter"]["mode"] == "deepseek_v4_pro_bpe"
        assert input_bound + output_reserved + CONTEXT_SAFETY_MARGIN_TOKENS <= 65536
        (root / ("budget-before-send-" + str(len(calls)) + ".json")).write_text(json.dumps(budget, indent=2) + chr(10))
        response = await original_send(client, request, **kwargs)
        await response.aread()
        calls.append(dict(request=payload, http_status=response.status_code, response=response.json(), budget=budget))
        (root / "cloud-calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2) + "\n")
        return response

    httpx.AsyncClient.send = observed_send
    from character import memory_llm as ml

    admission = []
    original_parser = ml.parse_llm_proposals

    def observed_parser(response, **kwargs):
        proposals = original_parser(response, **kwargs)
        admission.append(dict(raw_response=response, input=kwargs, accepted=[asdict(p) for p in proposals]))
        (root / "writer-admission.json").write_text(
            json.dumps(admission, ensure_ascii=False, indent=2, default=str) + chr(10)
        )
        return proposals

    ml.parse_llm_proposals = observed_parser
    from services.character_context import CharacterContextService

    from api import generate as api
    from api.auth import _hash_password
    from app.main import create_app
    from cache.config_cache import invalidate_config_cache
    from character.memory_llm import get_memory_enrichment_scheduler, shutdown_memory_enrichment
    from db.adapter import db, is_pg_mode

    assert is_pg_mode()
    proof = {
        "synthetic_only": True,
        "transport": "authenticated_ASGI",
        "database_mode": "PostgreSQL",
        "database": label,
        "source_database": source,
        "citations_enabled": False,
        "primary_output_tokens": args.answer_tokens,
        "retrieval": [],
        "prepared": [],
        "generation": [],
        "auth_statuses": [],
    }
    original_guard_validation = None
    if args.compare_original_guard:
        import importlib.util
        import sys

        from character import output_guard

        prior_path = phase / "original-backend_character_output_guard.py"
        preflight = json.loads((phase / "preflight.json").read_text())
        assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == preflight["original_guard_sha256"]
        module_name = "_native_original_output_guard"
        spec = importlib.util.spec_from_file_location(module_name, prior_path)
        previous = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = previous
        spec.loader.exec_module(previous)
        original_guard_validation = output_guard.validate_reply
        proof["guard_validation_comparisons"] = []

        def observed_guard_validation(reply, guard):
            current = original_guard_validation(reply, guard)
            prior = previous.validate_reply(reply, guard)
            proof["guard_validation_comparisons"].append(
                dict(
                    actual_reply=reply,
                    actual_guard=asdict(guard) if guard is not None else None,
                    current_violations=list(current),
                    previous_violations=list(prior),
                    actual_cloud_calls_so_far=len(calls),
                    comparison_model_calls=0,
                )
            )
            (root / "guard-validation-observed.json").write_text(
                json.dumps(proof["guard_validation_comparisons"], ensure_ascii=False, indent=2) + chr(10)
            )
            return current

        output_guard.validate_reply = observed_guard_validation
    original_retrieve = api._retrieve_rag_bundle

    async def observed_retrieve(*pos, **kwargs):
        try:
            result = await original_retrieve(*pos, **kwargs)
        except Exception as error:
            proof.setdefault("retrieval_errors", []).append({"type": type(error).__name__, "message": str(error)})
            (root / "retrieval-error-observed.json").write_text(
                json.dumps(proof["retrieval_errors"], ensure_ascii=False, indent=2) + chr(10)
            )
            raise
        proof["retrieval"].append(result)
        return result

    api._retrieve_rag_bundle = observed_retrieve
    original_prepare = CharacterContextService.prepare_turn

    async def observed_prepare(service, turn, character_id):
        result = await original_prepare(service, turn, character_id)
        proof["prepared"].append(
            {
                "semantic_status": result.semantic_review_status,
                "policy_status": result.contextual_policy_status,
                "actual_interaction": asdict(result.interaction),
                "actual_decision": asdict(result.decision),
                "prepared_reply_guard": asdict(result.reply_guard),
                "history": list(result.history),
                "recall": result.memory_recall,
                "reference_context": result.compiled.reference_context,
                "used_memory_ids": list(result.compiled.used_memory_ids),
                "memory_packets": [asdict(packet) for packet in result.compiled.memory_packets],
                "selection_status": result.memory_selection_status,
                "received_at": result.received_at.isoformat(),
                "memory_budget": result.memory_budget,
                "episodic_context": result.compiled.episodic_reference_context,
                "source_candidate_context": result.compiled.source_candidate_context,
                "selection_candidate_count": result.memory_selection_candidate_count,
            }
        )
        (root / "preparation-observed.json").write_text(
            json.dumps(proof, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        return result

    CharacterContextService.prepare_turn = observed_prepare
    original_generation = api.generate_character_response

    async def observed_generation(request, generate):
        call_start = len(calls)
        result = await original_generation(request, generate)
        proof["generation"].append(
            {
                "messages": list(result.plan.messages),
                "retrieval": asdict(result.plan.retrieval),
                "response_mode": result.response_mode,
                "model_invoked": result.model_invoked,
                "citation_repair_status": result.citation_repair_status,
                "reply_guard_mode": request.reply_guard_mode,
                "guard_retried": result.guard_retried,
                "guard_violations": list(result.guard_violations),
                "guard_post_retry_violations": list(result.guard_post_retry_violations),
                "guard_fallback": result.guard_fallback,
                "observed_primary_calls": sum(
                    c["request"].get("max_tokens") == args.answer_tokens
                    and any("<user_query>" in m.get("content", "") for m in c["request"].get("messages", []))
                    for c in calls[call_start:]
                ),
            }
        )
        return result

    api.generate_character_response = observed_generation
    app = create_app()
    password = secrets.token_urlsafe(24)
    connection = await asyncpg.connect(user="boot", database=label, host=str(cluster / "socket"), port=25433)
    try:
        assert await connection.fetchval("SHOW data_directory") == str(cluster / "data")
        admin = await connection.fetchrow("SELECT id,username,role FROM users WHERE id=1")
        assert admin["role"] == "admin"
        password_hash = await asyncio.to_thread(_hash_password, password)
        assert (
            await connection.fetchval("UPDATE users SET password_hash=$1 WHERE id=1 RETURNING id", password_hash) == 1
        )
    finally:
        await connection.close()
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="https://stage3-evaluation.invalid",
                timeout=180,
            ) as client,
        ):
            await asyncio.to_thread(
                db.update_config,
                {
                    "modelProvider": "openai_compat",
                    "openaiCompatModel": "deepseek-v4-pro",
                    "openaiCompatBaseUrl": "https://api.deepseek.com",
                    "useKnowledgeBase": False,
                    "maxTokens": args.answer_tokens,
                    "temperature": 0.2,
                    "topP": 0.9,
                },
            )
            invalidate_config_cache()
            login = await client.post("/api/auth/login", json={"username": admin["username"], "password": password})
            me = await client.get("/api/auth/me")
            proof["auth_statuses"] = [login.status_code, me.status_code]
            assert proof["auth_statuses"] == [200, 200] and str(me.json()["user"]["id"]) == "1"
            if args.reuse_verified_seed:
                proof["knowledge_base"] = seed_proof["knowledge_base"]
                proof["documents_imported"] = seed_proof["documents_imported"]
                proof["knowledge_imports_replayed"] = 0
            else:
                base = await client.post(
                    "/api/knowledge/bases",
                    json={
                        "name": fixture["knowledge_base"],
                        "description": "Complete synthetic four-document mixed-chain test",
                    },
                )
                assert base.status_code == 200
                base_id = base.json()["base"]["id"]
                proof["knowledge_base"] = base.json()
                proof["documents_imported"] = []
                for document in fixture["documents"]:
                    imported = await client.post(
                        "/api/knowledge/documents", json={**document, "knowledge_base_id": base_id}
                    )
                    proof["documents_imported"].append({"status": imported.status_code, "response": imported.json()})
                    (root / "imported-documents.json").write_text(
                        json.dumps(proof["documents_imported"], ensure_ascii=False, indent=2) + chr(10)
                    )
                    assert imported.status_code == 200
            chat_password = secrets.token_urlsafe(24)
            if args.reuse_verified_seed:
                connection = await asyncpg.connect(
                    user="boot", database=label, host=str(cluster / "socket"), port=25433
                )
                try:
                    assert await connection.fetchval("SHOW data_directory") == str(cluster / "data")
                    user = dict(
                        await connection.fetchrow(
                            "SELECT id,username,role FROM users WHERE username=$1", "stage64-native-user"
                        )
                    )
                    assert str(user["id"]) == seed_proof["seed_scope"]["owner"] and user["role"] != "admin"
                    await connection.execute(
                        "UPDATE users SET password_hash=$1 WHERE id=$2",
                        await asyncio.to_thread(_hash_password, chat_password),
                        user["id"],
                    )
                finally:
                    await connection.close()
            else:
                user = await asyncio.to_thread(
                    db.add_user, "stage64-native-user", await asyncio.to_thread(_hash_password, chat_password), False
                )
            client.cookies.clear()
            login = await client.post(
                "/api/auth/login", json={"username": "stage64-native-user", "password": chat_password}
            )
            me = await client.get("/api/auth/me")
            proof["chat_auth_statuses"] = [login.status_code, me.status_code]
            assert (
                proof["chat_auth_statuses"] == [200, 200]
                and str(me.json()["user"]["id"]) == str(user["id"])
                and str(user["id"]) != "1"
            )
            from repositories.character_memory import DatabaseCharacterMemoryRepository

            from character.models import UserScope

            owner = str(user["id"])
            scope = UserScope("web", "web-character", owner, owner, "private")
            repo = DatabaseCharacterMemoryRepository(db)
            if args.reuse_verified_seed:
                scheduler = get_memory_enrichment_scheduler()
                proof.update(
                    seed_http_status=seed_proof["seed_http_status"],
                    seed_response=seed_proof["seed_response"],
                    seed_scope=seed_proof["seed_scope"],
                    seed_template_verified=True,
                    seed_model_calls_replayed=0,
                )
            else:
                seed_response = await client.post(
                    "/api/generate",
                    json={
                        "message": fixture["source_message"],
                        "characterId": "tsukiyashiro_kisaki",
                        "loraId": "default",
                        "sessionId": "stage78-native",
                        "sessionType": "private",
                    },
                )
                proof.update(
                    seed_http_status=seed_response.status_code,
                    seed_response=seed_response.json(),
                    seed_scope={"owner": owner, "conversation": owner, "character": "tsukiyashiro_kisaki"},
                )
                (root / "seed-response-observed.json").write_text(
                    json.dumps(proof, ensure_ascii=False, indent=2, default=str) + chr(10)
                )
                assert seed_response.status_code == 200
                scheduler = get_memory_enrichment_scheduler()
                flushed = False
                for _observation in range(4):
                    flushed = await scheduler.flush_memory(timeout=45)
                    (root / "scheduler-seed-status.json").write_text(
                        json.dumps(
                            {"flushed": flushed, "status": asdict(scheduler.status)},
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                        + chr(10)
                    )
                    if flushed:
                        break
                assert flushed, "Actual writer still active after observation; inspect same handle"
            if not args.reuse_verified_seed or resume_provider_block:
                proof["bridge_turns"] = list(seed_proof["successful_bridge_turns"]) if resume_provider_block else []
                start = len(proof["bridge_turns"])
                if resume_provider_block:
                    proof["successful_bridge_turns_replayed"] = 0
                for index, bridge in enumerate(fixture["bridges"]):
                    if index < start:
                        continue
                    bridge_response = await client.post(
                        "/api/generate",
                        json={
                            "message": bridge["message"],
                            "characterId": "tsukiyashiro_kisaki",
                            "loraId": "default",
                            "sessionId": "stage78-new-topic-" + str(index + 1),
                            "sessionType": "private",
                        },
                    )
                    observed = {
                        "index": index,
                        "status": bridge_response.status_code,
                        "response": bridge_response.json(),
                        "input_chars": len(bridge["message"]),
                    }
                    proof["bridge_turns"].append(observed)
                    (root / "bridge-progress.json").write_text(
                        json.dumps(proof["bridge_turns"], ensure_ascii=False, indent=2) + chr(10)
                    )
                    assert bridge_response.status_code == 200
                    flushed = False
                    for _bridge_observation in range(4):
                        flushed = await scheduler.flush_memory(timeout=45)
                        (root / "scheduler-bridge-status.json").write_text(
                            json.dumps(
                                {"bridge": index, "flushed": flushed, "status": asdict(scheduler.status)},
                                indent=2,
                                default=str,
                            )
                            + chr(10)
                        )
                        if flushed:
                            break
                    assert flushed, "Bridge writer still live after observation; inspect same native handle"
                proof["bridge_turns_completed"] = len(proof["bridge_turns"])
            else:
                proof["bridge_turns"] = seed_proof["bridge_turns"]
                proof["bridge_turns_completed"] = seed_proof["bridge_turns_completed"]
                proof["successful_bridge_turns_replayed"] = 0
            seed_records = await repo.list_memory_records(
                "tsukiyashiro_kisaki", scope, limit=None, include_inactive=True
            )
            if args.reuse_verified_seed:
                assert seed_records == seed_proof["seed_records"]
                proof["original_seed_claims_exact_verified"] = True
            assert len(seed_records) == 3, "Unrelated synthetic directory must not manufacture personal memories"
            proof.update(
                seed_method="actual_authenticated_native_generate_then_real_semantic_writer_no_fabricated_memories",
                seed_records=seed_records,
                writer_admission_before_question=list(admission),
                scheduler_before_question=asdict(scheduler.status),
                calls_before_question=len(calls),
                prepared_before_question_count=len(proof["prepared"]),
            )
            connection = await asyncpg.connect(user="boot", database=label, host=str(cluster / "socket"), port=25433)
            try:
                async with connection.transaction(readonly=True):
                    assert await connection.fetchval("SHOW data_directory") == str(cluster / "data")
                    owner_key = json.dumps(("web", "web-character", owner))
                    scope_key = json.dumps(("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner))
                    sources = await connection.fetch(
                        "SELECT * FROM memory_sources WHERE owner_key=$1 AND scope_key=$2", owner_key, scope_key
                    )
                    assert any(row["body"] == fixture["source_message"] for row in sources)
                    proof["durable_seed_sources"] = [dict(row) for row in sources]
                    proof["durable_seed_verified_before_question"] = True
            finally:
                await connection.close()
            await asyncio.to_thread(db.update_config, {"useKnowledgeBase": True})
            invalidate_config_cache()
            if not args.reuse_verified_seed or resume_provider_block:
                backup = root / "before-question.dump"
                await asyncio.to_thread(
                    subprocess.run,
                    [
                        str(pg_tools / "pg_dump"),
                        "-h",
                        str(cluster / "socket"),
                        "-p",
                        "25433",
                        "-U",
                        "boot",
                        "-d",
                        label,
                        "-Fc",
                        "-f",
                        str(backup),
                    ],
                    env=pg_env,
                    check=True,
                    timeout=45,
                    stdout=subprocess.DEVNULL,
                )
                assert backup.stat().st_size > 1000
                shutil.copytree(root / "vectors", root / "before-question-vectors")
                proof["before_question_backup"] = {
                    "database_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
                    "bytes": backup.stat().st_size,
                    "prior_same_task_answers": 0,
                    "successful_seed_claim_ids": [r["id"] for r in seed_records],
                }
            else:
                proof["before_question_backup"] = seed_proof["before_question_backup"]
                proof["restore_before_question_not_terminal_answer"] = True
            (root / "before-question.json").write_text(
                json.dumps(proof, ensure_ascii=False, indent=2, default=str) + chr(10)
            )
            response = await client.post(
                "/api/generate",
                json={
                    "message": fixture["question"],
                    "characterId": "tsukiyashiro_kisaki",
                    "loraId": "default",
                    "sessionId": "stage78-native",
                    "sessionType": "private",
                },
            )
            assert await get_memory_enrichment_scheduler().flush_memory(timeout=45)
            proof.update(
                http_status=response.status_code,
                response=response.json(),
                user_fact_records_after=[
                    row
                    for row in await repo.list_memory_records(
                        "tsukiyashiro_kisaki", scope, limit=None, include_inactive=True
                    )
                ],
                sync_pending_final=len(db._pending),
                writer_admission=admission,
                scheduler_final=asdict(get_memory_enrichment_scheduler().status),
                cloud_calls=len(calls),
                primary_calls=sum(
                    c["request"].get("max_tokens") == args.answer_tokens
                    and any("<user_query>" in m.get("content", "") for m in c["request"].get("messages", []))
                    for c in calls
                ),
            )
            (root / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str) + "\n")
            if args.expect_primary_blocked:
                assert proof["http_status"] in (200, 500) and proof["primary_calls"] == 1
                gate = json.loads((root / "primary-input-observed.json").read_text())
                assert not all(
                    gate["document_bodies_present"][str(i) + ":" + fixture["documents"][i]["title"]]
                    for i in fixture["required_gate_document_indices"]
                )
            else:
                expected_api_turns = (
                    (1 + len(fixture["bridges"]) - len(seed_proof["successful_bridge_turns"]))
                    if resume_provider_block
                    else (1 if args.reuse_verified_seed else 11)
                )
                assert proof["http_status"] == 200 and len(proof["generation"]) == expected_api_turns
                # Existing guard retry is an additional actual provider request,
                # not another successful native API turn. Bind the measurement
                # to each actual generation receipt instead of assuming one call.
                for generation in proof["generation"]:
                    assert generation["response_mode"] != "task_composite"
                    expected_requests = int(generation["model_invoked"]) * (1 + int(generation["guard_retried"]))
                    assert generation["observed_primary_calls"] == expected_requests
                assert proof["primary_calls"] == sum(g["observed_primary_calls"] for g in proof["generation"])
            print(
                json.dumps(
                    {
                        "result_saved": True,
                        "http_status": response.status_code,
                        "actual_cloud_calls": len(calls),
                        "actual_primary_calls": proof["primary_calls"],
                    }
                )
            )
    finally:
        await shutdown_memory_enrichment()
        httpx.AsyncClient.send = original_send
        if original_guard_validation is not None:
            output_guard.validate_reply = original_guard_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-primary-blocked", action="store_true")
    parser.add_argument("--reuse-verified-seed", action="store_true")
    parser.add_argument("--compare-original-guard", action="store_true")
    parser.add_argument("--resume-provider-block", action="store_true")
    parser.add_argument("--provider-access-restored", action="store_true")
    parser.add_argument("--seed-variant", default="native-pg-baseline")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--variant", default="native-pg")
    parser.add_argument("--answer-tokens", type=int, choices=[1024, 2048], default=2048)
    asyncio.run(run(parser.parse_args()))
