"""Authenticated complete six-field personal profile lookup."""

import argparse
import asyncio
import base64
import json
import os
import re
import secrets
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from html import unescape
from pathlib import Path


async def run(args):
    phase = Path(args.phase).resolve()
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    assert phase.parent == runtime / "backups/backend-chain-20261001" and re.fullmatch(r"stage[1-9]\d*", phase.name)
    assert re.fullmatch(r"native-pg(?:-[a-z]{1,12})*", args.variant)
    root = phase / args.variant
    root.mkdir(mode=0o700, exist_ok=False)
    cluster = runtime / "evaluations/r148pg.s3"
    label = "stage3_" + phase.name + "_" + args.variant.replace("-", "_")
    source = "stage3_stage57_budget_fixed"
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
    vectors = cluster / "stage57-budget-fixed/vectors"
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
        if payload.get("max_tokens") == args.answer_tokens and any(
            "<user_query>" in m.get("content", "") for m in payload.get("messages", [])
        ):
            wire = unescape(payload["messages"][-1]["content"])
            (root / "primary-input-observed.json").write_text(
                json.dumps(
                    {"request": payload, "prepared": proof["prepared"], "seed_records": proof.get("seed_records", [])},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            match = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
            assert fixture["question"] in wire and match, "Original full profile question lost"
            assert all(item["expected"] in match[1] for item in fixture["sources"]), (
                "A requested saved profile fact was lost before actual model"
            )
            assert len(proof["prepared"][-1]["memory_packets"]) == len(fixture["sources"])
            assert all(
                item["body"] not in m["content"]
                for item in fixture["sources"]
                for m in payload["messages"]
                if m["role"] == "system"
            )
        response = await original_send(client, request, **kwargs)
        await response.aread()
        calls.append(dict(request=payload, http_status=response.status_code, response=response.json()))
        (root / "cloud-calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2) + "\n")
        return response

    httpx.AsyncClient.send = observed_send
    from api import generate as api
    from api.auth import _hash_password
    from app.main import create_app
    from cache.config_cache import invalidate_config_cache
    from character.memory_llm import get_memory_enrichment_scheduler, shutdown_memory_enrichment
    from db.adapter import db, is_pg_mode
    from services.character_context import CharacterContextService

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
    original_retrieve = api._retrieve_rag_bundle

    async def observed_retrieve(*pos, **kwargs):
        result = await original_retrieve(*pos, **kwargs)
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
                "history": list(result.history),
                "recall": result.memory_recall,
                "reference_context": result.compiled.reference_context,
                "used_memory_ids": list(result.compiled.used_memory_ids),
                "memory_packets": [asdict(packet) for packet in result.compiled.memory_packets],
                "selection_status": result.memory_selection_status,
                "received_at": result.received_at.isoformat(),
                "memory_budget": result.memory_budget,
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
        result = await original_generation(request, generate)
        proof["generation"].append(
            {
                "messages": list(result.plan.messages),
                "retrieval": asdict(result.plan.retrieval),
                "response_mode": result.response_mode,
                "model_invoked": result.model_invoked,
                "citation_repair_status": result.citation_repair_status,
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
            chat_password = secrets.token_urlsafe(24)
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
            from character.memory_extractor import extract_memories
            from character.models import UserScope
            from character.rule_memory_writer import write_rule_memory
            from repositories.character_memory import DatabaseCharacterMemoryRepository

            owner = str(user["id"])
            scope = UserScope("web", "web-character", owner, owner, "private")
            repo = DatabaseCharacterMemoryRepository(db)
            anchor = datetime.now(timezone.utc)
            seed_sources = []
            for item in fixture["sources"]:
                stamp = datetime.fromisoformat(item["observed_at"])
                assert stamp <= anchor, "Synthetic source time cannot be in the future"
                assert (
                    await repo.capture_source(
                        "tsukiyashiro_kisaki", scope, source_message_id=item["id"], body=item["body"], observed_at=stamp
                    )
                    == "recorded"
                )
                parsed = extract_memories(item["body"], reference_time=stamp)
                assert len(parsed) == 1 and parsed[0].memory_key == item["field"]
                assert await write_rule_memory(
                    repo, "tsukiyashiro_kisaki", scope, parsed[0], item["id"], observed_at=stamp
                )
                seed_sources.append({**item, "observed_at": stamp.isoformat()})
            seed_records = await repo.list_memory_records(
                "tsukiyashiro_kisaki", scope, limit=None, include_inactive=True
            )
            assert len(seed_records) == len(fixture["sources"]) and all(
                row["status"] == "active" for row in seed_records
            )
            assert {row["memory_key"] for row in seed_records} == {item["field"] for item in fixture["sources"]}
            proof.update(
                seed_method="actual_source_capture_and_rule_write",
                seed_scope={"owner": owner, "conversation": owner, "character": "tsukiyashiro_kisaki"},
                seed_sources=seed_sources,
                seed_records=seed_records,
                seed_reference_time=anchor.isoformat(),
            )
            connection = await asyncpg.connect(user="boot", database=label, host=str(cluster / "socket"), port=25433)
            try:
                async with connection.transaction(readonly=True):
                    assert await connection.fetchval("SHOW data_directory") == str(cluster / "data")
                    durable = await connection.fetch(
                        "SELECT id,status,content,memory_key,valid_from,valid_to FROM character_memories WHERE character_id=$1 AND platform=$2 AND adapter=$3 AND sender_id=$4 AND conversation_id=$5 AND conversation_type=$6",
                        "tsukiyashiro_kisaki",
                        "web",
                        "web-character",
                        owner,
                        owner,
                        "private",
                    )
                    assert len(durable) == len(fixture["sources"]) and all(row["status"] == "active" for row in durable)
                    for item in fixture["sources"]:
                        row = next(row for row in durable if row["memory_key"] == item["field"])
                        assert item["expected"] in row["content"] and datetime.fromisoformat(
                            row["valid_from"]
                        ) == datetime.fromisoformat(item["observed_at"])
                    proof["durable_declared_intervals_verified"] = True
                    proof["durable_seed_verified_before_question"] = True
            finally:
                await connection.close()
            response = await client.post(
                "/api/generate",
                json={
                    "message": fixture["question"],
                    "characterId": "tsukiyashiro_kisaki",
                    "loraId": "default",
                    "sessionId": "stage64-native",
                    "sessionType": "private",
                },
            )
            assert await get_memory_enrichment_scheduler().flush_memory(timeout=45)
            proof.update(
                http_status=response.status_code,
                response=response.json(),
                profile_records_after=[
                    row
                    for row in await repo.list_memory_records(
                        "tsukiyashiro_kisaki", scope, limit=None, include_inactive=True
                    )
                ],
                sync_pending_final=len(db._pending),
                cloud_calls=len(calls),
                primary_calls=sum(
                    c["request"].get("max_tokens") == args.answer_tokens
                    and any("<user_query>" in m.get("content", "") for m in c["request"].get("messages", []))
                    for c in calls
                ),
            )
            (root / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str) + "\n")
            if args.expect_primary_blocked:
                assert proof["http_status"] == 500 and proof["primary_calls"] == 0
            else:
                assert proof["http_status"] == 200 and proof["primary_calls"] == 1
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-primary-blocked", action="store_true")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--variant", default="native-pg")
    parser.add_argument("--answer-tokens", type=int, choices=[1024, 2048], default=2048)
    asyncio.run(run(parser.parse_args()))
