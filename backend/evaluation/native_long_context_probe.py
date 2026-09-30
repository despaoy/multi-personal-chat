"""Disposable PostgreSQL + actual app authentication, queue and native provider."""

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


def audit_long_plan(proof, cloud_calls):
    """Check complete source provenance against recorded wire messages."""
    source = proof["cases"][5]["message"]
    query = proof["cases"][6]["message"]
    claims = [c for c in proof["claims"] if json.loads(c.get("evidence_json") or "[]") == [source]]
    cold_calls = [
        c
        for c in cloud_calls
        if c["request"].get("max_tokens") == 1024 and query in c["request"]["messages"][-1]["content"]
    ]
    packets = []
    for call in cold_calls:
        content = unescape(call["request"]["messages"][-1]["content"])
        region = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", content, re.S)
        if region:
            for line in region.group(1).splitlines():
                if line.startswith("- {"):
                    packets.append(json.loads(line[2:]))
    plan_packets = [p for p in packets if p.get("evidence") == [source]]
    reply = proof["generation"][6]["response"].get("reply", "")
    return dict(
        complete_plan_source_persisted=bool(claims),
        plan_stored_as_source_observation=bool(claims)
        and all(
            json.loads(c["metadata_json"]).get("content_semantics") == "quoted_source"
            and json.loads(c["metadata_json"]).get("described_subject") == "not_resolved"
            for c in claims
        ),
        complete_plan_reaches_cold_answer_wire=bool(plan_packets),
        cold_plan_packet_keeps_observation_semantics=bool(plan_packets)
        and all(
            p.get("subject_scope") == "not_resolved"
            and p.get("content_semantics") == "quoted_source"
            and p.get("speaker_role") == "user"
            and p.get("temporal_mode") == "observation"
            for p in plan_packets
        ),
        cold_answer_wire_has_no_dialogue_history=bool(cold_calls)
        and all(
            len(c["request"]["messages"]) == 2 and c["request"]["messages"][0]["role"] == "system" for c in cold_calls
        ),
        long_plan_is_not_system_instruction=bool(cold_calls)
        and all(
            source not in m["content"] for c in cold_calls for m in c["request"]["messages"] if m["role"] == "system"
        ),
        cold_answer_preserves_rain_cancellation=bool(
            re.search(r"下雨.{0,15}(取消|不去)|(取消|不去).{0,15}下雨", reply)
        ),
        cold_answer_preserves_not_started=bool(
            re.search(r"(尚未|还没(?:有)?|未曾).{0,6}(开始|实施)|未开始|未实施", reply)
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
    cases = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures/deepseek_native_long_cases.json").read_text()
    )["cases"]
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
    db.update_config(dict(useKnowledgeBase=False, temperature=0.2, maxTokens=1024, topP=0.9))
    app = create_app()
    proof = dict(
        cases=cases,
        generation=[],
        prepared_diagnostics=prepared_diagnostics,
        generation_diagnostics=generation_diagnostics,
        provider="native_openai_compat",
        transport="authenticated_ASGI",
        source_ablation=False,
        rag_enabled=False,
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
            for case in cases:
                cold_state["active"] = case["id"] == "cold_personal_plan_read"
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
        final_reply = proof["generation"][4]["response"].get("reply", "")
        final_history = prepared_diagnostics[4]["history"] if prepared_diagnostics else []
        final_model_messages = generation_diagnostics[4].get("model_messages", []) if generation_diagnostics else []
        long_queries = [c["message"] for c in cases[1:4]]
        proof["checks"] = dict(
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
            full_prior_materials_loaded=all(
                any(h["role"] == "user" and h["content"] == q for h in final_history) for q in long_queries
            ),
            full_prior_materials_reach_model=all(
                any(h["role"] == "user" and h["content"] == q for h in final_model_messages) for q in long_queries
            ),
            full_current_materials_reach_model=all(
                any(q in h.get("content", "") for c in cloud_calls for h in c["request"].get("messages", []))
                for q in long_queries
            ),
            all_long_reviewers_applied=len(prepared_diagnostics) == len(cases)
            and all(
                p["selection_status"] in {"selected", "empty"}
                and p["semantic_status"] == "applied"
                and p["policy_status"] == "applied"
                for p in prepared_diagnostics[1:4]
            ),
            final_answer_contains_all_eight_fields=all(value in final_reply for value in cases[4]["expected"]),
            no_wrong_local_completions=not local_review_calls,
            memory_writer_no_errors=proof["memory_status"]["failed"] == 0,
            fiction_not_persisted_as_personal_claims=not any(
                any(
                    v in json.dumps(c, ensure_ascii=False)
                    for v in ["CX-417-K", "青鹭过桥", "顾澄", "榆影仓房", "秋月二十三", "待验收"]
                )
                for c in proof["claims"]
            ),
        )

        payloads = []
        for call in cloud_calls:
            messages = call["request"].get("messages", [])
            if not messages:
                continue
            try:
                payload = json.loads(messages[-1]["content"])
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                payloads.append((call["request"].get("max_tokens"), payload))
        answer_calls = [c for c in cloud_calls if c["request"].get("max_tokens") == 1024]
        proof["checks"].update(
            complete_long_queries_reach_answer=all(
                any(q in c["request"]["messages"][-1]["content"] for c in answer_calls) for q in long_queries
            ),
            complete_long_queries_reach_semantic=all(
                any(p.get("current_message") == q for _, p in payloads) for q in long_queries
            ),
            complete_long_queries_reach_selection=all(
                any(b == 2048 and p.get("query") == q for b, p in payloads) for q in long_queries
            ),
            complete_long_queries_reach_policy=all(
                any(b == 160 and p.get("query") == q for b, p in payloads) for q in long_queries
            ),
            complete_personal_plan_reaches_writer=any(
                p.get("current_user_message") == cases[5]["message"] for _, p in payloads
            ),
            cold_read_has_no_history=len(prepared_diagnostics[-1]["history"]) == 0
            and (cold_state["underlying_history_count"] or 0) > 0,
            cold_plan_answer_has_all_fields=all(
                v in proof["generation"][-1]["response"].get("reply", "") for v in cases[6]["expected"]
            ),
            cold_plan_has_selected_memories=bool(prepared_diagnostics[-1]["used_memory_ids"]),
        )
        proof["controlled_ablation"] = dict(
            history_only_on_last_personal_plan_read=True,
            raw_source_recall_enabled=False,
            underlying_history_count=cold_state["underlying_history_count"],
        )
        proof["checks"].update(audit_long_plan(proof, cloud_calls))
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
