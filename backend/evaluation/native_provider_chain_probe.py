"""Disposable PostgreSQL + actual app authentication, queue and native provider."""

import argparse
import asyncio
import base64
import json
import os
import re
import secrets
import time
from dataclasses import asdict
from pathlib import Path


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

    if not 0 < args.semantic_review_timeout <= 120:
        raise ValueError("Semantic review timeout must be within (0, 120]")
    ROOT, OUT, key_path = isolated_probe_paths(args.root, args.run_label, args.api_key_file)
    bootstrap_url = "postgresql+asyncpg://boot@/postgres?host=" + str(ROOT / "socket") + "&port=25433"
    await verify_cluster(bootstrap_url, ROOT / "data")
    database_name = "stage3_" + args.run_label.replace("-", "_")
    OUT.mkdir(exist_ok=False)
    cases = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures/deepseek_native_chain_cases.json").read_text()
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
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED=str(args.reviewers).lower(),
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS=str(args.semantic_review_timeout),
        CONTEXTUAL_MEMORY_SELECTION_ENABLED=str(args.reviewers).lower(),
        CONTEXTUAL_DECISION_POLICY_ENABLED=str(args.reviewers).lower(),
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
        started = time.monotonic()
        response = await original_send(client, request, **kwargs)
        await response.aread()
        data = response.json()
        cloud_calls.append(
            dict(
                url=str(request.url),
                request=body,
                http_status=response.status_code,
                response=data,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        (OUT / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
        return response

    httpx.AsyncClient.send = observed_send
    from app.main import create_app
    from character.memory_llm import get_memory_enrichment_scheduler, shutdown_memory_enrichment
    from db.adapter import db, is_pg_mode
    from infra.concurrency_control import inference_runtime
    from services.character_context import CharacterContextService

    evaluation_state = {"cold": False}
    prepared_diagnostics = []
    original_history = CharacterContextService._load_history
    original_prepare = CharacterContextService.prepare_turn

    async def ablated_history(service, turn, scope, character_id):
        actual = await original_history(service, turn, scope, character_id)
        if evaluation_state["cold"]:
            evaluation_state["cold_underlying_history_count"] = len(actual)
            return []
        return actual

    async def observed_prepare(service, turn, character_id):
        prepared = await original_prepare(service, turn, character_id)
        prepared_diagnostics.append(
            dict(
                user_acts=[dict(id=s.signal_id, score=s.score) for s in prepared.interaction.user_acts],
                semantic_status=prepared.semantic_review_status,
                semantic_triggers=prepared.semantic_review_reasons,
                selection_status=prepared.memory_selection_status,
                selection_reason=prepared.memory_selection_reason,
                selection_candidates=prepared.memory_selection_candidate_count,
                policy_status=prepared.contextual_policy_status,
                policy_reason=prepared.contextual_policy_reason,
                decision_strategies=prepared.decision.strategy_ids,
                dynamic_context=prepared.compiled.dynamic_context,
                cold=evaluation_state["cold"],
                history_count=len(prepared.history),
                used_memory_ids=prepared.compiled.used_memory_ids,
                reference_context=prepared.compiled.reference_context,
                episodic_context=prepared.compiled.episodic_reference_context,
            )
        )
        return prepared

    CharacterContextService._load_history = ablated_history
    CharacterContextService.prepare_turn = observed_prepare
    assert is_pg_mode()
    db.update_config(dict(useKnowledgeBase=False, temperature=0.2, maxTokens=1024, topP=0.9))
    app = create_app()
    proof = dict(
        postgres_real=True,
        authentication="actual_register_login_cookie",
        queue="native_inference_runtime",
        generation_provider="native_openai_compat",
        model="deepseek-v4-pro",
        production_modified=False,
        generation=[],
    )
    password = secrets.token_urlsafe(24)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://stage3-evaluation.invalid", timeout=180
        ) as client:
            register = await client.post("/api/auth/register", json=dict(username=args.run_label, password=password))
            proof["registration_status"] = register.status_code
            client.cookies.clear()
            login = await client.post("/api/auth/login", json=dict(username=args.run_label, password=password))
            proof["login_status"] = login.status_code
            proof["login_cookie_received"] = "access_token" in client.cookies
            me = await client.get("/api/auth/me")
            proof["authenticated_me_status"] = me.status_code
            assert register.status_code == 200 and login.status_code == 200 and me.status_code == 200
            for index, message in enumerate(case["message"] for case in cases[:3]):
                session = args.run_label if index < 2 else args.run_label + "-cold"
                evaluation_state["cold"] = index == 2
                if index == 2:
                    proof["cold_history_records_before_request"] = len(db.get_messages(session_id=session))
                    assert proof["cold_history_records_before_request"] == 0
                response = await client.post(
                    "/api/generate",
                    json=dict(
                        message=message,
                        characterId="tsukiyashiro_kisaki",
                        loraId="default",
                        sessionId=session,
                        sessionType="private",
                    ),
                )
                body = response.json()
                proof["generation"].append(dict(index=index, http_status=response.status_code, response=body))
                scheduler = get_memory_enrichment_scheduler()
                if scheduler is not None:
                    assert await scheduler.flush_memory(timeout=90)
                    proof["generation"][-1]["memory_status"] = asdict(scheduler.status)
                identity = str(me.json()["user"]["id"])
                proof["generation"][-1]["stored_claims"] = db.list_character_memory_claims(
                    "tsukiyashiro_kisaki",
                    "web",
                    "web-character",
                    identity,
                    "private",
                    identity,
                    limit=None,
                    include_inactive=True,
                )
            evaluation_state["cold"] = False
            if args.reviewers:
                dynamic_cases = [
                    item["message"]
                    for item in json.loads(
                        (
                            Path(__file__).resolve().parents[1] / "tests/fixtures/deepseek_native_review_cases.json"
                        ).read_text()
                    )["cases"]
                ]
                for message in dynamic_cases:
                    response = await client.post(
                        "/api/generate",
                        json=dict(
                            message=message,
                            characterId="tsukiyashiro_kisaki",
                            loraId="default",
                            sessionId=args.run_label + "-dynamic",
                            sessionType="private",
                        ),
                    )
                    proof["generation"].append(
                        dict(http_status=response.status_code, response=response.json(), dynamic_case=True)
                    )
                    assert await get_memory_enrichment_scheduler().flush_memory(timeout=90)
            proof["local_review_calls"] = local_review_calls
            queue_before = inference_runtime.stats()["submitted"]
            session = args.run_label + "-cold"
            first = asyncio.create_task(
                client.post(
                    "/api/generate",
                    json=dict(
                        message=cases[3]["message"],
                        characterId="tsukiyashiro_kisaki",
                        loraId="default",
                        sessionId=session,
                        sessionType="private",
                    ),
                )
            )
            for _ in range(300):
                if inference_runtime.stats()["submitted"] > queue_before:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("first request was not admitted")
            first_response, second_response = await asyncio.gather(
                first,
                client.post(
                    "/api/generate",
                    json=dict(
                        message=cases[4]["message"],
                        characterId="tsukiyashiro_kisaki",
                        loraId="default",
                        sessionId=session,
                        sessionType="private",
                    ),
                ),
            )
            proof["concurrent_same_session"] = [
                dict(http_status=response.status_code, response=response.json())
                for response in [first_response, second_response]
            ]
            scheduler = get_memory_enrichment_scheduler()
            assert await scheduler.flush_memory(timeout=90)
            proof["memory_status_after_concurrency"] = asdict(scheduler.status)
            proof["claims_after_concurrency"] = db.list_character_memory_claims(
                "tsukiyashiro_kisaki",
                "web",
                "web-character",
                identity,
                "private",
                identity,
                limit=None,
                include_inactive=True,
            )
            proof["prepared_diagnostics"] = prepared_diagnostics
            proof["cold_underlying_history_count"] = evaluation_state.get("cold_underlying_history_count")
            proof["controlled_ablation"] = {
                "history_read_override_on_third_turn_only": True,
                "raw_source_recall_enabled": False,
                "provider_authentication_queue_storage_unmodified": True,
            }
            proof["queue_stats"] = inference_runtime.stats()
            proof["cloud_calls"] = [
                dict(
                    http_status=c["http_status"],
                    actual_model=c["response"].get("model"),
                    finish_reasons=[x.get("finish_reason") for x in c["response"].get("choices", [])],
                    output_budget=c["request"].get("max_tokens"),
                    content_chars=[
                        len(x.get("message", {}).get("content") or "") for x in c["response"].get("choices", [])
                    ],
                )
                for c in cloud_calls
            ]
            proof["checks"] = {
                "all_generations_success": all(x["http_status"] == 200 for x in proof["generation"]),
                "all_cloud_calls_completed": all(
                    c["http_status"] == 200
                    and all(x.get("finish_reason") == "stop" for x in c["response"].get("choices", []))
                    for c in cloud_calls
                ),
                "cold_answer_has_all_four_fields": all(
                    value in proof["generation"][2]["response"].get("reply", "")
                    for value in ["岚舟", "宣城", "丽水", "环境工程"]
                ),
                "cold_compilation_history_is_zero": prepared_diagnostics[2]["history_count"] == 0,
                "cold_compilation_has_four_memories": len(prepared_diagnostics[2]["used_memory_ids"]) == 4,
                "cold_compilation_has_no_raw_source_context": not prepared_diagnostics[2]["episodic_context"],
                "same_session_concurrent_generations_completed": all(
                    item["http_status"] == 200 for item in proof["concurrent_same_session"]
                ),
                "queued_read_observes_latest_complete_update": all(
                    value in proof["concurrent_same_session"][1]["response"].get("reply", "")
                    for value in ["舟山", "宣城", "环境工程"]
                ),
                "native_memory_writer_no_errors": proof["memory_status_after_concurrency"]["failed"] == 0,
                "actual_model_is_deepseek_v4_pro": all(
                    c["response"].get("model") == "deepseek-v4-pro" for c in cloud_calls
                ),
                "native_queue_has_no_failures": proof["queue_stats"]["failed"] == 0
                and proof["queue_stats"]["completed"] == (8 if args.reviewers else 5),
                "current_residence_persisted": any(
                    c["memory_key"] == "user_residence" and c["status"] == "active" and "舟山" in c["content"]
                    for c in proof["claims_after_concurrency"]
                ),
            }
            (OUT / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
            if args.reviewers:
                boundary = prepared_diagnostics[3:6]

                def compiled_boundary(p, *, explicit=False):
                    # Assert the independent behavioral contract, not the
                    # compiler constant: a renamed template cannot make
                    # advice or questions silently acceptable.
                    advice = r"不得提供建议|不给方案"
                    if not explicit:
                        advice += r"|未请求建议时，不自动给方案"
                    questions = r"本轮不得追问|本轮不追问|本轮不得使用心理咨询式追问或任何追问"
                    text = p["dynamic_context"]
                    return bool(re.search(advice, text) and re.search(questions, text))

                proof["checks"].update(
                    optional_reviewers_never_use_local_provider=not local_review_calls,
                    all_triggered_semantic_reviews_applied=all(p["semantic_status"] == "applied" for p in boundary),
                    all_memory_selections_completed=all(
                        p["selection_status"] in {"empty", "selected"} for p in prepared_diagnostics
                    ),
                    all_native_policies_applied=all(p["policy_status"] == "applied" for p in prepared_diagnostics),
                    current_explicit_advice_boundaries_preserved=all(
                        compiled_boundary(p, explicit=True) for p in boundary[:2]
                    ),
                    effective_advice_boundaries_preserved=all(compiled_boundary(p) for p in boundary),
                    complete_dynamic_context_reaches_model=all(
                        any(
                            p["dynamic_context"] in m.get("content", "")
                            for c in cloud_calls
                            for m in c["request"].get("messages", [])
                        )
                        for p in boundary
                    ),
                    boundary_strategies_exclude_advice_and_probes=all(
                        not set(p["decision_strategies"]) & {"offer_suggestion", "gentle_probe", "clarify_need"}
                        for p in boundary
                    ),
                    boundary_answers_do_not_question_user=all(
                        "？" not in g["response"].get("reply", "") and "?" not in g["response"].get("reply", "")
                        for g in proof["generation"][3:6]
                    ),
                )
            proof["native_optional_reviewers_enabled"] = args.reviewers
            proof["semantic_review_timeout_seconds"] = args.semantic_review_timeout
            (OUT / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
            assert all(proof["checks"].values()), proof["checks"]
            print(
                json.dumps(
                    {
                        "result": str(OUT / "result.json"),
                        "generation_statuses": [x["http_status"] for x in proof["generation"]],
                        "cloud_calls": proof["cloud_calls"],
                    },
                    ensure_ascii=False,
                )
            )
    await shutdown_memory_enrichment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewers", action="store_true")
    parser.add_argument("--semantic-review-timeout", type=float, default=5.0)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    asyncio.run(main(parser.parse_args()))
