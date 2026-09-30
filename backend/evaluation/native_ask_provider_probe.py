"""Native knowledge import, embedding, retrieval, memory and DeepSeek probe."""

import argparse
import asyncio
import base64
import json
import os
import re
import secrets
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

    ROOT, OUT, key_path = isolated_probe_paths(args.root, args.run_label, args.api_key_file)
    bootstrap_url = "postgresql+asyncpg://boot@/postgres?host=" + str(ROOT / "socket") + "&port=25433"
    await verify_cluster(bootstrap_url, ROOT / "data")
    database_name = "stage3_" + args.run_label.replace("-", "_")
    OUT.mkdir(exist_ok=False)
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
        VLLM_ENABLED="true",
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
        CHARACTER_RAG_INDEX_ROOT="/home/boot/lhm/multi-personal-chat/backend/data/knowledge/tsukiyashiro_kisaki/character_knowledge_index_v3",
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
    from character.memory_llm import shutdown_memory_enrichment
    from db.adapter import db

    db.update_config(dict(useKnowledgeBase=True, temperature=0.2, maxTokens=1024, topP=0.9))
    app = create_app()
    proof = dict(
        transport="authenticated_ASGI",
        provider="native_openai_compat",
        cases=[],
        native_index=os.environ["CHARACTER_RAG_INDEX_ROOT"],
    )
    queries = [
        "月社妃是哪本魔法之书的主人公？请依据知识库给出书名和引用。",
        "请查月社妃的身份资料，说明她对应的魔法之书，并附上来源标记。",
    ]
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="https://stage3-evaluation.invalid",
                timeout=180,
            ) as client,
        ):
            password = secrets.token_urlsafe(24)
            register = await client.post("/api/auth/register", json=dict(username=args.run_label, password=password))
            client.cookies.clear()
            login = await client.post("/api/auth/login", json=dict(username=args.run_label, password=password))
            assert register.status_code == login.status_code == 200
            for query in queries:
                before = len(cloud_calls)
                response = await client.post(
                    "/api/ask", json=dict(message=query, temperature=0.23, maxTokens=1024, topP=0.77)
                )
                proof["cases"].append(
                    dict(
                        query=query,
                        http_status=response.status_code,
                        response=response.json(),
                        cloud_call_range=[before, len(cloud_calls)],
                    )
                )
                (OUT / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
            response = await client.post(
                "/api/ask/stream", json=dict(message=queries[0], temperature=0.24, maxTokens=1024, topP=0.78)
            )
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            proof["stream"] = dict(http_status=response.status_code, events=events)
        done = next((e for e in events if e.get("type") == "done"), {})
        stream_reply = "".join(e.get("text", e.get("delta", "")) for e in events if e.get("type") == "delta")
        proof["checks"] = dict(
            native_nonstream_answers_success=all(
                c["http_status"] == 200
                and c["cloud_call_range"][1] > c["cloud_call_range"][0]
                and not c["response"].get("abstained")
                for c in proof["cases"]
            ),
            nonstream_answer_has_correct_book=all("缟玛瑙" in c["response"].get("answer", "") for c in proof["cases"]),
            nonstream_answer_has_bound_citations=all(c["response"].get("citations") for c in proof["cases"]),
            nonstream_model_id_matches_selection=all(
                c["response"].get("model") == "openai_compat/deepseek-v4-pro" for c in proof["cases"]
            ),
            no_wrong_local_completions=not local_review_calls,
            all_real_cloud_calls_completed=len(cloud_calls) == 3
            and all(
                c["http_status"] == 200
                and c["response"].get("model") == "deepseek-v4-pro"
                and all(x.get("finish_reason") == "stop" for x in c["response"].get("choices", []))
                for c in cloud_calls
            ),
            real_wire_preserves_roles=bool(cloud_calls)
            and all(
                c["request"]["messages"][0]["role"] == "system" and c["request"]["messages"][-1]["role"] == "user"
                for c in cloud_calls
            ),
            real_wire_preserves_top_p=bool(cloud_calls)
            and all(c["request"].get("top_p") in {0.77, 0.78} for c in cloud_calls),
            real_wire_preserves_temperature=bool(cloud_calls)
            and all(c["request"].get("temperature") in {0.23, 0.24} for c in cloud_calls),
            real_wire_contains_complete_book_evidence=bool(cloud_calls)
            and all("缟玛瑙" in c["request"]["messages"][-1]["content"] for c in cloud_calls),
            stream_finished=response.status_code == 200
            and bool(done)
            and not any(e.get("type") == "error" for e in events),
            stream_answer_has_correct_book="缟玛瑙" in stream_reply,
            stream_has_bound_citations=any(e.get("type") == "citations" and e.get("citations") for e in events),
        )
        proof["cloud_summary"] = [
            dict(
                model=c["response"].get("model"),
                status=c["http_status"],
                temperature=c["request"].get("temperature"),
                top_p=c["request"].get("top_p"),
            )
            for c in cloud_calls
        ]
        (OUT / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
        print(json.dumps(dict(checks=proof["checks"], cloud_calls=len(cloud_calls)), ensure_ascii=False))
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
