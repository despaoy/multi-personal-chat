"""Native complete-source evidence, narrative citation and follow-up cache probe."""

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


def build_native_fixture_index(root):
    import numpy as np

    from knowledge.multiscale_rag.constants import EMBEDDING_TEXT_VERSION, INDEX_FORMAT_VERSION
    from knowledge.multiscale_rag.index_builder import _card_vector_text, _evidence_vector_text
    from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider
    from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference

    fixture_root = Path(__file__).resolve().parents[1] / "tests/fixtures"
    fixture = json.loads((fixture_root / "deepseek_grounded_evidence_cases.json").read_text())
    from knowledge.retrieval_core.registry import get_default_registry

    config = get_default_registry().require("tsukiyashiro_kisaki")
    cards = []
    evidence = []
    source_texts = {}
    for item in fixture["sources"]:
        path = fixture_root / "grounded_sources" / item["source_file"]
        text = path.read_text().strip()
        source_texts[item["id"]] = text
        ref = SourceReference(
            str(path.relative_to(Path(__file__).resolve().parents[2])), 1, len(text.splitlines()), item["id"]
        )
        meta = dict(
            scale="card",
            subject=config.canonical_entity(item["subject"]) or item["subject"],
            predicate=item["predicate"],
            value=item["value"],
            story_unit_id="stage8-permit-control",
            volume_number=1,
            story_title="完整许可记录（合成测试）",
            embedding_text_version=EMBEDDING_TEXT_VERSION,
        )
        card = KnowledgeIndexDocument(
            id=item["id"],
            domain_id="tsukiyashiro_kisaki",
            document_type="fact",
            title=item["title"],
            summary=item["summary"],
            content=item["summary"] + "\n证据：" + text,
            embedding_text="",
            keywords=["天文观测室", "许可", "批准", "实际进展"],
            entities=[config.canonical_entity(e) or e for e in item["entities"]],
            source=ref,
            metadata=meta,
            reality_status="objective",
            temporal_scope="current",
            content_scope="main_story",
            index_version=INDEX_FORMAT_VERSION,
        )
        card.embedding_text = _card_vector_text(card)
        child = KnowledgeIndexDocument(
            id=item["id"] + "-evidence",
            domain_id=card.domain_id,
            document_type="evidence",
            title=item["title"] + "原文",
            summary=text[:80],
            content=text,
            embedding_text="",
            keywords=list(card.keywords),
            entities=list(card.entities),
            source=ref,
            metadata={**meta, "scale": "evidence", "parent_id": card.id},
            reality_status=card.reality_status,
            temporal_scope=card.temporal_scope,
            content_scope=card.content_scope,
            index_version=INDEX_FORMAT_VERSION,
        )
        child.embedding_text = _evidence_vector_text(child, card)
        cards.append(card)
        evidence.append(child)
    provider = LocalMeanPoolingEmbeddingProvider(model_path=os.environ["EMBEDDING_MODEL_PATH"])
    for name, docs in [("card_index", cards), ("scene_story_index", []), ("evidence_index", evidence)]:
        path = root / name
        path.mkdir(parents=True, exist_ok=False)
        (path / "documents.jsonl").write_text("".join(d.to_jsonl() + "\n" for d in docs))
        np.save(path / "vectors.npy", provider.embed_texts([d.embedding_text for d in docs]), allow_pickle=False)
        (path / "manifest.json").write_text(
            json.dumps(
                dict(
                    index_version=INDEX_FORMAT_VERSION,
                    embedding_text_version=EMBEDDING_TEXT_VERSION,
                    document_count=len(docs),
                    embedding_dim=384,
                )
            )
        )
    return fixture, source_texts


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
        CHARACTER_RAG_INDEX_ROOT=str(OUT / "native-index"),
        CORRECTIVE_RAG_ENABLED="false",
        RERANKER_ENABLED="false",
        CHAT_CONVERSATION_BURST="10",
        CHAT_SENDER_BURST="10",
    )

    fixture, source_texts = build_native_fixture_index(OUT / "native-index")

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
    cases = fixture["cases"]
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
            for case in cases:
                before = len(cloud_calls)
                response = await client.post(
                    "/api/ask",
                    json=dict(
                        message=case["message"],
                        history=case["history"],
                        domainId="tsukiyashiro_kisaki",
                        topK=3,
                        temperature=0.23,
                        maxTokens=1024,
                        topP=0.77,
                    ),
                )
                proof["cases"].append(
                    dict(
                        case=case,
                        http_status=response.status_code,
                        response=response.json(),
                        cloud_call_range=[before, len(cloud_calls)],
                    )
                )
                (OUT / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
            response = await client.post(
                "/api/ask/stream",
                json=dict(
                    message=cases[0]["message"],
                    domainId="tsukiyashiro_kisaki",
                    topK=3,
                    temperature=0.24,
                    maxTokens=1024,
                    topP=0.78,
                ),
            )
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            proof["stream"] = dict(http_status=response.status_code, events=events)
        done = next((e for e in events if e.get("type") == "done"), {})
        stream_reply = "".join(e.get("text", e.get("delta", "")) for e in events if e.get("type") == "delta")
        qualifier = fixture["late_qualifier"]
        kisaki = proof["cases"][2]
        ruri = proof["cases"][3]

        def answer_for(case):
            return case["response"].get("answer", "")

        def negative(answer):
            return bool(re.search(r"未.{0,6}(获批|批准|许可)|尚未|没有.{0,8}(获批|批准|许可)", answer))

        def selected(case):
            return cloud_calls[slice(*case["cloud_call_range"])]

        def content(call):
            from html import unescape

            return unescape(call["request"]["messages"][-1]["content"])

        citations = [
            c
            for case in proof["cases"]
            for c in case["response"].get("citations", [])
            if c.get("document_id") == "stage8-kisaki-permit"
        ]
        proof["checks"] = dict(
            native_answers_http_200=all(c["http_status"] == 200 for c in proof["cases"]),
            native_answers_grounded=all(not c["response"].get("abstained", True) for c in proof["cases"]),
            source_input_is_complete=qualifier in source_texts["stage8-kisaki-permit"],
            actual_wire_keeps_late_source_qualifier=bool(cloud_calls)
            and all(qualifier in content(c) for c in cloud_calls),
            first_answer_keeps_real_status=negative(answer_for(proof["cases"][0])),
            mixed_answer_keeps_real_status=negative(answer_for(proof["cases"][1])),
            kisaki_followup_keeps_real_status=negative(answer_for(kisaki)),
            ruri_followup_uses_new_subject="琉璃" in answer_for(ruri)
            and bool(re.search(r"已.{0,6}(获批|批准|许可)|取得.{0,6}许可", answer_for(ruri))),
            identical_question_different_history_not_cached=bool(selected(kisaki)) and bool(selected(ruri)),
            actual_wire_keeps_each_followup_history=all(
                any(
                    m.get("role") == "user" and m.get("content") == case["case"]["history"][0]["content"]
                    for m in selected(case)[0]["request"]["messages"]
                )
                for case in [kisaki, ruri]
                if selected(case)
            )
            and bool(selected(kisaki))
            and bool(selected(ruri)),
            public_citations_keep_narrative_scope=bool(citations)
            and all(
                c.get("reality_status") == "objective"
                and c.get("temporal_scope") == "current"
                and c.get("content_scope") == "main_story"
                and c.get("story_unit_id") == "stage8-permit-control"
                for c in citations
            ),
            public_citations_keep_complete_source_evidence=bool(citations)
            and all(qualifier in c.get("evidence_excerpt", "") for c in citations),
            all_selected_cloud_calls_completed=len(cloud_calls) == 5
            and all(
                c["http_status"] == 200
                and c["response"].get("model") == "deepseek-v4-pro"
                and all(x.get("finish_reason") == "stop" for x in c["response"].get("choices", []))
                for c in cloud_calls
            ),
            no_wrong_local_completions=not local_review_calls,
            stream_completed=response.status_code == 200
            and bool(done)
            and not any(e.get("type") == "error" for e in events),
            stream_keeps_real_status=negative(stream_reply),
        )
        proof["complete_sources"] = source_texts
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
