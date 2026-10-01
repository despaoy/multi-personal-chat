"""Actual authenticated PostgreSQL chat over complete legacy field sources."""

import argparse
import asyncio
import base64
import importlib.util
import json
import os
import re
import secrets
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path


async def main(args):
    from evaluation.conversation_source_probe import verify_cluster
    from evaluation.native_provider_chain_probe import isolated_probe_paths

    root, out, key_path = isolated_probe_paths(args.root, args.run_label, args.api_key_file)
    bootstrap = "postgresql+asyncpg://boot@/postgres?host=" + str(root / "socket") + "&port=25433"
    await verify_cluster(bootstrap, root / "data")
    out.mkdir(mode=0o700, exist_ok=False)
    database = "stage3_" + args.run_label.replace("-", "_")
    import asyncpg

    connection = await asyncpg.connect(user="boot", database="postgres", host=str(root / "socket"), port=25433)
    try:
        assert await connection.fetchval("SHOW data_directory") == str(root / "data")
        await connection.execute("CREATE DATABASE " + database)
    finally:
        await connection.close()
    key = key_path.read_text().strip()
    os.environ.update(
        DATABASE_URL="postgresql+asyncpg://boot@/" + database + "?host=" + str(root / "socket") + "&port=25433",
        ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        USE_POSTGRESQL="true",
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        ASTRBOT_INTEGRATION_TOKEN=secrets.token_urlsafe(48),
        MULTIPERSONAL_BACKEND_URL="https://legacy-evaluation.invalid",
        ALLOWED_ORIGINS="https://legacy-evaluation.invalid",
        ALLOW_PUBLIC_REGISTRATION="false",
        SECURITY_MIDDLEWARE_ENABLED="true",
        LOG_LEVEL="INFO",
        BACKEND_WORKERS="1",
        AUDIT_LOG_DIR=str(out / "audit"),
        BACKUP_DIR=str(out / "backups"),
        MODEL_PROVIDER="openai_compat",
        VLLM_ENABLED="false",
        VLLM_BASE_URL="http://127.0.0.1:1",
        VLLM_BASE_URLS="http://127.0.0.1:1",
        OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
        OPENAI_COMPAT_API_KEY=key,
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS="65536",
        MEMORY_LLM_ENABLED="false",
        MEMORY_SOURCE_RECALL_ENABLED="false",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED="false",
        CONTEXTUAL_MEMORY_SELECTION_ENABLED="false",
        CONTEXTUAL_DECISION_POLICY_ENABLED="false",
        REDIS_URL="redis://127.0.0.1:1/0",
        EMBEDDING_MODEL_PATH="/home/boot/lhm/multipersonal-runtime/models/paraphrase-multilingual-MiniLM-L12-v2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        CHAT_CONVERSATION_BURST="30",
        CHAT_SENDER_BURST="30",
    )
    await verify_cluster(os.environ["DATABASE_URL"], root / "data")
    if args.baseline_reader:
        for suffix in ("memory_query", "memory_service"):
            name = "character." + suffix
            path = Path(
                "/home/boot/lhm/multipersonal-runtime/backups/backend-chain-20261001/stage11/character_"
                + suffix
                + ".py"
            )
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    import httpx

    original_send = httpx.AsyncClient.send
    calls = []

    async def observed_send(client, request, **kwargs):
        if request.url.host != "api.deepseek.com":
            if request.url.path.endswith("/chat/completions"):
                raise AssertionError("Unexpected non-DeepSeek model endpoint")
            return await original_send(client, request, **kwargs)
        started = time.monotonic()
        response = await original_send(client, request, **kwargs)
        await response.aread()
        calls.append(
            dict(
                request=json.loads(request.content),
                http_status=response.status_code,
                response=response.json(),
                elapsed_seconds=time.monotonic() - started,
            )
        )
        (out / "cloud-calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2))
        return response

    httpx.AsyncClient.send = observed_send
    from app.main import create_app
    from character.models import MemoryItem, UserScope
    from db.adapter import db, is_pg_mode
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from services.character_context import CharacterContextService

    assert is_pg_mode()
    diagnostics = []
    underlying = []
    original_history = CharacterContextService._load_history
    original_prepare = CharacterContextService.prepare_turn

    async def cold_history(service, turn, scope, character_id):
        history = await original_history(service, turn, scope, character_id)
        underlying.append(len(history))
        return []

    async def observed_prepare(service, turn, character_id):
        prepared = await original_prepare(service, turn, character_id)
        diagnostics.append(
            dict(
                history_count=len(prepared.history),
                memory_budget=prepared.memory_budget,
                memory_recall=prepared.memory_recall,
                fields=dict(prepared.compiled.memory_field_presence),
                packets=[asdict(item) for item in prepared.compiled.memory_packets],
                used_ids=prepared.compiled.used_memory_ids,
                reference_context=prepared.compiled.reference_context,
                source_context=prepared.compiled.conversation_reference_context,
                episodic_context=prepared.compiled.episodic_reference_context,
            )
        )
        return prepared

    CharacterContextService._load_history = cold_history
    CharacterContextService.prepare_turn = observed_prepare
    db.update_config(dict(useKnowledgeBase=False, temperature=0.2, maxTokens=800, topP=0.9))
    if Path(args.fixture).name != args.fixture or not args.fixture.endswith("_cases.json"):
        raise ValueError("Use a named repository fixture, without path traversal")
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures" / args.fixture
    inputs = json.loads(fixture.read_text())
    if args.case_id:
        requested = set(args.case_id)
        available = {case["id"] for case in inputs["cases"]}
        if not requested <= available:
            raise ValueError("Unknown case IDs: " + ", ".join(sorted(requested - available)))
        inputs["cases"] = [case for case in inputs["cases"] if case["id"] in requested]
    if not inputs["cases"]:
        raise ValueError("At least one complete case is required")
    results = []
    repo = DatabaseCharacterMemoryRepository(db)
    app = create_app()
    password = secrets.token_urlsafe(24)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="https://legacy-evaluation.invalid",
            timeout=180,
        ) as client,
    ):
        register = await client.post("/api/auth/register", json=dict(username=args.run_label, password=password))
        assert register.status_code == 200
        client.cookies.clear()
        login = await client.post("/api/auth/login", json=dict(username=args.run_label, password=password))
        assert login.status_code == 200 and "access_token" in client.cookies
        me = await client.get("/api/auth/me")
        assert me.status_code == 200
        identity = str(me.json()["user"]["id"])
        scope = UserScope("web", "web-character", identity, identity, "private")
        for case in inputs["cases"]:
            db.clear_character_memories("tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity)
            at = datetime.now(timezone.utc)
            ids = {}
            for record in case["records"]:
                source = record["source"]
                await repo.capture_source(
                    "tsukiyashiro_kisaki",
                    scope,
                    source_message_id=case["id"] + "-" + record["id"],
                    body=source,
                    observed_at=at,
                )
                previous = ids.get(record.get("supersedes"))
                row = await repo.append_claim(
                    "tsukiyashiro_kisaki",
                    scope,
                    MemoryItem("", "user_fact", "用户明确提到：" + record["quote"], 0.9),
                    memory_key=record["key"],
                    evidence=(record["quote"],),
                    confidence=0.98,
                    source_message_id=case["id"] + "-" + record["id"],
                    relation_type="SUPERSEDE" if previous else "ADD",
                    supersedes_memory_id=previous,
                    valid_from=(at + timedelta(microseconds=len(ids))).isoformat(),
                    valid_to=(at + timedelta(microseconds=len(ids))).isoformat()
                    if record.get("proposed_zero_width")
                    else None,
                    observed_at=(at + timedelta(microseconds=len(ids))).isoformat()
                    if record.get("metadata", {}).get("temporal_provenance")
                    else None,
                    metadata=record.get("metadata"),
                )
                ids[record["id"]] = row["id"]
            status_response = await client.post(
                "/api/generate",
                json=dict(
                    message="你保存了我的姓名和居住地吗？",
                    characterId="tsukiyashiro_kisaki",
                    loraId="default",
                    sessionId=args.run_label + "-" + case["id"] + "-status",
                    sessionType="private",
                ),
            )
            assert status_response.status_code == 200
            before = await repo.list_memory_records("tsukiyashiro_kisaki", scope, include_inactive=True)
            start = len(calls)
            response = await client.post(
                "/api/generate",
                json=dict(
                    message=case["query"],
                    characterId="tsukiyashiro_kisaki",
                    loraId="default",
                    sessionId=args.run_label + "-" + case["id"],
                    sessionType="private",
                ),
            )
            after = await repo.list_memory_records("tsukiyashiro_kisaki", scope, include_inactive=True)
            results.append(
                dict(
                    id=case["id"],
                    storage_status_response=status_response.json(),
                    http_status=response.status_code,
                    response=response.json(),
                    cloud_call_start=start,
                    cloud_call_end=len(calls),
                    diagnostics=diagnostics[-1],
                    storage_unchanged=before == after,
                )
            )
            (out / "result.json").write_text(
                json.dumps(
                    dict(
                        baseline_reader=args.baseline_reader,
                        cases=results,
                        cloud_calls=len(calls),
                        underlying_history=underlying,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    httpx.AsyncClient.send = original_send
    assert all(result["http_status"] == 200 for result in results)
    assert all(result["storage_unchanged"] and result["diagnostics"]["history_count"] == 0 for result in results)
    checks = {
        "actual_provider_and_finish": bool(calls)
        and all(
            c["http_status"] == 200
            and c["response"]["model"] == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        "empty_history": all(c["diagnostics"]["history_count"] == 0 for c in results),
        "stored_rows_unchanged": all(c["storage_unchanged"] for c in results),
    }
    if not args.baseline_reader:
        for record, case in zip(results, inputs["cases"]):
            fields = record["diagnostics"]["fields"]
            expected = case["expected"]
            checks[case["id"] + "_expected_values"] = all(
                value in record["response"]["reply"]
                for field, value in expected.items()
                if field in {"name", "origin", "residence"} and value is not None
            )
            if expected["residence"] is None:
                source = case["records"][1]["source"]
                wire = "\n".join(
                    unescape(m["content"])
                    for c in calls[record["cloud_call_start"] : record["cloud_call_end"]]
                    for m in c["request"]["messages"]
                )
                checks[case["id"] + "_complete_source_in_model"] = source in wire
                checks[case["id"] + "_uncertainty"] = fields["residence"] is None and bool(
                    re.search("无法确认|不能确认|无法确定|不能确定", record["response"]["reply"])
                )
            else:
                checks[case["id"] + "_field_presence"] = all(fields[f] is True for f in ("name", "origin", "residence"))
    final = dict(
        baseline_reader=args.baseline_reader,
        cases=results,
        cloud_calls=len(calls),
        underlying_history=underlying,
        checks=checks,
    )
    (out / "result.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
    assert all(checks.values()), [name for name, ok in checks.items() if not ok]
    print(
        json.dumps(
            dict(
                cases=len(results),
                cloud_calls=len(calls),
                baseline_reader=args.baseline_reader,
                checks_passed=len(checks),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--baseline-reader", action="store_true")
    parser.add_argument("--fixture", default="deepseek_legacy_field_cases.json")
    parser.add_argument("--case-id", action="append", help="Run only a selected case; repeat for additional IDs")
    asyncio.run(main(parser.parse_args()))
