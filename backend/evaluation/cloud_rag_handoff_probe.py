"""Native shared API retrieval and generation over complete synthetic sources."""

import argparse
import asyncio
import base64
import json
import os
import re
import secrets
from dataclasses import asdict
from datetime import datetime
from html import unescape
from pathlib import Path


async def run(args):
    root = Path(args.root).resolve()
    fixture = json.loads((root / "fixture.json").read_text())
    key = Path(args.api_key_file).read_text().strip()
    assert key and fixture["synthetic"]
    os.environ.update(
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        DATABASE_PATH=str(root / "native.sqlite"),
        DATABASE_URL="",
        USE_POSTGRESQL="false",
        MODEL_PROVIDER="openai_compat",
        OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS="65536",
        OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
        OPENAI_COMPAT_API_KEY=key,
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        MEMORY_SOURCE_RECALL_ENABLED="true",
        MEMORY_LLM_ENABLED="false",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED="true",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS="30",
        CONTEXTUAL_MEMORY_SELECTION_ENABLED="true",
        CONTEXTUAL_DECISION_POLICY_ENABLED="true",
        REDIS_URL="redis://127.0.0.1:1/0",
        AUDIT_LOG_DIR=str(root / "native-audit"),
        BACKUP_DIR=str(root / "native-backups"),
        CHARACTER_RAG_INDEX_ROOT=str(Path(args.index_root).resolve())
        if args.index_root
        else str(root / "synthetic-index"),
        EMBEDDING_MODEL_PATH="/home/boot/lhm/multipersonal-runtime/models/paraphrase-multilingual-MiniLM-L12-v2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        INTENT_MODEL_PATH=str(root / "no-intent-model"),
        CHARACTER_INDEPENDENT_TASKS_ENABLED="false",
    )
    import httpx

    from api import generate as api
    from character.memory_query import lookup_fields, storage_fields
    from db.database import SQLiteDB
    from db.schemas import MessageRequest
    from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
    from inference.model_manager import get_model_manager
    from services.character_context import TurnInput, build_character_context_service

    question = fixture["question"]
    assert not (lookup_fields(question) or storage_fields(question)), (
        "Fixture accidentally routes as a private-only field lookup"
    )
    calls = []
    original_send = httpx.AsyncClient.send

    async def observed_send(client, request, **kwargs):
        if request.url.host != "api.deepseek.com":
            return await original_send(client, request, **kwargs)
        payload = json.loads(request.content)
        assert payload.get("model") == "deepseek-v4-pro", "Isolated DB model overrides environment"
        response = await original_send(client, request, **kwargs)
        await response.aread()
        calls.append(dict(request=payload, http_status=response.status_code, response=response.json()))
        (root / "native-cloud-calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2) + "\n")
        return response

    httpx.AsyncClient.send = observed_send
    db = SQLiteDB(root / "native.sqlite")
    db.update_config(
        {
            "modelProvider": "openai_compat",
            "openaiCompatModel": "deepseek-v4-pro",
            "openaiCompatBaseUrl": "https://api.deepseek.com",
        }
    )
    from cache.config_cache import invalidate_config_cache

    invalidate_config_cache()
    fields = dict(
        character_id="tsukiyashiro_kisaki",
        platform="web",
        adapter="stage58-native",
        sender_id="wenxi-native",
        conversation_type="private",
        conversation_id="wenxi-native",
    )
    assert (
        db.capture_memory_source(
            **fields,
            source_message_id=fixture["private"]["id"],
            body=fixture["private"]["body"],
            observed_at=datetime.fromisoformat(fixture["private"]["observed_at"]),
        )
        == "recorded"
    )
    initial = db.list_memory_sources(**fields, limit=100)
    assert len(initial) == 1 and db.list_character_memory_claims(**fields, include_inactive=True) == []
    prepared = await build_character_context_service(db, source_recall_enabled=True).prepare_turn(
        TurnInput(
            question,
            "web",
            "stage58-native",
            "wenxi-native",
            "wenxi-native",
            "private",
            received_at=datetime.now().astimezone(),
        ),
        fields["character_id"],
    )
    proof = {
        "synthetic_only": True,
        "scope": fields,
        "question": question,
        "prepared": {
            "semantic_status": prepared.semantic_review_status,
            "policy_status": prepared.contextual_policy_status,
            "source_status": prepared.compiled.memory_source_status,
            "source_candidate": prepared.compiled.source_candidate_context,
            "recall": prepared.memory_recall,
        },
        "retrieval": [],
        "generation": [],
    }
    original_retrieve = api._retrieve_rag_bundle

    async def observed_retrieve(*pos, **kwargs):
        bundle = await original_retrieve(*pos, **kwargs)
        proof["retrieval"].append(bundle)
        return bundle

    api._retrieve_rag_bundle = observed_retrieve
    original_generation = api.generate_character_response

    async def observed_generation(request, generate):
        proof["requested_budget"] = {
            "window": request.context_window_tokens,
            "evidence_max_chars": request.evidence_max_chars,
        }
        result = await original_generation(request, generate)
        proof["generation"].append(
            {
                "messages": list(result.plan.messages),
                "retrieval": asdict(result.plan.retrieval),
                "context": asdict(result.plan.character_context),
                "raw_reply": result.reply,
                "response_mode": result.response_mode,
            }
        )
        return result

    api.generate_character_response = observed_generation
    manager = get_model_manager()

    async def model_generate(*, messages, temperature, max_tokens, top_p=None, **kwargs):
        if max_tokens == 1024:
            wire = unescape(messages[-1]["content"])
            assert fixture["public"]["body"] in wire, "Complete public packet missing before actual model call"
            match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
            packet = json.loads(match[1]) if match else {}
            assert any(
                row["source_id"] == fixture["private"]["id"] and row["text"] == fixture["private"]["body"]
                for row in packet.get("records", [])
            ), "Complete scoped private speech missing before actual model call"
            assert (
                sum(estimated_tokens(m["content"]) + 4 for m in messages) + 1024 + CONTEXT_SAFETY_MARGIN_TOKENS <= 65536
            )
        reply, _ = await manager.async_generate(
            prompt=messages[-1]["content"], session_history=messages[:-1], rag_docs=None, max_tokens_override=max_tokens
        )
        return reply

    reply, rag_used, meta = await api._generate_with_retrieval(
        MessageRequest(
            message=question,
            characterId=fields["character_id"],
            platform="web",
            adapter="stage58-native",
            senderId="wenxi-native",
            conversationId="wenxi-native",
            conversationType="private",
            sessionId="stage58-native",
        ),
        "default",
        runtime_config={"maxTokens": 1024, "temperature": 0.2, "topP": 0.9, "useKnowledgeBase": True},
        prepared_character_turn=prepared,
        message_db=db,
        model_generate=model_generate,
    )
    proof.update(
        reply=reply,
        rag_used=rag_used,
        meta=meta,
        original_sources_unchanged=db.list_memory_sources(**fields, limit=100) == initial,
        claims_remain_empty=db.list_character_memory_claims(**fields, include_inactive=True) == [],
        cloud_calls=len(calls),
        primary_calls=sum(c["request"].get("max_tokens") == 1024 for c in calls),
    )
    (root / "native-result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "actual_cloud_calls": len(calls),
                "primary_calls": proof["primary_calls"],
                "result_saved": True,
                "rag_used": rag_used,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--index-root")
    asyncio.run(run(parser.parse_args()))
