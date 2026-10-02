"""Whole long-source budget deferral and final confirmed conditions."""

import asyncio
import hashlib
import json
import secrets
from datetime import datetime, timedelta


async def run_deferred_source_probe(client, args, proof, fixture, cloud_calls, *, database, output):
    from api.auth import _hash_password
    from character.memory_llm import get_memory_enrichment_scheduler
    from character.models import UserScope
    from character.source_memory import SourceMemoryService
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    password = secrets.token_urlsafe(24)
    user = await asyncio.to_thread(
        database.add_user, args.run_label, await asyncio.to_thread(_hash_password, password), False
    )
    identity = str(user["id"])
    client.cookies.clear()
    login = await client.post("/api/auth/login", json={"username": args.run_label, "password": password})
    me = await client.get("/api/auth/me")
    proof["fixture_auth_statuses"] = [login.status_code, me.status_code]
    assert proof["fixture_auth_statuses"] == [200, 200] and str(me.json()["user"]["id"]) == identity
    scope = dict(
        character_id="tsukiyashiro_kisaki",
        platform="web",
        adapter="web-character",
        sender_id=identity,
        conversation_type="private",
        conversation_id=identity,
    )
    us = UserScope("web", "web-character", identity, identity, "private")
    proof["fixture_scope"] = scope
    target = fixture["target"]
    assert (
        await asyncio.to_thread(
            database.capture_memory_source,
            **scope,
            source_message_id=target["id"],
            body=target["body"],
            observed_at=datetime.fromisoformat(target["observed_at"]),
        )
        == "recorded"
    )
    for index in range(fixture["noise"]["count"]):
        assert (
            await asyncio.to_thread(
                database.capture_memory_source,
                **scope,
                source_message_id="inventory-" + str(index),
                body=fixture["noise"]["body_template"].format(index=index),
                observed_at=datetime.fromisoformat(fixture["noise"]["observed_at"]) + timedelta(seconds=index),
            )
            == "recorded"
        )
    repo = DatabaseCharacterMemoryRepository(database)

    async def snapshot():
        sources = await repo.list_sources(scope["character_id"], us, limit=100)
        claims = await repo.list_memory_records(scope["character_id"], us, limit=None, include_inactive=True)
        return dict(
            sources=sources,
            claims=claims,
            sha256=hashlib.sha256(json.dumps([sources, claims], sort_keys=True, default=str).encode()).hexdigest(),
        )

    initial = await snapshot()
    assert len(initial["sources"]) == 1 and initial["claims"] == []
    service = SourceMemoryService(repo, max_chars=16384, defer_budget=True)
    old = await service.recall(scope["character_id"], us, fixture["question"])
    fixed = await service.recall(
        scope["character_id"], us, fixture["question"], retrieval_context=fixture["history"][0]["content"]
    )
    from inference.provider_context import get_provider_context_budget

    budget = get_provider_context_budget()
    proof["pg_source_checks"] = {
        "same_one_full_original_source": len(initial["sources"]) == 1,
        "zero_claims_before_generation": initial["claims"] == [],
        "native_cloud_budget_and_deferral": budget.window_tokens == 65536
        and budget.source_max_chars == 16384
        and budget.defer_source_budget,
        "old_producer_omits_whole_original": old.diagnostics["status"] == "budget_omitted" and not old.context,
        "fresh_full_candidate_identity": any(
            row["source_id"] == target["id"] for row in json.loads(fixed.candidate_context)["records"]
        ),
        "fresh_full_candidate_body": any(
            row["text"] == target["body"] for row in json.loads(fixed.candidate_context)["records"]
        ),
        "same_original_storage_after_recall": initial == await snapshot(),
        "contextual_lane_scoped": fixed.diagnostics["contextual_search_enabled"]
        and fixed.diagnostics["contextual_read_count"] == 1,
    }
    assert all(proof["pg_source_checks"].values())
    history = fixture["history"]
    history_id = "source-topic-history"
    stamp = datetime.now().astimezone()
    assert (
        await asyncio.to_thread(
            database.capture_memory_source,
            **scope,
            source_message_id=history_id,
            body=history[0]["content"],
            observed_at=stamp,
        )
        == "recorded"
    )
    await asyncio.to_thread(
        database.add_message,
        dict(
            platform="web",
            adapter="web-character",
            senderId=identity,
            userId=identity,
            conversationType="private",
            sessionType="private",
            conversationId=args.run_label,
            sessionId=args.run_label,
            characterId=scope["character_id"],
            loraName="default",
            sourceMessageId=history_id,
            message=history[0]["content"],
            reply="",
            modelName="synthetic-fixture-history",
        ),
    )
    proof["server_history_seeded_complete"] = True
    before = await snapshot()
    before_calls = len(cloud_calls)
    response = await client.post(
        "/api/generate",
        json={
            "message": fixture["question"],
            "characterId": scope["character_id"],
            "loraId": "default",
            "sessionId": args.run_label,
            "sessionType": "private",
        },
    )
    assert await get_memory_enrichment_scheduler().flush_memory(timeout=45)
    proof["generation"].append(
        dict(
            id=fixture["cases"][0]["id"],
            http_status=response.status_code,
            response=response.json(),
            cloud_call_range=[before_calls, len(cloud_calls)],
        )
    )
    after = await snapshot()
    original_ids = {r["source_message_id"] for r in before["sources"]}
    proof["original_source_rows_unchanged"] = before["sources"] == [
        r for r in after["sources"] if r["source_message_id"] in original_ids
    ]
    proof["claims_before_generation"] = before["claims"]
    proof["claims_after_generation"] = after["claims"]
    proof["sync_pending_final"] = len(database._pending)
    from evaluation.deferred_source_budget_audit import audit_deferred_source

    proof["checks"] = audit_deferred_source(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values())
