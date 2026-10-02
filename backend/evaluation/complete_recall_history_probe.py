"""Old project constraints recovered before contextual selection of a long-history ellipsis."""

import asyncio
import hashlib
import json
import secrets
from datetime import datetime, timedelta


async def run_history_probe(client, args, proof, fixture, cloud_calls, *, database, output):
    from api.auth import _hash_password
    from character.memory_llm import get_memory_enrichment_scheduler
    from character.memory_service import CharacterMemoryService
    from character.models import UserScope
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from services.character_context import compile_user_recall_context

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
    stamp = datetime.fromisoformat(target["observed_at"])
    assert (
        await asyncio.to_thread(
            database.capture_memory_source,
            **scope,
            source_message_id=target["id"],
            body=target["body"],
            observed_at=stamp,
        )
        == "recorded"
    )
    t = await asyncio.to_thread(
        database.append_character_memory_claim,
        **scope,
        memory_type="user_fact",
        memory_key=target["key"],
        content=target["body"],
        source_message_id=target["id"],
        source_message_ids_json=json.dumps([target["id"]]),
        evidence_json=json.dumps([target["body"]], ensure_ascii=False),
        observed_at=stamp.isoformat(),
    )
    proof["target_claim_id"] = t["id"]
    for index in range(fixture["noise"]["count"]):
        body = fixture["noise"]["body_template"].format(index=index)
        sid = "noise-" + str(index)
        stamp = datetime.fromisoformat(fixture["noise"]["observed_at"]) + timedelta(seconds=index)
        assert (
            await asyncio.to_thread(
                database.capture_memory_source, **scope, source_message_id=sid, body=body, observed_at=stamp
            )
            == "recorded"
        )
        await asyncio.to_thread(
            database.append_character_memory_claim,
            **scope,
            memory_type="user_fact",
            memory_key="inventory_" + str(index),
            content=body,
            source_message_id=sid,
            source_message_ids_json=json.dumps([sid]),
            evidence_json=json.dumps([body]),
            observed_at=stamp.isoformat(),
        )
    repo = DatabaseCharacterMemoryRepository(database)

    async def snapshot():
        records = await repo.list_memory_records(scope["character_id"], us, limit=None, include_inactive=True)
        return hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()

    before = await snapshot()
    service = CharacterMemoryService(repo, semantic_enabled=False)
    history = fixture["history"]
    clipped = "\n".join(row["content"][-600:] for row in history[-6:] if row["role"] == "user")[-1200:]
    full = compile_user_recall_context(history)
    old, oldcount, _ = await service.recall_with_diagnostics(
        scope["character_id"],
        us,
        fixture["question"],
        for_contextual_selection=True,
        retrieval_context=clipped,
        reference_time=datetime.now().astimezone(),
    )
    current, count, trace = await service.recall_with_diagnostics(
        scope["character_id"],
        us,
        fixture["question"],
        for_contextual_selection=True,
        retrieval_context=full,
        reference_time=datetime.now().astimezone(),
    )
    proof["pg_recall_checks"] = {
        "actual_same37_full_records": oldcount == count == 37,
        "old_suffix_loses_available_topic": "榆桐" not in clipped and fixture["topic"] in history[0]["content"],
        "old_topic_candidate_missing": str(t["id"]) not in [item.memory_id for item in old],
        "same24_candidate_budget": len(old) == len(current) == 24,
        "full_history_recovers_target": str(t["id"]) in [item.memory_id for item in current],
        "full_history_exact": full == history[0]["content"],
        "no_memory_rewrite_for_recall": before == await snapshot(),
    }
    assert all(proof["pg_recall_checks"].values())
    proof["offline_candidate_ids"] = [item.memory_id for item in current]
    proof["offline_old_candidate_ids"] = [item.memory_id for item in old]
    # Stateful web history is server-owned; a browser history payload is ignored.
    history_source_id = "topic-history-source"
    history_stamp = datetime.now().astimezone()
    assert (
        await asyncio.to_thread(
            database.capture_memory_source,
            **scope,
            source_message_id=history_source_id,
            body=history[0]["content"],
            observed_at=history_stamp,
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
            sourceMessageId=history_source_id,
            message=history[0]["content"],
            reply="",
            modelName="synthetic-fixture-history",
        ),
    )
    proof["server_history_seeded_complete"] = True
    before_calls = len(cloud_calls)
    response = await client.post(
        "/api/generate",
        json={
            "message": fixture["question"],
            "history": history,
            "characterId": scope["character_id"],
            "loraId": "default",
            "sessionId": args.run_label,
            "sessionType": "private",
        },
    )
    assert await get_memory_enrichment_scheduler().flush_memory(timeout=45)
    proof["generation"].append(
        {
            "id": fixture["cases"][0]["id"],
            "http_status": response.status_code,
            "response": response.json(),
            "cloud_call_range": [before_calls, len(cloud_calls)],
        }
    )
    proof["seeded_claims_unchanged_after_generation"] = before == await snapshot()
    proof["sync_pending_final"] = len(database._pending)
    from evaluation.complete_recall_history_audit import audit_history

    proof["checks"] = audit_history(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values())
