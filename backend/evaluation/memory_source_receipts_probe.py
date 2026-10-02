"""Full scoped legacy claim receipt fanout, then one native current-major question."""

import asyncio
import hashlib
import json
import secrets
from dataclasses import asdict
from datetime import datetime, timedelta


async def run_receipt_probe(client, args, proof, fixture, cloud_calls, *, database, output, source_proof):
    import asyncpg

    from api.auth import _hash_password
    from character.memory_llm import get_memory_enrichment_scheduler
    from character.memory_service import CharacterMemoryService
    from character.models import UserScope
    from db import memory_source
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
        memory_key=target["memory_key"],
        content="本人专业：" + fixture["expected_major"],
        source_message_id=target["id"],
        source_message_ids_json=json.dumps([target["id"]]),
        evidence_json=json.dumps([target["body"]], ensure_ascii=False),
        observed_at=stamp.isoformat(),
    )
    ids = []
    bodies = []
    stamp = datetime.fromisoformat(fixture["inventory"]["observed_at"])
    for index in range(fixture["inventory"]["count"]):
        sid = fixture["inventory"]["id_template"].format(index=index)
        body = fixture["inventory"]["body_template"].format(index=index)
        assert (
            await asyncio.to_thread(
                database.capture_memory_source,
                **scope,
                source_message_id=sid,
                body=body,
                observed_at=stamp + timedelta(microseconds=index),
            )
            == "recorded"
        )
        ids.append(sid)
        bodies.append(body)
    metadata = {
        "temporal_provenance": {"version": 1, "producer": "semantic_memory", "validity_authority": "unspecified"}
    }
    i = await asyncio.to_thread(
        database.append_character_memory_claim,
        **scope,
        memory_type="shared_event",
        memory_key="synthetic_inventory",
        content="完整合成库存盘点合集，非本人专业资料。",
        source_message_id=ids[0],
        source_message_ids_json=json.dumps(ids),
        evidence_json=json.dumps(bodies, ensure_ascii=False),
        metadata_json=json.dumps(metadata),
        observed_at=stamp.isoformat(),
    )
    proof["claim_ids"] = {"target": t["id"], "inventory": i["id"]}
    database_name = database.execute_sql("SELECT current_database() AS name", {})[0]["name"]
    c = await asyncpg.connect(user="boot", database=database_name, host=str(output.parent / "socket"), port=25433)
    try:
        assert await c.fetchval("SHOW data_directory") == str(output.parent / "data")
        owner = memory_source.owner_scope(**scope)
        source_scope = memory_source.source_scope(**scope)
        async with c.transaction():
            await c.fetchval(
                "SELECT revoked_before FROM memory_source_fences WHERE owner_key=$1 FOR UPDATE", owner["owner_key"]
            )
            for sid in ids[1:]:
                key = memory_source.source_identity(source_scope, sid)["source_key"]
                assert await c.fetchval("SELECT state FROM memory_sources WHERE source_key=$1", key) == "recorded"
                await c.execute(
                    "INSERT INTO memory_source_links(memory_id,source_key) VALUES($1,$2) ON CONFLICT DO NOTHING",
                    i["id"],
                    key,
                )

        async def snapshot():
            return {
                table: hashlib.sha256(
                    json.dumps([dict(row) for row in await c.fetch(sql, *params)], sort_keys=True, default=str).encode()
                ).hexdigest()
                for table, sql, params in [
                    (
                        "claims",
                        "SELECT * FROM character_memories WHERE id=ANY($1::bigint[]) ORDER BY id",
                        ([t["id"], i["id"]],),
                    ),
                    (
                        "sources",
                        "SELECT * FROM memory_sources WHERE scope_key=$1 AND source_message_id=ANY($2::text[]) ORDER BY source_key",
                        (source_scope["scope_key"], [target["id"], *ids]),
                    ),
                    (
                        "links",
                        "SELECT * FROM memory_source_links WHERE memory_id=ANY($1::bigint[]) ORDER BY memory_id,source_key",
                        ([t["id"], i["id"]],),
                    ),
                ]
            }

        before = await snapshot()
        pairs = ((t["id"], target["id"]), *((i["id"], sid) for sid in ids))
        oldlinks = await asyncio.to_thread(database.linked_memory_sources, **scope, memory_ids=(t["id"], i["id"]))
        exact = await asyncio.to_thread(database.linked_memory_source_receipts, **scope, claim_sources=pairs)
        repo = DatabaseCharacterMemoryRepository(database)

        class OldReader(DatabaseCharacterMemoryRepository):
            linked_source_receipts = None

        olditems, oldcount, oldtrace = await CharacterMemoryService(
            OldReader(database), semantic_enabled=False
        ).recall_with_diagnostics(
            scope["character_id"],
            us,
            fixture["question"],
            for_contextual_selection=True,
            reference_time=datetime.now().astimezone(),
        )
        items, count, trace = await CharacterMemoryService(repo, semantic_enabled=False).recall_with_diagnostics(
            scope["character_id"],
            us,
            fixture["question"],
            for_contextual_selection=True,
            reference_time=datetime.now().astimezone(),
        )
        target_item = next(item for item in items if item.memory_id == str(t["id"]))
        proof["pg_receipt_checks"] = {
            "old_topk_reproduces_100": len(oldlinks) == 100
            and not any(row["source_message_id"] == target["id"] for row in oldlinks),
            "exact_request_returns_101": len(exact) == len(pairs) == 101,
            "old_projection_unverified": oldtrace["field_presence"]["major"] is None,
            "current_major_proven": trace["field_presence"]["major"] is True
            and target_item.memory_key == "user_major"
            and target_item.temporal_mode != "observation",
            "original_body_preserved": target["body"] in target_item.evidence,
            "exact_sql_authority_path": trace["source_receipt_pairs_requested"]
            == trace["source_receipt_pairs_returned"]
            == 101
            and trace["source_authority_reader"] == "exact_claim_source_pairs",
            "old_snapshot_unchanged_after_read": before == await snapshot(),
        }
        assert all(proof["pg_receipt_checks"].values())
        proof["receipt_target_item"] = asdict(target_item)
        proof["receipt_recall_diagnostics"] = trace
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
            {
                "id": fixture["cases"][0]["id"],
                "http_status": response.status_code,
                "response": response.json(),
                "cloud_call_range": [before_calls, len(cloud_calls)],
            }
        )
        proof["seeded_rows_unchanged_after_generation"] = before == await snapshot()
        proof["sync_pending_final"] = len(database._pending)
    finally:
        await c.close()
    from evaluation.memory_source_receipts_audit import audit_receipts

    proof["checks"] = audit_receipts(proof, fixture, cloud_calls)
    (output / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str))
    (output / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
    print(json.dumps({"checks": proof["checks"], "calls": len(cloud_calls)}, ensure_ascii=False))
    if args.require_success:
        assert all(proof["checks"].values())
