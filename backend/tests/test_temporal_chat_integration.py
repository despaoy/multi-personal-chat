"""Opt-in real Temporal dev-server acceptance; no cloud model calls."""

import asyncio
import copy
import json
import os
from functools import partial
from pathlib import Path
from time import time

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from character.library import CharacterSettings
from character.memory_llm import MemoryLlmConfig
from db.database import SQLiteDB
from repositories.character_chat import CharacterChatRepository
from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository
from services import temporal_chat
from services.character_chat_enrichment import process_enrichment
from services.character_chat_summary import enrich_summary
from workflows.chat_enrichment import ChatEnrichmentWorkflow


@pytest.mark.skipif(os.getenv("RUN_TEMPORAL_INTEGRATION") != "1", reason="requires local Temporal dev server")
def test_real_proactive_worker_commits_one_private_application_message(tmp_path, monkeypatch):
    from test_character_chat_push import device_keys

    from repositories.character_chat_proactive import CharacterChatProactiveRepository
    from repositories.character_chat_push import CharacterChatPushRepository
    from services.temporal_proactive import ChatProactiveActivities, WebPushConfig, dispatch_proactive
    from workflows.chat_proactive import ChatProactiveWorkflow

    db = SQLiteDB(tmp_path / "proactive.db")
    repo = CharacterChatProactiveRepository(db)
    character = repo.chats.characters.create("owner", CharacterSettings(name="测试", persona="谨慎", greeting="你好"))
    session = repo.chats.create("owner", character["id"], "主动消息")
    identity = session["id"]
    repo.set_policy("owner", identity, 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None, quiet_end=None)
    CharacterChatPushRepository(db).register(
        "owner", identity, "https://fcm.googleapis.com/fcm/send/private-subscription", *device_keys(),
    )
    due = time() - 1
    repo.schedule("owner", identity, 1, "job", "event", "PRIVATE-TOPIC", due, now=due - 1)
    idle = repo.chats.create("owner", character["id"], "久未联系")
    repo.chats.commit_turn(idle, "idle-start", "hi", "hi", "test")
    repo.set_policy("owner", idle["id"], 0, enabled=True, cadence_hours=24, timezone="UTC", quiet_start=None,
                    quiet_end=None, inactivity_hours=72, now=due - 72 * 3600)
    reminders = []

    class Sender:
        def send(self, endpoint, public_key, auth, tag):
            reminders.append(tag)
            return 201

    monkeypatch.setattr(WebPushConfig, "from_env", staticmethod(lambda: None))

    async def exercise():
        cache = Path(__file__).resolve().parents[2] / ".uv-cache" / "temporal"
        cache.mkdir(parents=True, exist_ok=True)
        async with await WorkflowEnvironment.start_local(download_dest_dir=str(cache), ui=False) as env:
            queue = "proactive-fixture"
            await dispatch_proactive(db, env.client, queue, 1)
            await dispatch_proactive(db, env.client, queue, 1)
            async with Worker(env.client, task_queue=queue, workflows=[ChatProactiveWorkflow],
                              activities=[ChatProactiveActivities(db).deliver]):
                handle = env.client.get_workflow_handle(f"character-chat-proactive-{queue}-1")
                assert await asyncio.wait_for(handle.result(), 30) == 2
                assert "PRIVATE-TOPIC" not in (await handle.fetch_history()).to_json()
                monkeypatch.setattr(WebPushConfig, "from_env", staticmethod(Sender))
                await dispatch_proactive(db, env.client, queue, 2)
                push_handle = env.client.get_workflow_handle(f"character-chat-proactive-{queue}-2")
                assert await asyncio.wait_for(push_handle.result(), 30) == 0
                push_history = (await push_handle.fetch_history()).to_json()
                assert "private-subscription" not in push_history and "PRIVATE-TOPIC" not in push_history
                assert len(reminders) == 1
            message, = repo.view("owner", identity)["jobs"]
            assert message["status"] == "sent" and "PRIVATE-TOPIC" in message["content"]
            assert repo.view("owner", idle["id"])["jobs"][0]["kind"] == "inactivity"
            assert repo.deliver_due() == 0

    try:
        asyncio.run(exercise())
    finally:
        db.close_connection()


@pytest.mark.skipif(os.getenv("RUN_TEMPORAL_INTEGRATION") != "1", reason="requires local Temporal dev server")
def test_real_worker_success_and_terminal_failure(tmp_path, monkeypatch):
    if os.getenv("RUN_CHARACTER_PG_INTEGRATION") == "1":
        from db.pg_database import PgDatabase, SyncPgAdapter

        db = SyncPgAdapter(PgDatabase(os.environ["DATABASE_URL"]))
    else:
        db = SQLiteDB(tmp_path / "chat.db")
    repo = CharacterChatRepository(db)
    character = repo.characters.create("1", CharacterSettings(name="测试", persona="谨慎", greeting="你好"))
    attempts = {}
    monkeypatch.setenv("SUMMARY_LLM_ENABLED", "true")

    class SummaryWriter:
        async def complete(self, messages):
            first = json.loads(messages[-1]["content"])["turns"][0]
            return json.dumps({"summary": "双方讨论未来计划。", "evidence": [
                {"turn_id": first["turn_id"], "role": "user", "quote": first["user"]}
            ]}, ensure_ascii=False)

    summary = partial(enrich_summary, config=MemoryLlmConfig(True, "https://test.invalid", "test"),
                      completion=SummaryWriter())

    async def memory(message, session, request_id):
        attempts[request_id] = attempts.get(request_id, 0) + 1
        if request_id == "fail" and attempts[request_id] <= 3:
            raise RuntimeError("PRIVATE-MODEL-CONTENT")
        state = copy.deepcopy(session["state"])
        state["memory_processing"] = "completed"
        return state

    async def relationship(message, reply, session, request_id):
        state = copy.deepcopy(session["state"])
        state["relationship_processing"] = "completed"
        return state

    async def controlled(turn_id, **kwargs):
        return await process_enrichment(turn_id, database=db, enrich=memory, relationship=relationship,
                                        summary=summary, expected_revision=kwargs["expected_revision"])

    monkeypatch.setattr(temporal_chat, "process_enrichment", controlled)

    async def exercise():
        cache = Path(__file__).resolve().parents[2] / ".uv-cache" / "temporal"
        cache.mkdir(parents=True, exist_ok=True)
        async with await WorkflowEnvironment.start_local(download_dest_dir=str(cache), ui=False) as env:
            activities = temporal_chat.ChatEnrichmentActivities(db)
            async with Worker(env.client, task_queue="chat-integration", workflows=[ChatEnrichmentWorkflow],
                              activities=[activities.enrich, activities.fail]):
                for outcome in ("success", "fail"):
                    session = repo.create("1", character["id"], outcome)
                    state = dict(session["state"], memory_processing="pending", relationship_processing="pending")
                    turn = repo.commit_turn(session, outcome, "PRIVATE-USER-CONTENT", "PRIVATE-REPLY", "test",
                                            state=state, enqueue_enrichment=True)
                    assert await temporal_chat.dispatch_enrichment(db, env.client, "chat-integration") == 1
                    handle = env.client.get_workflow_handle("character-enrichment-" + turn["id"] + "-2")
                    assert await asyncio.wait_for(handle.result(), 90) == ("completed" if outcome == "success" else "failed")
                    current = repo.get("1", session["id"])
                    assert current["state"]["memory_processing"] == ("completed" if outcome == "success" else "failed")
                    assert repo.turns("1", session["id"])[0]["reply"] == "PRIVATE-REPLY"
                    history = (await handle.fetch_history()).to_json()
                    assert "PRIVATE-" not in history
                    assert await temporal_chat.dispatch_enrichment(db, env.client, "chat-integration") == 0
                    if outcome == "fail":
                        retries = CharacterChatEnrichmentRepository(db)
                        retries.retry("1", session["id"], current["revision"])
                        assert await temporal_chat.dispatch_enrichment(db, env.client, "chat-integration") == 1
                        retry_handle = env.client.get_workflow_handle("character-enrichment-" + turn["id"] + "-4")
                        assert await asyncio.wait_for(retry_handle.result(), 30) == "completed"
                        assert len(repo.turns("1", session["id"])) == 1
                        assert repo.get("1", session["id"])["state"]["memory_processing"] == "completed"
                assert attempts == {"success": 1, "fail": 4}

            # Accept an outbox job while no worker is polling, then start a new worker.
            session = repo.create("1", character["id"], "restart")
            turn = repo.commit_turn(session, "restart", "PRIVATE-OFFLINE", "PRIVATE-REPLY", "test",
                                    enqueue_enrichment=True)
            assert await temporal_chat.dispatch_enrichment(db, env.client, "chat-integration") == 1
            assert await temporal_chat.dispatch_enrichment(db, env.client, "chat-integration") == 0
            restarted = temporal_chat.ChatEnrichmentActivities(db)
            async with Worker(env.client, task_queue="chat-integration", workflows=[ChatEnrichmentWorkflow],
                              activities=[restarted.enrich, restarted.fail]):
                handle = env.client.get_workflow_handle("character-enrichment-" + turn["id"] + "-2")
                assert await asyncio.wait_for(handle.result(), 30) == "completed"
                assert attempts["restart"] == 1
                assert repo.get("1", session["id"])["state"]["memory_processing"] == "completed"

                session = repo.create("1", character["id"], "摘要联调")
                for index in range(12):
                    turn = repo.commit_turn(session, f"summary-{index}", f"PRIVATE-SUMMARY-{index}", "讨论计划", "test",
                                            enqueue_enrichment=index == 11)
                    session = repo.get("1", session["id"])
                assert await temporal_chat.dispatch_enrichment(db, env.client, "chat-integration") == 1
                handle = env.client.get_workflow_handle(f"character-enrichment-{turn['id']}-{session['revision']}")
                assert await asyncio.wait_for(handle.result(), 30) == "completed"
                current = repo.get("1", session["id"])
                assert current["state"]["summary_processing"] == "completed"
                assert len(current["state"]["summaries"][0]["sources"]) == 6
                assert "PRIVATE-" not in (await handle.fetch_history()).to_json()

    try:
        asyncio.run(exercise())
    finally:
        close = getattr(db, "close", None) or db.close_connection
        close()
