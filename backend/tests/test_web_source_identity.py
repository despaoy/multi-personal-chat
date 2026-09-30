"""HTTP receipt identity must reach saved dialogue and actual completion writes."""
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.profile_registry import CharacterProfileRegistry
from db.database import SQLiteDB
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput


@pytest.mark.asyncio
@pytest.mark.parametrize("message,captured", [
    ("我现在住在芜湖。", True),
    ("不要记住，我喜欢素描。", False),
    ("我在讨论小说：主角说‘我搬家了’，不是我的经历。", False),
])
async def test_http_web_receipt_survives_real_save_and_completion_without_client_id(tmp_path, monkeypatch, message, captured):
    import api.generate as gen
    import character.memory_llm as memory_llm
    from inference import model_manager

    database = SQLiteDB(tmp_path / "web.sqlite")
    repo = DatabaseCharacterMemoryRepository(database)
    registry = CharacterProfileRegistry()
    registry.load_profiles()
    context = CharacterContextService(registry, repo, DatabaseMessageRepository(database))

    class EmptyWriter:
        async def complete(self, messages):
            return '{"memories":[]}'

        async def close(self):
            pass

    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "unused"), completion=EmptyWriter())
    monkeypatch.setattr(memory_llm, "get_memory_enrichment_scheduler", lambda: scheduler)
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: SimpleNamespace(_current_provider=SimpleNamespace(value="vllm")))
    monkeypatch.setattr(gen, "response_cache", None)
    monkeypatch.setattr(gen, "circuit_breaker_registry", None)
    monkeypatch.setattr(gen, "_vllm_client", object())

    async def available():
        return True

    async def model_reply(*args, **kwargs):
        return "知道了。", False, {}

    monkeypatch.setattr(gen, "_ensure_vllm", available)
    monkeypatch.setattr(gen, "_generate_with_vllm", model_reply)
    requests = []

    class Queued:
        async def generate_queued(self, request, current_user):
            requests.append(request)
            return await gen._generate_reply_impl(request, current_user, character_service=context,
                message_db=database, enable_rag=False, record_invocation=False)

    app = FastAPI()
    app.include_router(gen.router)
    app.dependency_overrides[gen.get_current_user] = lambda: {"id": "alice"}
    app.dependency_overrides[gen.get_request_chat_generation_service] = lambda: Queued()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payload = dict(message=message, characterId="tsukiyashiro_kisaki",
                           userId="untrusted", senderId="untrusted", sessionId="room", loraName="default")
            for _ in range(2):
                response = await client.post("/api/generate", json=payload)
                assert response.status_code == 200, response.text
                assert await scheduler.flush_memory(timeout=3)
        ids = [request.sourceMessageId for request in requests]
        assert all(ids) and len(set(ids)) == 2
        assert all(request.senderId == "alice" for request in requests)
        saved = database._get_connection().execute('SELECT sourceMessageId FROM messages ORDER BY id').fetchall()
        assert [row[0] for row in saved] == ids
        scope = (await context.prepare_turn(TurnInput(
            "我住哪里？", "web", "web-character", "alice", "room", "private"), "tsukiyashiro_kisaki")).user_scope
        assert {row["source_message_id"] for row in await repo.list_sources("tsukiyashiro_kisaki", scope)} == (set(ids) if captured else set())
    finally:
        await scheduler.shutdown(timeout=3)


@pytest.mark.asyncio
async def test_http_preserves_supplied_identity_and_does_not_assign_branch_or_integration_ids():
    from unittest.mock import AsyncMock

    import api.generate as gen
    from db.schemas import MessageRequest

    service = SimpleNamespace(generate_queued=AsyncMock(return_value="ok"))
    for changes in (dict(characterId="tsukiyashiro_kisaki", sourceMessageId="provided"),
                    dict(platform="qq", adapter="nonebot", sourceMessageId="external"),
                    dict(platform="web", branchId="branch", sourceMessageId="")):
        request = MessageRequest(message="hello", **changes)
        await gen.generate_reply(request, {"id": "alice"}, service)
        assert request.sourceMessageId == changes["sourceMessageId"]
