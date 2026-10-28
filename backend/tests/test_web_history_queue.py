"""Real web requests queue by history ownership rather than client tab names."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService
from test_application_lifespan_ownership import applications as applications

from app.config import create_access_token
from character.profile_registry import get_default_profile_registry
from infra.concurrency_control import InferenceRuntime

FIRST = "合成第一轮：请把完整材料润色成简短通知：尚未付款，仅周五执行，出差顺延。"
SECOND = "合成第二轮：请继续润色刚才的通知，不要遗漏日期、否定和出差例外。"
ANSWER = "合成控制通知：尚未付款，仅周五执行，出差时顺延。"
ROLE = "tsukiyashiro_kisaki"


@pytest.mark.parametrize("same_owner", [True, False])
async def test_real_web_queue_preserves_history_order_and_independent_owners(
    applications,
    monkeypatch,
    same_owner,
):
    from services import character_context

    for name in (
        "DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED",
        "CONTEXTUAL_MEMORY_SELECTION_ENABLED",
        "CONTEXTUAL_DECISION_POLICY_ENABLED",
    ):
        monkeypatch.setenv(name, "false")
    state = applications
    database = state.databases[0]
    users = [
        database.add_user("synthetic-queue-alice", "unused-hash-for-signed-token"),
        database.add_user("synthetic-queue-bob", "unused-hash-for-signed-token"),
    ]
    tokens = [create_access_token(user["username"], int(user["id"]), user["role"]) for user in users]
    monkeypatch.setenv("INFERENCE_WORKERS", "2")
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT", "10")
    runtime = InferenceRuntime()
    state.containers[0].inference_runtime = runtime
    service = CharacterContextService(
        get_default_profile_registry(),
        DatabaseCharacterMemoryRepository(database),
        DatabaseMessageRepository(database),
        memory_service=SimpleNamespace(recall_with_diagnostics=AsyncMock(return_value=((), 0, {"status": "no_match"}))),
        source_recall_enabled=False,
    )
    monkeypatch.setattr(character_context, "build_character_context_service", lambda owner: service)
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    second_submitted = asyncio.Event()
    release = asyncio.Event()
    wire = []
    queue_keys = []
    submit = runtime.submit

    async def observed_submit(factory, **kwargs):
        queue_keys.append(kwargs["session_id"])
        if len(queue_keys) == 2:
            second_submitted.set()
        return await submit(factory, **kwargs)

    monkeypatch.setattr(runtime, "submit", observed_submit)

    async def respond(request):
        body = json.loads(request.content)
        current = body["messages"][-1]["content"]
        wire.append(body)
        if FIRST in current:
            first_started.set()
            await release.wait()
        else:
            assert SECOND in current
            second_started.set()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    tasks = []
    async with state.running(0):
        await state.pools[0]._client.aclose()
        state.pools[0]._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients = [
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=state.apps[0], raise_app_exceptions=False),
                base_url="https://isolated",
                headers={"Authorization": "Bearer " + tokens[index]},
            )
            for index in [0, 0 if same_owner else 1]
        ]
        try:
            first = asyncio.create_task(
                clients[0].post(
                    "/api/generate",
                    json={
                        "message": FIRST,
                        "characterId": ROLE,
                        "sessionType": "private",
                        "conversationType": "private",
                        "sessionId": "shared-tab",
                    },
                )
            )
            tasks.append(first)
            await asyncio.wait_for(first_started.wait(), timeout=5)
            second = asyncio.create_task(
                clients[1].post(
                    "/api/generate",
                    json={
                        "message": SECOND,
                        "characterId": ROLE,
                        "sessionType": "private",
                        "conversationType": "private",
                        "sessionId": "other-tab" if same_owner else "shared-tab",
                    },
                )
            )
            tasks.append(second)
            await asyncio.wait_for(second_submitted.wait(), timeout=5)
            # Force the old race to completion without timing-based sleeps.
            # Equal keys use the runtime's existing, separately tested lock.
            if not same_owner or queue_keys[0] != queue_keys[1]:
                await asyncio.wait_for(second_started.wait(), timeout=5)
            release.set()
            responses = await asyncio.gather(*tasks)
            assert [response.status_code for response in responses] == [200, 200]
            assert len(wire) == 2
            first_wire = next(body for body in wire if FIRST in body["messages"][-1]["content"])
            second_wire = next(body for body in wire if SECOND in body["messages"][-1]["content"])
            assert any(FIRST in row["content"] for row in first_wire["messages"])
            if same_owner:
                assert any(FIRST in row["content"] for row in second_wire["messages"])
                assert any(ANSWER in row["content"] for row in second_wire["messages"])
            else:
                assert all(FIRST not in row["content"] for row in second_wire["messages"])
            rows = database.get_messages()
            assert len(rows) == 2 and {row["message"] for row in rows} == {FIRST, SECOND}
            assert len({row["senderId"] for row in rows}) == (1 if same_owner else 2)
            assert {row["sessionId"] for row in rows} == ({"shared-tab", "other-tab"} if same_owner else {"shared-tab"})
            assert all(row["sourceMessageId"].startswith("web:") for row in rows)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            for client in clients:
                await client.aclose()
    assert runtime.stats()["active"] == 0
