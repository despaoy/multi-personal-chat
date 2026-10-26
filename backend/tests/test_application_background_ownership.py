"""One application's shutdown must not stop another application's memory work."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from test_delivery_memory import deliver, generated

from character import memory_llm
from character.models import build_user_scope
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services import delivery_memory, turn_completion
from services.character_context import _TurnOutcome


@pytest.fixture
async def applications(tmp_path, monkeypatch):
    from test_application_lifespan_ownership import applications as original

    callbacks = [
        (turn_completion, "start_turn_completions"),
        (turn_completion, "shutdown_turn_completions"),
        (delivery_memory, "ensure_delivery_memory_worker"),
        (delivery_memory, "shutdown_delivery_memory"),
        (memory_llm, "shutdown_memory_enrichment"),
    ]
    real = [(module, name, getattr(module, name)) for module, name in callbacks]
    source = original.__wrapped__(tmp_path, monkeypatch)
    state = await anext(source)
    for module, name, function in real:
        monkeypatch.setattr(module, name, function)
    monkeypatch.setattr(delivery_memory, "POLL_SECONDS", 0.01)
    actual_completion = memory_llm.OpenAICompatibleMemoryCompletion
    wire, clients = [], []

    def completion(config):
        client = actual_completion(config)

        def respond(request):
            payload = json.loads(request.content)
            wire.append((request.url.host, payload))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"memories":[]}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
                },
            )

        client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(memory_llm, "OpenAICompatibleMemoryCompletion", completion)
    state.memory_wire, state.memory_clients = wire, clients
    try:
        yield state
    finally:
        await source.aclose()


async def test_delivery_and_accepted_completions_survive_other_application_shutdown(applications):
    state = applications
    database = state.databases[0]
    completed = []

    async def complete(prepared, turn, **kwargs):
        completed.append(turn.message)
        return _TurnOutcome(source_capture="recorded", memory_enrichment_status="no_change")

    async with state.running(0):
        first = delivery_memory.ensure_delivery_memory_worker(database)
        first.service = SimpleNamespace(complete_turn=complete)
        runtime = turn_completion.get_turn_completion_runtime(database)
        for stage in ["before", "after"]:
            if stage == "after":
                release, started = asyncio.Event(), asyncio.Event()

                async def accepted(started=started, release=release):
                    started.set()
                    await release.wait()
                    return "accepted-result"

                async with state.running(1):
                    other = turn_completion.get_turn_completion_runtime(state.databases[1])
                    assert other is not runtime
                    pending = asyncio.create_task(runtime.run(accepted, timeout=None))
                    await asyncio.wait_for(started.wait(), 1)
                assert other.closed and not runtime.closed and not pending.done()
                assert not first.closed and not first.task.done()
                release.set()
                assert await pending == "accepted-result"
            key, owner, stored = generated(database)
            deliver(database, key, owner, stored)
            first.wakeup.set()
            await first.wait_idle(timeout=3)
            receipt = json.loads(database.integration_receipt("get", key=key)["response"])
            assert receipt["memory_completion_state"] == "completed"
            assert receipt["reply"] == stored["reply"]
            assert completed[-1] == stored["context"]["request"]["message"]
        assert len(completed) == 2 and runtime.active == runtime.reserved == 0
    assert runtime.closed and first.closed and first.task.done()
    assert database._delivery_memory_worker is None


async def test_semantic_writer_uses_own_startup_configuration_and_survives_other_shutdown(applications):
    state = applications
    for index, container in enumerate(state.containers):
        container.startup_env.update(
            MEMORY_LLM_ENABLED="true",
            MEMORY_LLM_BASE_URL=f"https://memory-{index}.invalid/v1",
            MEMORY_LLM_MODEL=f"synthetic-memory-{index}",
            MEMORY_LLM_API_KEY=f"synthetic-memory-key-{index}",
        )
    scope = build_user_scope("qq", "unit", "synthetic-reader", "synthetic-reader", "private")
    observed = datetime(2026, 10, 26, tzinfo=timezone.utc)
    message = "我本人长期喜欢深蓝色油墨，这是明确的个人偏好。"

    async def process(index, source):
        database = state.databases[index]
        scheduler = memory_llm.get_memory_enrichment_scheduler(database)
        result = await scheduler.schedule_and_wait(
            repository=DatabaseCharacterMemoryRepository(database),
            character_id="tsukiyashiro_kisaki",
            user_scope=scope,
            message=message,
            rule_hints=(),
            source_message_id=source,
            observed_at=observed,
        )
        assert result["status"] == "no_change" and result["persisted"] == 0, result
        rows = database.list_memory_sources(
            "tsukiyashiro_kisaki",
            scope.platform,
            scope.adapter,
            scope.sender_id,
            scope.conversation_type,
            scope.conversation_id,
        )
        assert any(row["source_message_id"] == source and row["body"] == message for row in rows)
        assert state.memory_wire[-1][1]["model"] == f"synthetic-memory-{index}"
        assert message in state.memory_wire[-1][1]["messages"][-1]["content"]
        return scheduler

    async with state.running(0):
        first = await process(0, "synthetic-owner-before")
        async with state.running(1):
            other = await process(1, "synthetic-owner-other")
            assert other is not first
        assert other.status.closed and not first.status.closed
        assert state.databases[1]._memory_enrichment_scheduler is None
        assert await process(0, "synthetic-owner-after") is first
    assert first.status.closed and state.databases[0]._memory_enrichment_scheduler is None
    assert [host for host, _ in state.memory_wire] == ["memory-0.invalid", "memory-1.invalid", "memory-0.invalid"]
    assert all(client._client is None for client in state.memory_clients)
