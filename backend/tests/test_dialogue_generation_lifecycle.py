"""Real ASGI dialogue ownership and cancellation, with synthetic I/O only.

Research is a valid empty result except for the real HTTP-resource cancellation
case. Model batches follow the actual complete prompt example, never a network
provider. The runner supplies isolated JWT/application/database configuration.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from services import character_research as research

from api import training
from app.dependencies import get_current_admin
from inference import model_manager

DESCRIPTION = "完全虚构角色青汐研究员；仅限预约者，不免除原件要求。"
ENDPOINT = "/api/training/generate-dialogues"
BODY = dict(character_description=DESCRIPTION, num_dialogues=1, style="合成谨慎风格")


def complete_prompt_batch(prompt, ordinal):
    """Use the actual example only after checking its complete wire structure."""
    header = re.search(r"请生成(\d+)组对话数据，每组恰好(\d+)轮对话。", prompt)
    assert header is not None
    count, turns = map(int, header.groups())
    section = prompt.split("【输出格式】", 1)[1]
    examples, _end = json.JSONDecoder().raw_decode(section[section.index("["):])
    assert isinstance(examples, list) and len(examples) == count
    for example in examples:
        assert isinstance(example, dict) and set(example) == {"conversations"}
        messages = example["conversations"]
        assert isinstance(messages, list) and len(messages) == 2 * turns
        for index, message in enumerate(messages):
            assert isinstance(message, dict) and set(message) == {"role", "content"}
            assert message["role"] == ("user" if index % 2 == 0 else "assistant")
            assert isinstance(message["content"], str) and message["content"].strip()
    batch = deepcopy(examples)
    for item_index, item in enumerate(batch):
        for index, message in enumerate(item["conversations"]):
            message["content"] = (
                f"  合成第{ordinal}批第{item_index + 1}组第{index // 2 + 1}轮："
                "2031年以前不适用；仅限预约者，不能免除原件。\n  "
            )
    return count, turns, batch


def mapped_messages(batch):
    return [[{"from": "human" if message["role"] == "user" else "gpt", "value": message["content"]}
             for message in item["conversations"]] for item in batch]


async def reached(*events):
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in events)), timeout=3)


@pytest.fixture
async def rig(monkeypatch):
    state = dict(is_generating=False, cancel_requested=False, progress=0, total=0,
                 batch_num=0, total_batches=0, generated_count=0, new_dialogues=[],
                 all_generated_dialogues=[], started_at=None)
    lock = asyncio.Lock()
    monkeypatch.setattr(training, "generation_state", state)
    monkeypatch.setattr(training, "get_generation_state_lock", lambda: lock)
    monkeypatch.setattr(training, "_dialogue_generation_task", None, raising=False)
    search = AsyncMock(return_value="")
    monkeypatch.setattr(training, "search_character_info", search)
    context = SimpleNamespace(state=state, lock=lock, search=search, calls=[], on_model=None,
                              tasks=[], releases=[])

    async def model(**kwargs):
        ordinal = len(context.calls) + 1
        count, turns, batch = complete_prompt_batch(kwargs["prompt"], ordinal)
        assert kwargs["session_history"] == []
        assert kwargs["max_tokens_override"] == (2048 if turns <= 2 else 4096 if turns <= 5 else 6144)
        context.calls.append((count, turns, batch))
        if context.on_model is not None:
            await context.on_model(ordinal)
        return json.dumps(batch, ensure_ascii=False), 0.25

    context.manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"),
        async_generate=AsyncMock(side_effect=model),
        generate=Mock(side_effect=AssertionError("Synchronous model generation is forbidden")),
    )
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: context.manager)
    app = FastAPI()
    app.include_router(training.router)
    app.dependency_overrides[get_current_admin] = lambda: {"id": 901, "role": "admin"}
    context.app = app
    try:
        yield context
    finally:
        # Release event gates before joining every owned HTTP/child task, also
        # when an assertion fails. No background task survives this fixture.
        for release in context.releases:
            release.set()
        child = training._dialogue_generation_task
        tasks = list(context.tasks)
        if child is not None and child not in tasks:
            tasks.append(child)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def asgi_client(rig):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url="http://fixture.invalid")


def start_post(rig, client, path=ENDPOINT, *, count=1, name="generation-http"):
    kwargs = {"json": {**BODY, "num_dialogues": count}} if path == ENDPOINT else {}
    task = asyncio.create_task(client.post(path, **kwargs), name=name)
    rig.tasks.append(task)
    return task


def assert_released(rig):
    assert rig.state["is_generating"] is False and rig.state["cancel_requested"] is False
    assert training._dialogue_generation_task is not None and training._dialogue_generation_task.done()
    rig.manager.generate.assert_not_called()


def assert_cancelled_zero(response):
    assert response.status_code == 200
    assert response.json() == dict(success=False, cancelled=True, dialogues=[], total=0, cost_time=0.0)


async def test_user_cancel_interrupts_second_model_and_returns_only_completed_batch(rig):
    entered, finished = asyncio.Event(), asyncio.Event()

    async def model_wait(ordinal):
        if ordinal == 2:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

    rig.on_model = model_wait
    async with asgi_client(rig) as client:
        original = start_post(rig, client, count=2)
        await reached(entered)
        assert [(count, turns) for count, turns, _ in rig.calls] == [(1, 1), (1, 2)]
        assert rig.state["generated_count"] == 1 and rig.state["total"] == 2
        assert rig.state["progress"] == 50
        accepted = await client.post(ENDPOINT + "/cancel")
        assert accepted.status_code == 200 and accepted.json() == {"success": True, "message": "已发送取消请求"}
        response = await asyncio.wait_for(original, timeout=3)
        assert finished.is_set()
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is False and result["cancelled"] is True
        assert result["total"] == 1 and result["cost_time"] == 0.25
        assert result["dialogues"][0]["conversations"] == mapped_messages(rig.calls[0][2])[0]
        assert result["dialogues"][0]["turns"] == 1
        assert rig.state["all_generated_dialogues"] == result["dialogues"]
        assert rig.state["new_dialogues"] == result["dialogues"]
        assert rig.state["generated_count"] == 1 and rig.state["progress"] == 50
        assert rig.manager.async_generate.await_count == 2
        rig.search.assert_awaited_once_with(DESCRIPTION)
        assert_released(rig)


async def test_force_reset_waits_for_live_child_and_old_cleanup_cannot_clear_successor(rig, monkeypatch):
    old_entered, old_cleanup = asyncio.Event(), asyncio.Event()
    old_release, cleanup_release = asyncio.Event(), asyncio.Event()
    successor_entered, successor_release = asyncio.Event(), asyncio.Event()
    rig.releases.extend([old_release, cleanup_release, successor_release])
    gate = SimpleNamespace(enabled=False, blocked={})

    @asynccontextmanager
    async def state_lock():
        # Pause only the two old HTTP owners after their real child is done.
        # The actual shared asyncio.Lock still serializes every state access.
        event = gate.blocked.get(asyncio.current_task()) if gate.enabled else None
        if event is not None:
            event.set()
            await cleanup_release.wait()
        async with rig.lock:
            yield

    monkeypatch.setattr(training, "get_generation_state_lock", state_lock)

    async def model_wait(ordinal):
        if ordinal == 1:
            old_entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                old_cleanup.set()
                await old_release.wait()
        else:
            assert ordinal == 2
            successor_entered.set()
            await successor_release.wait()

    rig.on_model = model_wait
    async with asgi_client(rig) as client:
        original = start_post(rig, client, count=2, name="old-generation-http")
        await reached(old_entered)
        old_child = training._dialogue_generation_task
        reset = start_post(rig, client, ENDPOINT + "/force-reset", name="old-reset-http")
        await reached(old_cleanup)
        assert not reset.done() and not old_child.done()
        rejected = await client.post(ENDPOINT, json=BODY)
        assert rejected.status_code == 409
        assert rig.manager.async_generate.await_count == rig.search.await_count == 1
        owner_blocked, reset_blocked = asyncio.Event(), asyncio.Event()
        gate.blocked = {original: owner_blocked, reset: reset_blocked}
        gate.enabled = True
        old_release.set()
        await reached(owner_blocked, reset_blocked)
        assert old_child.done() and not original.done() and not reset.done()

        successor = start_post(rig, client, name="successor-generation-http")
        await reached(successor_entered)
        successor_child = training._dialogue_generation_task
        assert successor_child is not old_child and not successor_child.done()
        assert rig.state["is_generating"] is True and rig.state["total"] == 1
        assert rig.state["cancel_requested"] is False
        cleanup_release.set()
        old_response, reset_response = await asyncio.wait_for(asyncio.gather(original, reset), timeout=3)
        assert_cancelled_zero(old_response)
        assert reset_response.status_code == 200
        assert reset_response.json() == {"success": True, "message": "原生成任务已终止"}
        assert training._dialogue_generation_task is successor_child and not successor_child.done()
        assert rig.state["is_generating"] is True and rig.state["total"] == 1
        assert rig.state["cancel_requested"] is False and rig.state["generated_count"] == 0
        successor_release.set()
        response = await asyncio.wait_for(successor, timeout=3)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True and result["cancelled"] is False and result["total"] == 1
        assert result["dialogues"][0]["conversations"] == mapped_messages(rig.calls[1][2])[0]
        assert rig.manager.async_generate.await_count == rig.search.await_count == 2
        assert rig.state["all_generated_dialogues"] == result["dialogues"]
        assert rig.state["generated_count"] == 1 and rig.state["progress"] == 100
        assert_released(rig)


@pytest.mark.parametrize("user_cancel_first", [False, True], ids=["external-only", "overlapping-user-cancel"])
async def test_external_request_cancellation_propagates_even_with_user_cancel_flag(rig, user_cancel_first):
    entered, cleanup_entered, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleanup_release = asyncio.Event()
    rig.releases.append(cleanup_release)

    async def model_wait(_ordinal):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            try:
                if user_cancel_first:
                    await cleanup_release.wait()
            finally:
                finished.set()

    rig.on_model = model_wait
    async with asgi_client(rig) as client:
        original = start_post(rig, client)
        await reached(entered)
        if user_cancel_first:
            accepted = await client.post(ENDPOINT + "/cancel")
            assert accepted.status_code == 200
            await reached(cleanup_entered)
            assert rig.state["cancel_requested"] is True and not original.done()
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(original, timeout=3)
        assert original.cancelled() and finished.is_set()
        assert rig.state["generated_count"] == 0 and rig.state["all_generated_dialogues"] == []
        rig.manager.async_generate.assert_awaited_once()
        rig.search.assert_awaited_once_with(DESCRIPTION)
        assert_released(rig)


async def test_user_cancel_during_real_research_closes_both_http_operations_before_zero_result(rig, monkeypatch):
    entered = {name: asyncio.Event() for name in ("duckduckgo", "wikipedia")}
    finished = {name: asyncio.Event() for name in entered}
    clients, requests = [], []

    async def transport(request):
        if request.url.host == "html.duckduckgo.com":
            source = "duckduckgo"
            assert request.url.params["q"] == f"{DESCRIPTION} 角色 人物介绍"
        else:
            assert request.url.host == "zh.wikipedia.org"
            source = "wikipedia"
            assert request.url.params["srsearch"] == DESCRIPTION
        requests.append(source)
        entered[source].set()
        try:
            await asyncio.Event().wait()
        finally:
            finished[source].set()

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        client = real_client(transport=httpx.MockTransport(transport), **kwargs)
        clients.append(client)
        return client

    scoped_httpx = SimpleNamespace(**vars(httpx))
    scoped_httpx.AsyncClient = client_factory
    monkeypatch.setattr(research, "httpx", scoped_httpx)
    monkeypatch.setattr(training, "search_character_info", research.search_character_info)
    async with asgi_client(rig) as client:
        original = start_post(rig, client, count=2)
        await reached(*entered.values())
        accepted = await client.post(ENDPOINT + "/cancel")
        assert accepted.status_code == 200
        response = await asyncio.wait_for(original, timeout=3)
        assert_cancelled_zero(response)
        assert sorted(requests) == ["duckduckgo", "wikipedia"]
        assert all(event.is_set() for event in finished.values())
        assert len(clients) == 1 and clients[0].is_closed
        rig.manager.async_generate.assert_not_awaited()
        assert rig.state["generated_count"] == rig.state["progress"] == 0
        assert rig.state["total"] == 2 and rig.state["all_generated_dialogues"] == []
        assert_released(rig)


async def test_user_cancel_before_business_starts_returns_known_zero_and_releases_owner(rig, monkeypatch):
    held, release = asyncio.Event(), asyncio.Event()
    rig.releases.append(release)
    real_create_task = asyncio.create_task
    intercepted = []

    def gated_create_task(coroutine, *, name=None, **kwargs):
        assert name == "dialogue-generation"

        async def deferred_business():
            held.set()
            try:
                await release.wait()
            except BaseException:
                # The real business coroutine has deliberately never been
                # entered; close it rather than leaking an unawaited coroutine.
                coroutine.close()
                raise
            return await coroutine

        task = real_create_task(deferred_business(), name=name, **kwargs)
        intercepted.append(task)
        return task

    # Replace only training's module binding, never asyncio.create_task globally.
    scoped_asyncio = SimpleNamespace(**vars(asyncio))
    scoped_asyncio.create_task = gated_create_task
    monkeypatch.setattr(training, "asyncio", scoped_asyncio)
    async with asgi_client(rig) as client:
        original = start_post(rig, client, count=2)
        await reached(held)
        assert len(intercepted) == 1 and training._dialogue_generation_task is intercepted[0]
        assert rig.state["is_generating"] is True
        accepted = await client.post(ENDPOINT + "/cancel")
        assert accepted.status_code == 200
        response = await asyncio.wait_for(original, timeout=3)
        assert_cancelled_zero(response)
        assert intercepted[0].cancelled()
        rig.search.assert_not_awaited()
        rig.manager.async_generate.assert_not_awaited()
        assert rig.state["generated_count"] == rig.state["progress"] == 0
        assert rig.state["total"] == 2 and rig.state["all_generated_dialogues"] == []
        assert_released(rig)
