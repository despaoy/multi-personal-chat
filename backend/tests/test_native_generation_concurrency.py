"""Cloud dispatch uses admission capacity; mutable local adapters stay serial."""

from services import chat_answer

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from api import generate
from db.schemas import MessageRequest
from inference import model_manager

FACTS = "虚构题设：3张票，每张2元，共6元；尚未付款，下雨取消。请保留金额、否定和条件。"


@pytest.fixture
def installed(monkeypatch):
    state = SimpleNamespace(active=0, maximum=0, selected=None, calls=[], fault=None,
        entered=asyncio.Event(), release=None, capacity_reached=asyncio.Event(), expected_capacity=None)
    provider = SimpleNamespace(name="openai_compat", base_url="https://native.invalid/v1", model="fixture-cloud")

    async def complete(messages, **kwargs):
        state.active += 1
        state.maximum = max(state.maximum, state.active)
        if state.active == state.expected_capacity:
            state.capacity_reached.set()
        state.entered.set()
        state.calls.append((messages, state.selected))
        try:
            if state.release is not None:
                await state.release.wait()
            else:
                await asyncio.sleep(0.01)
            if state.fault:
                raise state.fault
            return messages[-1]["content"], 0.01
        finally:
            state.active -= 1

    provider.async_complete = complete
    provider.snapshot = lambda: provider
    provider.get_status = lambda: {"modelName": "fixture-cloud"}
    async def local(**kwargs):
        before = state.selected
        result = await complete([{"role": "user", "content": kwargs["prompt"]}])
        assert state.selected == before
        return result

    manager = SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat"),
        get_current_provider=lambda: provider, async_generate=local,
        set_lora_adapter=Mock(side_effect=lambda path: setattr(state, "selected", path)),
        get_status=lambda: {"currentProvider": manager._current_provider.value,
            "providers": {"openai_compat": {"modelName": "fixture-cloud"}, "transformers_peft": {"modelName": "fixture-local"}}})
    local_provider = SimpleNamespace(name="transformers_peft", async_generate=local,
        set_lora_adapter=manager.set_lora_adapter, get_status=lambda: {"modelName": "fixture-local"})
    manager._providers = {model_manager.ModelProvider.OPENAI_COMPAT: provider,
        model_manager.ModelProvider.TRANSFORMERS_PEFT: local_provider}
    manager.get_current_provider = lambda: manager._providers[model_manager.ModelProvider(manager._current_provider.value)]
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    database = SimpleNamespace(config={"useKnowledgeBase": False}, loras=[])
    monkeypatch.setattr(generate, "db", database)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", None)
    monkeypatch.setattr(generate, "get_lora_character_id", lambda _: None)
    semaphore = asyncio.Semaphore(2)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: semaphore)
    monkeypatch.setattr(generate, "get_lora_path_by_id", lambda name, **kwargs: "/fiction/" + name)

    async def compiled(request, lora_name, *, model_generate, **kwargs):
        text = await model_generate(messages=[{"role": "system", "content": "保留完整题设。"},
            {"role": "user", "content": request.message}], temperature=0.1, max_tokens=128, top_p=1)
        return text, False, {"modelInvoked": True}
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', compiled)
    return state, manager, database, semaphore


async def receive(index=0, lora="default"):
    return await generate._generate_reply_impl(MessageRequest(message=FACTS + f"独立编号{index}", loraName=lora),
        persist_message=False, record_invocation=False, enable_rag=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [1, 2])
async def test_independent_cloud_requests_overlap_only_up_to_admission_capacity(installed, monkeypatch, capacity):
    state, manager, _, _ = installed
    semaphore = asyncio.Semaphore(capacity)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: semaphore)
    state.expected_capacity = capacity
    state.release = asyncio.Event()
    tasks = [asyncio.create_task(receive(i)) for i in range(3)]
    try:
        await asyncio.wait_for(state.capacity_reached.wait(), 1)
    finally:
        state.release.set()
        results = await asyncio.gather(*tasks)
    assert state.maximum == capacity and state.active == 0 and semaphore._value == capacity
    assert all(result.reply == FACTS + f"独立编号{i}" for i, result in enumerate(results))
    assert len(state.calls) == 3
    manager.set_lora_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_cloud_does_not_wait_for_unrelated_local_adapter_lock(installed):
    state, _, _, _ = installed
    async with generate._get_local_model_lock():
        response = await asyncio.wait_for(receive(), timeout=1)
    assert FACTS in response.reply and state.maximum == 1


@pytest.mark.asyncio
async def test_local_adapter_selection_and_generation_remain_one_serial_operation(installed):
    state, manager, database, semaphore = installed
    manager._current_provider.value = "transformers_peft"
    database.loras = [{"id": name, "name": name, "status": "inactive"} for name in ["fixture-a", "fixture-b"]]
    await asyncio.gather(receive(0, "fixture-a"), receive(1, "fixture-b"))
    assert state.maximum == 1 and state.active == 0 and semaphore._value == 2
    assert {selection for _, selection in state.calls} == {"/fiction/fixture-a", "/fiction/fixture-b"}
    assert manager.set_lora_adapter.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failure_or_cancellation_releases_capacity_without_retry(installed, cancel):
    state, _, _, semaphore = installed
    state.release = asyncio.Event()
    state.fault = RuntimeError("fiction-provider-error") if not cancel else None
    task = asyncio.create_task(receive())
    await asyncio.wait_for(state.entered.wait(), 1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        state.release.set()
        with pytest.raises(HTTPException) as error:
            await task
        assert error.value.status_code == 500 and error.value.__cause__ is state.fault
    assert state.active == 0 and semaphore._value == 2 and len(state.calls) == 1




@pytest.mark.asyncio
async def test_provider_change_during_dispatch_does_not_relabel_the_completed_response(installed):
    state, manager, _, semaphore = installed
    state.release = asyncio.Event()
    task = asyncio.create_task(receive())
    try:
        await asyncio.wait_for(state.entered.wait(), 1)
        manager._current_provider.value = "ollama"
    finally:
        state.release.set()
    result = await task
    assert result.model == "fixture-cloud (openai_compat)" and FACTS in result.reply
    assert len(state.calls) == 1 and state.active == 0 and semaphore._value == 2
    manager.set_lora_adapter.assert_not_called()
