"""Required adapter identity/prompt checks cannot turn failure into base generation."""

from services import chat_answer

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from test_lora_provider_selection import install_api

from api import generate
from db.schemas import MessageRequest
from inference import lora_registry
from inference.vllm_client import VLLMClient


def client_for(http, urls=("https://first.example",)):
    client = VLLMClient.__new__(VLLMClient)
    client._instances = [SimpleNamespace(base_url=url, name=f"vllm-{index}") for index, url in enumerate(urls)]
    client._model = "base-model"
    client._api_key = "fiction-key"
    client._timeout = 0.5
    client._ensure_client = AsyncMock(return_value=http)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("rows,expected", [([], []), ([{"id": "base-model"}], []),
    ([{"id": "base-model"}, {"id": "adapter-B"}, {"id": "adapter-A"}], ["adapter-A", "adapter-B"])])
async def test_complete_catalog_distinguishes_real_absence_from_loaded_adapters(rows, expected):
    calls = []
    def reply(request):
        calls.append(request)
        return httpx.Response(200, json={"data": rows})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
        client = client_for(http)
        assert await client.list_loras() == expected
    assert len(calls) == 1 and calls[0].headers["Authorization"] == "Bearer fiction-key"
    assert calls[0].extensions["timeout"]["read"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"data": None}, {"data": {}}, {"data": [1]},
    {"data": [{}]}, {"data": [{"id": 3}]}, {"data": [{"id": " "}]}])
async def test_invalid_catalog_is_not_replaced_by_an_empty_or_unknown_list(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as http:
        with pytest.raises(ValueError):
            await client_for(http).list_loras()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "json", "transport", "no_instances"])
async def test_catalog_failure_preserves_the_actual_error(failure):
    original = httpx.ConnectError("fiction-private-catalog-error")
    def reply(request):
        if failure == "transport":
            raise original
        return httpx.Response(503) if failure == "http" else httpx.Response(200, text="not-json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
        client = client_for(http, urls=() if failure == "no_instances" else ("https://first.example",))
        expected = {"http": httpx.HTTPStatusError, "json": ValueError,
                    "transport": httpx.ConnectError, "no_instances": RuntimeError}[failure]
        with pytest.raises(expected) as error:
            await client.list_loras()
        if failure == "transport":
            assert error.value is original
        if failure == "no_instances":
            client._ensure_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_pool_adapters_must_be_loaded_on_every_routable_instance():
    rows = {
        "first.example": ["base-model", "shared", "first-only"],
        "second.example": ["base-model", "shared", "second-only"],
    }
    calls = []
    def reply(request):
        calls.append(request.url.host)
        return httpx.Response(200, json={"data": [{"id": name} for name in rows[request.url.host]]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
        result = await client_for(http, urls=("https://first.example", "https://second.example")).list_loras()
    assert result == ["shared"] and sorted(calls) == ["first.example", "second.example"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("action", ["list", "load"])
async def test_probe_failure_or_cancellation_closes_other_pool_requests(cancel, action):
    started, exited, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    failure = httpx.ConnectError("fiction-pool-private-error")
    async def reply(request):
        if request.url.host == "slow.example":
            started.set()
            try:
                await release.wait()
            finally:
                exited.set()
            return httpx.Response(200, json={"data": [{"id": "base-model"}]})
        await started.wait()
        if not cancel:
            raise failure
        await release.wait()
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
        client = client_for(http, urls=("https://slow.example", "https://fail.example"))
        operation = asyncio.create_task(client.list_loras() if action == "list" else
            client.load_lora_adapter("fiction-adapter", "/synthetic/adapters/complete"))
        await asyncio.wait_for(started.wait(), 1)
        if cancel:
            operation.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else httpx.ConnectError) as error:
            await asyncio.wait_for(operation, 1)
        if not cancel:
            assert error.value is failure
        assert exited.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("inventory", ["ready", "absent", "unknown", "fault", "unavailable"])
async def test_selected_lora_requires_successful_inventory_before_generation(monkeypatch, inventory):
    state = install_api(monkeypatch, "vllm")
    failure = httpx.ConnectError("fiction-private-vllm-probe-error")
    listing = AsyncMock(side_effect=failure) if inventory == "fault" else AsyncMock(
        return_value=["controlled-adapter"] if inventory == "ready" else None if inventory == "unknown" else [])
    fake = SimpleNamespace(list_loras=listing, generate=AsyncMock())
    monkeypatch.setattr(generate, "get_vllm_client", AsyncMock(return_value=fake if inventory != 'unavailable' else None))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', state.generation)
    req = MessageRequest(message="完整题设：3张票，每张2元，计算总额。", loraName="controlled-adapter")
    if inventory == "ready":
        result = await generate._generate_reply_impl(req, record_invocation=False)
        assert result.reply == "根据完整当前输入回答。"
        state.generation.assert_awaited_once()
        state.save.assert_awaited_once()
    else:
        with pytest.raises(HTTPException) as error:
            await generate._generate_reply_impl(req, record_invocation=False)
        assert error.value.status_code == (409 if inventory == "absent" else 503)
        assert "fiction-private" not in str(error.value.detail)
        if inventory == "fault":
            assert error.value.__cause__ is failure
        state.generation.assert_not_awaited()
        state.save.assert_not_awaited()
    state.manager.set_provider.assert_not_called()
    if inventory == "unavailable":
        listing.assert_not_awaited()
    else:
        listing.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_base_model_does_not_require_adapter_inventory(monkeypatch):
    state = install_api(monkeypatch, "vllm")
    listing = AsyncMock(side_effect=AssertionError("Base model does not require adapter discovery"))
    monkeypatch.setattr(generate, "get_vllm_client", AsyncMock(return_value=SimpleNamespace(list_loras=listing, generate=AsyncMock())))
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', state.generation)
    result = await generate._generate_reply_impl(MessageRequest(message="完整题设：3张票，每张2元。", loraName="default"),
        record_invocation=False)
    assert result.reply == "根据完整当前输入回答。"
    listing.assert_not_awaited()


@pytest.mark.parametrize("entry,error_type", [(None, TypeError), ({}, KeyError), ({"system_prompt": None}, ValueError),
    ({"system_prompt": " "}, ValueError), ({"system_prompt": 3}, ValueError)])
def test_known_persona_prompt_fault_cannot_turn_into_empty_prompt(monkeypatch, entry, error_type):
    monkeypatch.setitem(lora_registry.LORA_REGISTRY, "minamo", entry)
    with pytest.raises(error_type):
        chat_answer.get_lora_system_prompt("minamo")


def test_real_prompt_read_error_survives_the_api_lookup(monkeypatch):
    failure = OSError("fiction-private-persona-read-error")
    class BrokenRegistry(dict):
        def __getitem__(self, key):
            raise failure
    monkeypatch.setattr(lora_registry, "LORA_REGISTRY", BrokenRegistry({"minamo": {"system_prompt": "完整合法角色提示词"}}))
    with pytest.raises(OSError) as error:
        chat_answer.get_lora_system_prompt("minamo")
    assert error.value is failure


def test_optional_prompt_for_base_or_custom_adapter_keeps_existing_semantics():
    assert chat_answer.get_lora_system_prompt("default") == ""
    assert chat_answer.get_lora_system_prompt("fixture-custom-adapter") == ""
    assert chat_answer.get_lora_system_prompt("minamo") == lora_registry.LORA_REGISTRY["minamo"]["system_prompt"]



@pytest.mark.parametrize("failure", ["http", "transport", "json", "structure", "no_instances"])
async def test_load_requires_complete_pool_catalog_before_any_mutation(failure):
    requests = []
    original = httpx.ConnectError("fiction-private-catalog-failure")

    def reply(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={})
        if request.url.host == "first.example":
            return httpx.Response(200, json={"data": [{"id": "base-model"}]})
        if failure == "transport":
            raise original
        if failure == "http":
            return httpx.Response(503)
        if failure == "json":
            return httpx.Response(200, text="not-json")
        return httpx.Response(200, json={})

    urls = () if failure == "no_instances" else ("https://first.example", "https://second.example")
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
        client = client_for(http, urls=urls)
        expected = {"http": httpx.HTTPStatusError, "transport": httpx.ConnectError,
                    "json": ValueError, "structure": ValueError, "no_instances": RuntimeError}[failure]
        with pytest.raises(expected) as caught:
            await client.load_lora_adapter("fiction-adapter", "/synthetic/adapters/complete")
        if failure == "transport":
            assert caught.value is original
        if failure == "no_instances":
            client._ensure_client.assert_not_awaited()
    assert all(request.method == "GET" for request in requests)
