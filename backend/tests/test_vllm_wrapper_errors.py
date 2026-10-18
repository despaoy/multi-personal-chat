"""Explicit disable, initialization fault and cleanup fault are distinct outcomes."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import generate
from app import config, main
from inference import vllm_client


@pytest.fixture
def rig(monkeypatch):
    client = SimpleNamespace(health_check=AsyncMock(return_value=[{"healthy": True}]))
    factory = AsyncMock(return_value=client)
    close = AsyncMock()
    monkeypatch.setattr(config, "is_vllm_enabled", lambda: True)
    monkeypatch.setattr(vllm_client, "get_vllm_client", factory)
    monkeypatch.setattr(vllm_client, "close_shared_vllm_client", close)
    return SimpleNamespace(client=client, factory=factory, close=close)


@pytest.mark.asyncio
async def test_explicit_disable_is_still_a_normal_disabled_result(rig, monkeypatch):
    monkeypatch.setattr(config, "is_vllm_enabled", lambda: False)
    assert await generate.get_vllm_client() is None
    assert await generate.vllm_status(current_user={"role": "admin"}) == {"enabled": False, "instances": []}
    rig.factory.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["get", "status"])
async def test_initialization_failure_is_safe_and_only_later_request_retries(rig, caplog, entry):
    failure = ConnectionError("synthetic-private-service-details")
    rig.factory.side_effect = [failure, rig.client]

    async def invoke():
        if entry == "get":
            return await generate.get_vllm_client()
        return await generate.vllm_status(current_user={"role": "admin"})

    with pytest.raises(HTTPException) as caught:
        await invoke()
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert "synthetic-private" not in caught.value.detail + caplog.text
    assert rig.factory.await_count == 1
    result = await invoke()
    if entry == "get":
        assert result is rig.client
    else:
        assert result == {"enabled": True, "instances": [{"healthy": True}]}
    assert rig.factory.await_count == 2


@pytest.mark.asyncio
async def test_empty_shared_factory_result_is_an_error(rig):
    rig.factory.return_value = None
    with pytest.raises(HTTPException) as caught:
        await generate.get_vllm_client()
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_cancelled_initialization_is_not_retried_within_the_request(rig):
    failure = asyncio.CancelledError()
    rig.factory.side_effect = [failure, rig.client]
    with pytest.raises(asyncio.CancelledError) as caught:
        await generate.get_vllm_client()
    assert caught.value is failure
    assert await asyncio.wait_for(generate.get_vllm_client(), 1) is rig.client
    assert rig.factory.await_count == 2


@pytest.mark.asyncio
async def test_status_probe_error_is_not_reported_as_disabled(rig, caplog):
    failure = OSError("synthetic-private-probe-details")
    rig.client.health_check.side_effect = [failure, [{"healthy": True}]]
    with pytest.raises(HTTPException) as caught:
        await generate.vllm_status(current_user={"role": "admin"})
    assert caught.value.status_code == 503 and caught.value.__cause__ is failure
    assert "synthetic-private" not in caught.value.detail + caplog.text
    assert await generate.vllm_status(current_user={"role": "admin"}) == {
        "enabled": True,
        "instances": [{"healthy": True}],
    }
    assert rig.factory.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("synthetic-private-close-details"), asyncio.CancelledError()])
async def test_close_error_reaches_lifecycle_handler_without_false_success(rig, caplog, failure):
    caplog.set_level(logging.INFO)
    await generate.get_vllm_client()
    rig.close.side_effect = failure
    with pytest.raises(type(failure)) as caught:
        await main._close_resource("vLLM 客户端", vllm_client, "close_shared_vllm_client")
    assert caught.value is failure
    rig.close.assert_awaited_once()
    assert "vLLM 客户端已关闭" not in caplog.text and "synthetic-private" not in caplog.text


@pytest.mark.asyncio
async def test_status_http_boundary_reports_initialization_error_as_503(rig):
    rig.factory.side_effect = ConnectionError("synthetic-private-HTTP-details")
    app = FastAPI()
    app.include_router(generate.router)
    app.dependency_overrides[generate.get_current_admin] = lambda: {"role": "admin"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://unit") as client:
        response = await client.get("/api/vllm/status")
    assert response.status_code == 503 and "初始化失败" in response.json()["detail"]
    assert "synthetic-private" not in response.text and '"enabled":false' not in response.text
