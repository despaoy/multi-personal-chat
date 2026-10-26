"""A configured statistics failure must not look like disabled functionality."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import enhanced
from app import config
from app.dependencies import get_current_admin
from inference import vllm_client


@pytest.fixture
def stats_api(monkeypatch):
    monkeypatch.setattr(config, "is_vllm_enabled", lambda: True)
    client = SimpleNamespace(get_stats=AsyncMock(return_value={"healthy_instances": 1}))
    getter = AsyncMock(return_value=client)
    monkeypatch.setattr(vllm_client, "get_vllm_client", getter)
    from app.runtime import RuntimeContainer
    from infra.concurrency_control import inference_runtime
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=None, is_pg_mode=lambda: False, inference_runtime=inference_runtime)
    app.include_router(enhanced.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin"}
    return SimpleNamespace(app=app, client=client, getter=getter)


@pytest.mark.parametrize("stage", ["initialize", "statistics"])
async def test_configured_failure_preserves_cause_without_logging_private_details(stats_api, caplog, stage):
    failure = RuntimeError("synthetic-private-credential-and-service-address")
    operation = stats_api.getter if stage == "initialize" else stats_api.client.get_stats
    operation.side_effect = failure
    caplog.set_level(logging.ERROR, logger="api.enhanced")
    with pytest.raises(HTTPException) as raised:
        await enhanced._get_vllm_load_balancer_stats()
    assert raised.value.status_code == 503
    assert raised.value.__cause__ is failure
    assert "synthetic-private" not in raised.value.detail + caplog.text
    assert "RuntimeError" in caplog.text
    operation.assert_awaited_once()


@pytest.mark.parametrize("stage", ["initialize", "statistics"])
@pytest.mark.parametrize("path", ["/api/enhanced/status", "/api/enhanced/stats", "/api/enhanced/load-balancer/stats"])
async def test_api_reports_configured_dependency_failure_instead_of_success(stats_api, stage, path):
    operation = stats_api.getter if stage == "initialize" else stats_api.client.get_stats
    operation.side_effect = ConnectionError("synthetic-private-upstream-address")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=stats_api.app), base_url="http://test") as http:
        response = await http.get(path)
    assert response.status_code == 503
    assert "vLLM" in response.json()["detail"]
    assert "synthetic-private" not in response.text
    operation.assert_awaited_once()


async def test_disabled_vllm_needs_no_client_and_keeps_status_contract(stats_api, monkeypatch):
    monkeypatch.setattr(config, "is_vllm_enabled", lambda: False)
    stats_api.getter.side_effect = AssertionError("Disabled vLLM must not initialize a client")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=stats_api.app), base_url="http://test") as http:
        status = await http.get("/api/enhanced/status")
        stats = await http.get("/api/enhanced/stats")
        direct = await http.get("/api/enhanced/load-balancer/stats")
    assert status.status_code == stats.status_code == 200
    assert status.json()["enhancedFeatures"]["loadBalancer"] is False
    assert status.json()["availableFeatures"]["loadBalancer"] is True
    assert "loadBalancer" not in stats.json()["stats"]
    assert direct.status_code == 503
    stats_api.getter.assert_not_awaited()


@pytest.mark.parametrize("healthy", [0, 1])
async def test_valid_stats_preserve_actual_health_instead_of_fabricating_a_result(stats_api, healthy):
    expected = {"healthy_instances": healthy, "total_instances": 1, "instances": [{"name": "synthetic"}]}
    stats_api.client.get_stats.return_value = expected
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=stats_api.app), base_url="http://test") as http:
        direct = await http.get("/api/enhanced/load-balancer/stats")
        general = await http.get("/api/enhanced/stats")
        status = await http.get("/api/enhanced/status")
    assert direct.status_code == general.status_code == status.status_code == 200
    assert direct.json()["stats"] == general.json()["stats"]["loadBalancer"] == expected
    assert status.json()["enhancedFeatures"]["loadBalancer"] is True


async def test_statistics_cancellation_is_not_translated_to_unavailability(stats_api):
    failure = asyncio.CancelledError()
    stats_api.client.get_stats.side_effect = failure
    with pytest.raises(asyncio.CancelledError) as raised:
        await enhanced._get_vllm_load_balancer_stats()
    assert raised.value is failure
