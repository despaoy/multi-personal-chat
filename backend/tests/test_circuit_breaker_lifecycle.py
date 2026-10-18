"""Regression tests for circuit-breaker API and application lifecycles."""

from __future__ import annotations

import asyncio

from api import enhanced
from infra.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry


def test_circuit_breaker_and_registry_restart_on_fresh_event_loops():
    breaker = CircuitBreaker(name="test")
    registry = CircuitBreakerRegistry()

    async def first_lifecycle():
        registered = await registry.get_or_create("test")
        await breaker.record_success()
        return breaker._get_lock(), registry._get_lock(), registered

    async def second_lifecycle():
        await breaker.record_success()
        registered = await registry.get("test")
        return breaker._get_lock(), registry._get_lock(), registered

    first_breaker_lock, first_registry_lock, first_registered = asyncio.run(
        first_lifecycle()
    )
    second_breaker_lock, second_registry_lock, second_registered = asyncio.run(
        second_lifecycle()
    )

    assert second_breaker_lock is not first_breaker_lock
    assert second_registry_lock is not first_registry_lock
    assert second_registered is first_registered

    asyncio.run(registry.clear())
    assert registry.names == []


def test_enhanced_reset_awaits_registry_and_breaker(monkeypatch):
    calls = []

    class Breaker:
        async def reset(self):
            calls.append("reset")

    class Registry:
        async def get(self, name):
            calls.append(("get", name))
            return Breaker()

    monkeypatch.setattr(enhanced, "circuit_breaker_registry", Registry())

    result = asyncio.run(
        enhanced.reset_circuit_breaker("vllm", current_user={"role": "admin"})
    )

    assert result["success"] is True
    assert calls == [("get", "vllm"), "reset"]


def test_app_config_uses_the_exported_global_registry():
    from app import config
    from infra.circuit_breaker import global_registry

    assert config.circuit_breaker_registry is global_registry

def test_managed_vllm_registration_follows_client_lifecycle(monkeypatch):
    from app import config
    from inference import vllm_client

    registry = CircuitBreakerRegistry()
    monkeypatch.setattr(config, "circuit_breaker_registry", registry)
    monkeypatch.setattr(vllm_client, "_shared_client", None)
    monkeypatch.setenv("VLLM_BASE_URLS", "https://registry.fixture.invalid")

    async def exercise():
        try:
            first = await vllm_client.get_vllm_client()
            assert await registry.get("vllm") is first._circuit_breaker
            await vllm_client.close_shared_vllm_client()
            assert await registry.get("vllm") is None
            independent = vllm_client.VLLMClient()
            assert await registry.get("vllm") is None
            await independent.close()
            second = await vllm_client.get_vllm_client()
            assert second is not first and await registry.get("vllm") is second._circuit_breaker
            await second._circuit_breaker.trip()
            await enhanced.reset_circuit_breaker("vllm", current_user={"role": "admin"})
            assert second._circuit_breaker.state.value == "closed"
        finally:
            await vllm_client.close_shared_vllm_client()

    monkeypatch.setattr(enhanced, "circuit_breaker_registry", registry)
    asyncio.run(exercise())


def test_registration_failure_keeps_shared_slot_empty_for_next_request(monkeypatch):
    from unittest.mock import AsyncMock

    import pytest

    from app import config
    from inference import vllm_client

    registry = CircuitBreakerRegistry()
    monkeypatch.setattr(config, "circuit_breaker_registry", registry)
    monkeypatch.setattr(vllm_client, "_shared_client", None)
    monkeypatch.setenv("VLLM_BASE_URLS", "https://registry.fixture.invalid")
    register = registry.register
    failure = RuntimeError("synthetic-registry-error")
    monkeypatch.setattr(registry, "register", AsyncMock(side_effect=failure))

    async def exercise():
        with pytest.raises(RuntimeError) as caught:
            await vllm_client.get_vllm_client()
        assert caught.value is failure and vllm_client._shared_client is None
        monkeypatch.setattr(registry, "register", register)
        try:
            client = await vllm_client.get_vllm_client()
            assert await registry.get("vllm") is client._circuit_breaker
        finally:
            await vllm_client.close_shared_vllm_client()

    asyncio.run(exercise())
