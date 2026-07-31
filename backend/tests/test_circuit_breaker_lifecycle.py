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