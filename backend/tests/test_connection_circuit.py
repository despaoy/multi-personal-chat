"""Explicitly selected cloud connections own independent failure state."""

import asyncio
import copy

import pytest
from fastapi import HTTPException

from api import enhanced
from inference import model_manager
from infra.circuit_breaker import CircuitOpenError, CircuitState
from test_circuit_strict_generation import (
    REPLY, archives, circuit_rig as circuit_rig, records, turn,
)

pytestmark = pytest.mark.usefixtures("no_live_context_review")


def select_connection(monkeypatch, manager, connection):
    connection.snapshot = lambda: connection
    connection.get_status = lambda: {"modelName": connection.model}
    monkeypatch.setattr(manager, "get_current_provider", lambda: connection)


@pytest.mark.parametrize("field,value", [
    ("base_url", "https://selected.fixture.invalid/v1"),
    ("model", "fiction-selected-model"),
    ("api_key", "synthetic-selected-credential"),
])
async def test_explicit_switch_keeps_each_connections_failure_state(circuit_rig, monkeypatch, field, value):
    rig = circuit_rig
    manager = model_manager.get_model_manager()
    original = manager.get_current_provider()
    failure = RuntimeError("synthetic old connection failure")
    rig.fault = failure
    for index in range(5):
        with pytest.raises(HTTPException) as caught:
            await turn(rig, f"old-failure-{index}")
        assert caught.value.__cause__ is failure
    old = await rig.registry.get(rig.circuit_name)
    assert old.state == CircuitState.OPEN and old.get_stats()["failure_calls"] == 5
    replacement = copy.copy(original)
    select_connection(monkeypatch, manager, replacement)
    # A fresh snapshot of the same connection must keep the existing trip.
    with pytest.raises(HTTPException) as caught:
        await turn(rig, "same-connection-snapshot")
    assert isinstance(caught.value.__cause__, CircuitOpenError)
    assert len(rig.calls) == 5 and rig.registry.names == [rig.circuit_name]
    setattr(replacement, field, value)
    rig.fault = None
    result = await turn(rig, "explicit-new-connection")
    assert result.reply == REPLY and result.modelInvoked
    assert result.model == f"{replacement.model} (openai_compat)"
    assert len(rig.calls) == 6 and len(rig.registry.names) == 2
    new_name, = set(rig.registry.names) - {rig.circuit_name}
    new = await rig.registry.get(new_name)
    assert new.state == CircuitState.CLOSED and new.get_stats()["success_calls"] == 1
    select_connection(monkeypatch, manager, original)
    with pytest.raises(HTTPException) as caught:
        await turn(rig, "old-connection-still-open")
    assert isinstance(caught.value.__cause__, CircuitOpenError)
    assert len(rig.calls) == 6 and len(archives(rig)) == 1
    monkeypatch.setattr(enhanced, "circuit_breaker_registry", rig.registry)
    stats = await enhanced.get_circuit_breaker_stats(current_user={"role": "admin"})
    assert set(stats["stats"]) == {rig.circuit_name, new_name}
    assert not any(secret in str(stats) for secret in [original.api_key, replacement.api_key, original.base_url])
    await enhanced.reset_circuit_breaker(rig.circuit_name, current_user={"role": "admin"})
    recovered = await turn(rig, "old-connection-manual-reset")
    assert recovered.reply == REPLY and old.state == CircuitState.CLOSED
    assert new.get_stats()["success_calls"] == 1 and new.get_stats()["failure_calls"] == 0
    assert len(rig.calls) == 7 and len(records(rig)) == 9 and len(archives(rig)) == 2


async def test_late_old_failure_cannot_trip_the_new_connection(circuit_rig, monkeypatch):
    rig = circuit_rig
    manager = model_manager.get_model_manager()
    original = manager.get_current_provider()
    complete = original.async_complete
    started, release = asyncio.Event(), asyncio.Event()
    failure = RuntimeError("synthetic delayed old connection failure")

    async def delayed(messages, **params):
        await complete(messages, **params)
        started.set()
        await release.wait()
        raise failure

    original.async_complete = delayed
    old = await rig.registry.get_or_create(rig.circuit_name, failure_threshold=1)
    pending = asyncio.create_task(turn(rig, "old-in-flight"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        replacement = copy.copy(original)
        replacement.api_key = "synthetic-selected-credential"
        replacement.async_complete = complete
        select_connection(monkeypatch, manager, replacement)
        first = await turn(rig, "new-before-old-fails")
        assert first.reply == REPLY and first.modelInvoked
        release.set()
        with pytest.raises(HTTPException) as caught:
            await pending
        assert caught.value.__cause__ is failure and old.state == CircuitState.OPEN
        second = await turn(rig, "new-after-old-fails")
        assert second.reply == REPLY and second.modelInvoked
        new_name, = set(rig.registry.names) - {rig.circuit_name}
        new = await rig.registry.get(new_name)
        assert new.state == CircuitState.CLOSED
        assert new.get_stats()["success_calls"] == 2 and new.get_stats()["failure_calls"] == 0
        assert len(rig.calls) == len(records(rig)) == 3 and len(archives(rig)) == 2
        assert rig.active == 0 and rig.semaphore._value == 4
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
