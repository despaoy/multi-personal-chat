""" native generation must reject an open circuit without default None.

SQLite accounting/archive, request compilation, registry and breaker are real.
The cloud completion boundary has complete synthetic input/output and never
creates an HTTP client. All provider fees are zero.
"""

import asyncio
from contextlib import suppress
from html import unescape
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection

from api import generate
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference import model_manager
from inference.request_context import resolve_request_context
from inference.context_budget import InputBudgetError
from infra.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError, CircuitState

pytestmark = pytest.mark.usefixtures("no_live_context_review")
FACTS = "完整虚构题设：林绮周四14:00开会，15:00结束；尚未出发，下雨改为线上。请准确复述时间、否定和条件。"
REPLY = "林绮周四14:00至15:00开会，尚未出发；下雨改为线上。"


def request(label):
    return MessageRequest(
        message=FACTS, loraName="default", platform="web", adapter="fiction-breaker-test",
        conversationId="fiction-breaker", sessionId="fiction-breaker", senderId="fiction-sender",
        sourceMessageId=f"fiction-source-{label}", traceId=f"fiction-circuit-{label}",
    )


@pytest.fixture
async def circuit_rig(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path / "strict-circuit.sqlite")
    database.update_config(dict(useKnowledgeBase=False, temperature=.2, maxTokens=128, topP=1.0))
    registry = CircuitBreakerRegistry()
    semaphore = asyncio.Semaphore(4)
    rig = SimpleNamespace(
        database=database, registry=registry, semaphore=semaphore,
        calls=[], fault=None, hold=None, active=0,
        entered=asyncio.Event(), cancelled=asyncio.Event(),
    )

    async def complete(messages, **params):
        # Preserve the real compiler; this boundary must receive the complete
        # fictional question, its negation and exception on every actual call.
        visible = unescape("\n".join(message["content"] for message in messages))
        assert FACTS in visible
        assert any(message["role"] == "system" for message in messages)
        assert params["temperature"] == .2 and params["max_tokens"] == 128 and params["top_p"] == 1.0
        rig.calls.append(dict(messages=[dict(message) for message in messages], params=dict(params), provider_fee=0))
        rig.active += 1
        rig.entered.set()
        try:
            if rig.hold is not None:
                await rig.hold.wait()
            if rig.fault is not None:
                raise rig.fault
            return REPLY, 0.0
        except asyncio.CancelledError:
            rig.cancelled.set()
            raise
        finally:
            rig.active -= 1

    connection = cloud_connection("fiction-circuit-model", completion=complete)
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value="openai_compat"),
        get_current_provider=lambda: connection,
    )
    rig.circuit_name = f"model_generate:{(await resolve_request_context(manager)).connection_id}"
    monkeypatch.setattr(model_manager, "get_model_manager", lambda: manager)
    monkeypatch.setattr(generate, "db", database)
    monkeypatch.setattr(generate, "response_cache", None)
    monkeypatch.setattr(generate, "circuit_breaker_registry", registry)
    monkeypatch.setattr(generate, "get_llm_semaphore", lambda: semaphore)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "只依据完整给定题设；保留时间、否定和例外。")
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=65536, evidence_max_chars=0))
    monkeypatch.setenv("CHARACTER_INDEPENDENT_TASKS_ENABLED", "false")
    try:
        yield rig
    finally:
        await registry.clear()
        database.close_connection()


async def turn(rig, label):
    return await generate._generate_reply_impl(request(label), enable_rag=False, message_db=rig.database)


def records(rig):
    return rig.database.execute_sql("SELECT * FROM model_invocations ORDER BY id")


def archives(rig):
    return rig.database.execute_sql("SELECT * FROM messages ORDER BY id")


def expire_open_interval(breaker):
    # Advance only this breaker's existing recovery interval; do not patch
    # process time, change the production threshold or sleep for thirty seconds.
    assert breaker.state == CircuitState.OPEN and breaker._last_failure_time is not None
    breaker._last_failure_time -= breaker.recovery_timeout + 1
    assert breaker.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_healthy_native_success_records_and_archives_real_reply(circuit_rig):
    rig = circuit_rig
    result = await turn(rig, "healthy")
    assert result.reply == REPLY and result.modelInvoked is True
    assert len(rig.calls) == 1 and rig.calls[0]["provider_fee"] == 0
    breaker = await rig.registry.get(rig.circuit_name)
    assert breaker.state == CircuitState.CLOSED
    stats = breaker.get_stats()
    assert stats["success_calls"] == 1 and stats["failure_calls"] == 0
    rows, saved = records(rig), archives(rig)
    assert len(rows) == len(saved) == 1
    assert rows[0]["errorType"] == "" and rows[0]["completionTokens"] is None
    assert saved[0]["message"] == FACTS and saved[0]["reply"] == REPLY
    assert saved[0]["traceId"] == request("healthy").traceId
    assert rig.active == 0 and rig.semaphore._value == 4


@pytest.mark.asyncio
async def test_five_real_failures_then_open_rejection_has_actual_cause_and_recovers(circuit_rig):
    rig = circuit_rig
    failure = RuntimeError("synthetic controlled provider failure")
    rig.fault = failure
    for number in range(5):
        with pytest.raises(HTTPException) as caught:
            await turn(rig, f"failed-{number}")
        assert caught.value.__cause__ is failure
    breaker = await rig.registry.get(rig.circuit_name)
    assert breaker.failure_threshold == 5 and breaker.state == CircuitState.OPEN
    assert len(rig.calls) == 5 and archives(rig) == []
    assert [row["errorType"] for row in records(rig)] == ["RuntimeError"] * 5

    with pytest.raises(HTTPException) as caught:
        await turn(rig, "rejected-sixth")
    # Baseline returns DEFAULT None and instead exposes a TypeError from tuple
    # unpacking. The rejection must preserve the actual open-circuit cause.
    assert isinstance(caught.value.__cause__, CircuitOpenError)
    assert caught.value.status_code == 503
    assert caught.value.detail["stage"] == "model_circuit"
    assert caught.value.detail["reason"] == "circuit_open"
    assert caught.value.detail["modelInvoked"] is False
    assert isinstance(caught.value.detail["message"], str) and caught.value.detail["message"]
    assert len(rig.calls) == 5 and archives(rig) == []
    rows = records(rig)
    assert len(rows) == 6
    assert [row["errorType"] for row in rows] == ["RuntimeError"] * 5 + ["CircuitOpenError"]
    assert all(row["completionTokens"] is None for row in rows)
    assert breaker.get_stats()["failure_calls"] == 5  # Rejection never executes the provider.

    expire_open_interval(breaker)
    rig.fault = None
    recovered = await turn(rig, "recovered")
    assert recovered.reply == REPLY and recovered.modelInvoked is True
    assert len(rig.calls) == 6 and sum(call["provider_fee"] for call in rig.calls) == 0
    assert breaker.state == CircuitState.CLOSED
    rows, saved = records(rig), archives(rig)
    assert len(rows) == 7 and rows[-1]["errorType"] == "" and rows[-1]["completionTokens"] is None
    assert len(saved) == 1 and saved[0]["reply"] == REPLY
    assert saved[0]["traceId"] == request("recovered").traceId
    assert rig.active == 0 and rig.semaphore._value == 4


@pytest.mark.asyncio
async def test_real_half_open_admission_rejects_concurrent_extra_probe():
    registry = CircuitBreakerRegistry()
    breaker = await registry.get_or_create(
        "fiction-half-open", failure_threshold=1, recovery_timeout=30,
        half_open_max_calls=1,
    )
    await breaker.record_failure()
    expire_open_interval(breaker)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def probe():
        calls.append("actual probe")
        entered.set()
        await release.wait()
        return "real controlled success"

    first = asyncio.create_task(breaker.call(probe))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(CircuitOpenError):
            await breaker.call(probe)
        assert calls == ["actual probe"] and breaker.state == CircuitState.HALF_OPEN
        release.set()
        assert await asyncio.wait_for(first, 1) == "real controlled success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.get_stats()["success_calls"] == 1 and breaker.get_stats()["failure_calls"] == 1
    finally:
        release.set()
        if not first.done():
            first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await registry.clear()


@pytest.mark.asyncio
async def test_cancelled_half_open_native_probe_releases_admission_and_can_recover(circuit_rig):
    rig = circuit_rig
    breaker = await rig.registry.get_or_create(
        rig.circuit_name, failure_threshold=1, recovery_timeout=30,
        half_open_max_calls=1,
    )
    await breaker.record_failure()
    expire_open_interval(breaker)
    rig.hold = asyncio.Event()
    pending = asyncio.create_task(turn(rig, "cancelled-probe"))
    try:
        await asyncio.wait_for(rig.entered.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert rig.cancelled.is_set() and rig.active == 0 and rig.semaphore._value == 4
        assert len(rig.calls) == 1 and records(rig) == [] and archives(rig) == []
        assert breaker.get_stats()["failure_calls"] == 1 and breaker.get_stats()["success_calls"] == 0
        rig.hold = None
        # This fails if cancellation consumed the only half-open slot forever.
        result = await asyncio.wait_for(turn(rig, "after-cancel"), 1)
        assert result.reply == REPLY and result.modelInvoked is True
        assert len(rig.calls) == 2 and breaker.state == CircuitState.CLOSED
        assert len(records(rig)) == len(archives(rig)) == 1
        assert records(rig)[0]["errorType"] == "" and archives(rig)[0]["traceId"] == request("after-cancel").traceId
    finally:
        if not pending.done():
            pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
async def test_stale_half_open_success_cannot_close_a_new_failure_trip():
    registry = CircuitBreakerRegistry()
    breaker = await registry.get_or_create(
        "fiction-stale-probe", failure_threshold=1, recovery_timeout=30,
        half_open_max_calls=2,
    )
    await breaker.record_failure()
    expire_open_interval(breaker)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    failure = RuntimeError("controlled newer probe failure")

    async def older_success():
        calls.append("older probe")
        entered.set()
        await release.wait()
        return "actual older result"

    async def newer_failure():
        calls.append("newer probe")
        raise failure

    async def fresh_success():
        calls.append("fresh recovery")
        return "actual fresh result"

    older = asyncio.create_task(breaker.call(older_success))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        # Both probes were admitted in HALF_OPEN. The second fails while the
        # first is still in flight, opening a newer circuit generation.
        with pytest.raises(RuntimeError) as caught:
            await breaker.call(newer_failure)
        assert caught.value is failure and breaker.state == CircuitState.OPEN
        release.set()
        assert await asyncio.wait_for(older, 1) == "actual older result"
        assert breaker.state == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await breaker.call(fresh_success)
        assert calls == ["older probe", "newer probe"]
        expire_open_interval(breaker)
        assert await breaker.call(fresh_success) == "actual fresh result"
        assert breaker.state == CircuitState.CLOSED
        assert calls == ["older probe", "newer probe", "fresh recovery"]
    finally:
        release.set()
        if not older.done():
            older.cancel()
        await asyncio.gather(older, return_exceptions=True)
        await registry.clear()


@pytest.mark.asyncio
async def test_late_failure_in_same_probe_round_reopens_after_first_success():
    registry = CircuitBreakerRegistry()
    breaker = await registry.get_or_create(
        "fiction-late-probe-failure", failure_threshold=1, recovery_timeout=30,
        half_open_max_calls=2,
    )
    await breaker.record_failure()
    expire_open_interval(breaker)
    success_entered, failure_entered = asyncio.Event(), asyncio.Event()
    release_success, release_failure = asyncio.Event(), asyncio.Event()
    calls = []
    failure = RuntimeError("controlled same-round late failure")

    async def successful_probe():
        calls.append("successful probe")
        success_entered.set()
        await release_success.wait()
        return "actual successful probe result"

    async def failing_probe():
        calls.append("failing probe")
        failure_entered.set()
        await release_failure.wait()
        raise failure

    async def rejected_probe():
        calls.append("must not execute")
        return "forbidden while open"

    successful = asyncio.create_task(breaker.call(successful_probe))
    failing = asyncio.create_task(breaker.call(failing_probe))
    try:
        await asyncio.wait_for(success_entered.wait(), 1)
        await asyncio.wait_for(failure_entered.wait(), 1)
        assert breaker.state == CircuitState.HALF_OPEN
        release_success.set()
        assert await asyncio.wait_for(successful, 1) == "actual successful probe result"
        assert breaker.state == CircuitState.CLOSED
        # The first success must not retire the authority of another probe
        # admitted in the same round: its actual late failure reopens service.
        release_failure.set()
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(failing, 1)
        assert caught.value is failure and breaker.state == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await breaker.call(rejected_probe)
        assert calls == ["successful probe", "failing probe"]
    finally:
        release_success.set()
        release_failure.set()
        for task in (successful, failing):
            if not task.done():
                task.cancel()
        await asyncio.gather(successful, failing, return_exceptions=True)
        await registry.clear()



async def test_capacity_failures_keep_audit_without_disabling_a_healthy_provider(circuit_rig, monkeypatch):
    rig = circuit_rig
    counters = []
    monkeypatch.setattr(generate, "increment", lambda name, *args, **kwargs: counters.append(name))
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=1024, evidence_max_chars=0))
    for index in range(5):
        value = request(f"capacity-{index}")
        value.message = FACTS * 30
        with pytest.raises(HTTPException) as caught:
            await generate._generate_reply_impl(value, enable_rag=False, message_db=rig.database)
        assert caught.value.status_code == 503
        assert caught.value.detail["kind"] == "context_budget_exceeded"
        assert isinstance(caught.value.__cause__, InputBudgetError)
    assert not rig.calls and not archives(rig)
    assert [row["errorType"] for row in records(rig)] == ["InputBudgetError"] * 5
    assert "model_failures" not in counters
    assert await rig.registry.get(rig.circuit_name) is None
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(window_tokens=65536, evidence_max_chars=0))
    result = await turn(rig, "healthy-after-capacity")
    assert result.modelInvoked and result.reply == REPLY and len(rig.calls) == 1
    breaker = await rig.registry.get(rig.circuit_name)
    assert breaker.state == CircuitState.CLOSED and breaker.get_stats()["failure_calls"] == 0
