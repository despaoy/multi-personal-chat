"""Rate limits keep validated configuration while lifecycle state resets."""

import math
from unittest.mock import Mock

import pytest

from infra import concurrency_control as controls


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_rate_fails_before_allocating_limiter_state(monkeypatch, rate):
    lock = Mock(side_effect=AssertionError("invalid limiter must not allocate state"))
    monkeypatch.setattr(controls.asyncio, "Lock", lock)
    with pytest.raises(ValueError, match="rate must be nonnegative and finite"):
        controls.TokenBucketLimiter(rate, 1)
    lock.assert_not_called()


@pytest.mark.parametrize("capacity", [0, -1])
def test_nonpositive_capacity_is_not_replaced_with_one(monkeypatch, capacity):
    lock = Mock(side_effect=AssertionError("invalid limiter must not allocate state"))
    monkeypatch.setattr(controls.asyncio, "Lock", lock)
    with pytest.raises(ValueError, match="capacity must be positive"):
        controls.TokenBucketLimiter(1.0, capacity)
    lock.assert_not_called()


async def test_zero_rate_preserves_finite_burst_refund_and_retry_contract():
    limiter = controls.TokenBucketLimiter(0.0, 2)
    assert await limiter.acquire("synthetic-user") == (True, 0.0)
    assert await limiter.acquire("synthetic-user") == (True, 0.0)
    assert await limiter.acquire("synthetic-user") == (False, 60.0)
    await limiter.refund("synthetic-user")
    assert await limiter.acquire("synthetic-user") == (True, 0.0)
    assert limiter.stats()["allowed"] == 2 and limiter.stats()["rejected"] == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"CHAT_GLOBAL_QPS": "19", "CHAT_GLOBAL_BURST": "99"},
        {"CHAT_GLOBAL_QPS": "synthetic-invalid-rate", "CHAT_GLOBAL_BURST": "0"},
    ],
)
async def test_shutdown_resets_owned_state_without_rereading_environment(monkeypatch, changed):
    values = {"GLOBAL": (0.0, 2), "CONVERSATION": (1.5, 3), "SENDER": (0.5, 4)}
    for scope, (rate, capacity) in values.items():
        monkeypatch.setenv(f"CHAT_{scope}_QPS", str(rate))
        monkeypatch.setenv(f"CHAT_{scope}_BURST", str(capacity))
    runtime = controls.InferenceRuntime()
    original = [runtime.global_limiter, runtime.conversation_limiter, runtime.sender_limiter]
    locks = [limiter._lock for limiter in original]
    for limiter in original:
        assert (await limiter.acquire("synthetic-user"))[0]
    for name, value in changed.items():
        monkeypatch.setenv(name, value)
    await runtime.shutdown()
    assert [runtime.global_limiter, runtime.conversation_limiter, runtime.sender_limiter] == original
    for limiter, lock, (rate, capacity) in zip(original, locks, values.values(), strict=True):
        stats = limiter.stats()
        assert limiter.rate == rate and limiter.capacity == capacity and math.isfinite(limiter.rate)
        assert stats["active_buckets"] == stats["allowed"] == stats["rejected"] == 0
        assert limiter._lock is not lock
        assert (await limiter.acquire("synthetic-user"))[0]


@pytest.mark.parametrize("scope", ["GLOBAL", "CONVERSATION", "SENDER"])
@pytest.mark.parametrize("kind,value,error", [("QPS", "-1", "rate"), ("BURST", "0", "capacity")])
def test_runtime_does_not_correct_invalid_limiter_configuration(monkeypatch, scope, kind, value, error):
    monkeypatch.setenv(f"CHAT_{scope}_{kind}", value)
    with pytest.raises(ValueError, match=error):
        controls.InferenceRuntime()
