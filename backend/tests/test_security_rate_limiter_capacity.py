from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from middleware.security import SlidingWindowLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_expired_keys_are_pruned_in_configured_batches():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(
        max_keys=10,
        cleanup_interval=6,
        cleanup_batch_size=2,
        clock=clock,
    )

    for index in range(5):
        allowed, _, _ = limiter.check_rpm(f"client-{index}", limit=5)
        assert allowed is True

    assert len(limiter._request_windows) == 5
    clock.advance(61)

    allowed, _, _ = limiter.check_rpm("fresh", limit=5)

    assert allowed is True
    assert list(limiter._request_windows) == ["client-2", "client-3", "client-4", "fresh"]


def test_capacity_rejects_new_key_then_reuses_one_expired_slot():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(
        max_keys=3,
        cleanup_interval=100,
        cleanup_batch_size=10,
        clock=clock,
    )

    for key in ("a", "b", "c"):
        assert limiter.check_rpm(key, limit=5)[0] is True

    allowed, count, retry_after = limiter.check_rpm("d", limit=5)
    assert allowed is False
    assert count == 5
    assert retry_after > 0
    assert "d" not in limiter._request_windows
    assert len(limiter._request_windows) == 3

    clock.advance(61)
    assert limiter.check_rpm("e", limit=5)[0] is True
    assert len(limiter._request_windows) == 3


def test_rejected_oversized_tpm_request_does_not_retain_empty_key():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(
        max_keys=3,
        cleanup_interval=4,
        cleanup_batch_size=2,
        clock=clock,
    )

    for index in range(100):
        allowed, _, _ = limiter.check_tpm(f"client-{index}", 11, limit=10)
        assert allowed is False

    assert len(limiter._token_windows) == 0


def test_unique_key_cardinality_is_atomic_and_bounded_under_concurrency():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(
        max_keys=10,
        cleanup_interval=1000,
        cleanup_batch_size=1,
        clock=clock,
    )

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(
                lambda index: limiter.check_rpm(f"client-{index}", limit=5),
                range(100),
            )
        )

    assert sum(1 for allowed, _, _ in results if allowed) == 10
    assert sum(1 for allowed, _, _ in results if not allowed) == 90
    assert len(limiter._request_windows) == 10


def test_rpm_and_tpm_cleanup_are_independent():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(
        max_keys=4,
        cleanup_interval=3,
        cleanup_batch_size=2,
        clock=clock,
    )

    assert limiter.check_tpm("token-a", 1, limit=10)[0] is True
    assert limiter.check_tpm("token-b", 1, limit=10)[0] is True
    assert limiter.check_rpm("rpm-a", limit=5)[0] is True
    assert limiter.check_rpm("rpm-b", limit=5)[0] is True

    clock.advance(61)
    assert limiter.check_rpm("fresh-rpm", limit=5)[0] is True
    assert set(limiter._request_windows) == {"fresh-rpm"}
    assert set(limiter._token_windows) == {"token-a", "token-b"}

    assert limiter.check_tpm("fresh-token", 1, limit=10)[0] is True
    assert set(limiter._token_windows) == {"fresh-token"}


def test_cleanup_uses_each_keys_window_duration():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(
        max_keys=4,
        cleanup_interval=3,
        cleanup_batch_size=2,
        clock=clock,
    )

    assert limiter.check_rpm("long", limit=5, window_sec=120)[0] is True
    assert limiter.check_rpm("short", limit=5, window_sec=10)[0] is True

    clock.advance(11)
    assert limiter.check_rpm("fresh", limit=5)[0] is True
    assert set(limiter._request_windows) == {"long", "fresh"}
