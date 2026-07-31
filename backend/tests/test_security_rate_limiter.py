from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from starlette.requests import Request

from middleware import security
from middleware.security import SlidingWindowLimiter


def _request_with_proxy_headers(*headers: tuple[bytes, bytes]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": list(headers),
            "client": ("127.0.0.1", 12345),
            "server": ("localhost", 8000),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_rpm_records_first_request_and_rejects_after_limit():
    limiter = SlidingWindowLimiter()

    results = [limiter.check_rpm("client", limit=3) for _ in range(4)]

    assert [allowed for allowed, _, _ in results] == [True, True, True, False]
    assert [count for _, count, _ in results] == [1, 2, 3, 3]
    assert results[-1][2] > 0


def test_tpm_records_first_request_and_preserves_rpm_window():
    limiter = SlidingWindowLimiter()

    assert limiter.check_tpm("client", token_count=4, limit=10)[:2] == (True, 4)
    assert limiter.check_rpm("client", limit=2)[:2] == (True, 1)
    assert limiter.check_tpm("client", token_count=6, limit=10)[:2] == (True, 10)
    assert limiter.check_tpm("client", token_count=1, limit=10)[:2] == (False, 10)
    assert limiter.check_rpm("client", limit=2)[:2] == (True, 2)
    assert limiter.check_rpm("client", limit=2)[:2] == (False, 2)


def test_rpm_limit_is_atomic_under_concurrent_requests():
    limiter = SlidingWindowLimiter()
    limit = 25

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(
                lambda _: limiter.check_rpm("shared-client", limit=limit),
                range(100),
            )
        )

    assert sum(1 for allowed, _, _ in results if allowed) == limit
    assert sum(1 for allowed, _, _ in results if not allowed) == 100 - limit


def test_proxy_headers_are_ignored_when_trust_is_disabled(monkeypatch):
    request = _request_with_proxy_headers(
        (b"x-forwarded-for", b"203.0.113.8"),
        (b"x-real-ip", b"198.51.100.7"),
    )

    monkeypatch.setattr(security, "TRUST_PROXY_HEADERS", False)

    assert security._get_client_ip(request) == "127.0.0.1"


def test_proxy_headers_are_used_only_when_trust_is_enabled(monkeypatch):
    forwarded = _request_with_proxy_headers(
        (b"x-forwarded-for", b"203.0.113.8, 10.0.0.2"),
    )
    real_ip = _request_with_proxy_headers((b"x-real-ip", b"198.51.100.7"))

    monkeypatch.setattr(security, "TRUST_PROXY_HEADERS", True)

    assert security._get_client_ip(forwarded) == "203.0.113.8"
    assert security._get_client_ip(real_ip) == "198.51.100.7"
