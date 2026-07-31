from __future__ import annotations

import json

import httpx
import pytest

from benchmarks.http_load_benchmark import (
    AuthConfig,
    BenchmarkConfigurationError,
    RequestOutcome,
    ScenarioConfig,
    StepResult,
    Thresholds,
    authenticate_client,
    build_parser,
    nearest_rank_percentile,
    parse_concurrency_steps,
    run_step,
)


def test_step_result_separates_rejections_and_uses_success_throughput():
    result = StepResult(
        mode="api",
        concurrency=6,
        duration_s=2.0,
        outcomes=[
            RequestOutcome(True, 200, 10.0),
            RequestOutcome(True, 204, 30.0),
            RequestOutcome(False, 429, 5.0, "limited"),
            RequestOutcome(False, 503, 8.0, "queue full"),
            RequestOutcome(False, 500, 7.0, "server error"),
            RequestOutcome(False, 0, 100.0, "timeout"),
        ],
    )

    assert result.total_count == 6
    assert result.success_count == 2
    assert result.rate_limited_count == 1
    assert result.unavailable_count == 1
    assert result.other_failure_count == 2
    assert result.attempted_rps == pytest.approx(3.0)
    assert result.success_rps == pytest.approx(1.0)
    assert result.p50_ms == 10.0
    assert result.p95_ms == 30.0
    assert result.p99_ms == 30.0
    assert result.status_counts == {0: 1, 200: 1, 204: 1, 429: 1, 500: 1, 503: 1}


def test_nearest_rank_percentile_and_concurrency_parser():
    assert nearest_rank_percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert nearest_rank_percentile([], 0.95) == 0
    assert parse_concurrency_steps("1,5,20", "api") == [1, 5, 20]
    assert parse_concurrency_steps(None, "inference") == [1, 2, 4]

    with pytest.raises(BenchmarkConfigurationError):
        parse_concurrency_steps("5,1", "api")
    with pytest.raises(BenchmarkConfigurationError):
        parse_concurrency_steps("1,1", "api")
    with pytest.raises(BenchmarkConfigurationError):
        parse_concurrency_steps("1,nope", "api")


@pytest.mark.asyncio
async def test_inference_step_reuses_client_and_assigns_unique_sessions():
    session_ids: list[str] = []
    authorization_headers: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        session_ids.append(body["sessionId"])
        authorization_headers.append(request.headers.get("Authorization", ""))
        return httpx.Response(
            200,
            json={"reply": "ok", "model": "test", "costTime": 0.01},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        await authenticate_client(client, AuthConfig(bearer_token="jwt-token"))
        result = await run_step(
            client,
            ScenarioConfig(mode="inference"),
            concurrency=4,
            total_requests=12,
            run_id="run-a",
        )

    assert result.success_count == 12
    assert len(session_ids) == 12
    assert len(set(session_ids)) == 12
    assert all(value.startswith("http-load:run-a:4:") for value in session_ids)
    assert authorization_headers == ["Bearer jwt-token"] * 12


@pytest.mark.asyncio
async def test_username_login_promotes_secure_cookie_to_bearer_on_shared_client():
    login_calls = 0
    api_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_calls, api_calls
        if request.url.path == "/api/auth/login":
            login_calls += 1
            assert json.loads(request.content) == {
                "username": "bench-user",
                "password": "secret",
            }
            return httpx.Response(
                200,
                json={"success": True},
                headers={
                    "Set-Cookie": "access_token=login-jwt; Path=/; Secure; HttpOnly"
                },
            )
        if request.url.path == "/api/stats":
            api_calls += 1
            assert request.headers["Authorization"] == "Bearer login-jwt"
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        await authenticate_client(
            client,
            AuthConfig(username="bench-user", password="secret"),
        )
        result = await run_step(
            client,
            ScenarioConfig(mode="api"),
            concurrency=3,
            total_requests=7,
            run_id="run-login",
        )

    assert login_calls == 1
    assert api_calls == 7
    assert result.success_count == 7


@pytest.mark.asyncio
async def test_api_key_is_sent_without_exposing_it_in_results():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "api-key-value"
        return httpx.Response(429, json={"detail": "rate limited"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        await authenticate_client(client, AuthConfig(api_key="api-key-value"))
        result = await run_step(
            client,
            ScenarioConfig(mode="api"),
            concurrency=1,
            total_requests=1,
            run_id="run-key",
        )

    assert result.rate_limited_count == 1
    assert "api-key-value" not in json.dumps(result.to_dict())


def test_thresholds_fail_independently_for_429_503_and_other_errors():
    result = StepResult(
        mode="api",
        concurrency=4,
        duration_s=1.0,
        outcomes=[
            RequestOutcome(True, 200, 10.0),
            RequestOutcome(False, 429, 10.0),
            RequestOutcome(False, 503, 10.0),
            RequestOutcome(False, 500, 10.0),
        ],
    )
    failures = Thresholds(
        min_success_rate=20.0,
        max_rate_limited_rate=0.0,
        max_unavailable_rate=0.0,
        max_other_failure_rate=0.0,
    ).evaluate(result)

    assert any(message.startswith("429 rate") for message in failures)
    assert any(message.startswith("503 rate") for message in failures)
    assert any(message.startswith("other failure rate") for message in failures)


def test_auth_validation_and_safe_default_entrypoint(monkeypatch):
    monkeypatch.delenv("BENCH_URL", raising=False)
    parser = build_parser()
    args = parser.parse_args([])

    assert args.url == "http://localhost"
    assert args.mode == "health"
    assert args.message == "\u8bf7\u7528\u4e00\u53e5\u8bdd\u56de\u590d\u8fd9\u6761\u5e76\u53d1\u538b\u6d4b\u6d88\u606f"
    with pytest.raises(BenchmarkConfigurationError):
        AuthConfig(bearer_token="jwt", api_key="key").validate()
    with pytest.raises(BenchmarkConfigurationError):
        AuthConfig(username="bench-user").validate()
