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
        assert body["message"] == ScenarioConfig(mode="inference").message
        session_ids.append(body["sessionId"])
        authorization_headers.append(request.headers.get("Authorization", ""))
        return httpx.Response(
            200,
            json={"reply": "ok", "model": "test", "costTime": 0.01, "modelInvoked": True},
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
                headers={"Set-Cookie": "access_token=login-jwt; Path=/; Secure; HttpOnly"},
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes,error",
    [
        ({"modelInvoked": False, "warnings": ["cached_model_response"]}, "did not invoke"),
        ({"modelInvoked": False, "model": "memory/lookup"}, "did not invoke"),
        ({"modelInvoked": "true"}, "invalid inference"),
        ({"model": None}, "invalid inference"),
        ({"reply": "   "}, "invalid inference"),
        ({"costTime": float("inf")}, "invalid inference"),
    ],
    ids=["cached", "non-model", "invalid-invocation", "invalid-model", "blank-reply", "invalid-cost"],
)
async def test_inference_statistics_require_complete_fresh_model_response(changes, error):
    body = {"reply": "完整合成回复", "model": "synthetic-model", "costTime": 0.125, "modelInvoked": True}
    body.update(changes)

    def handler(request):
        return httpx.Response(200, content=json.dumps(body), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost") as client:
        result = await run_step(
            client, ScenarioConfig(mode="inference"), concurrency=1, total_requests=1, run_id="truth"
        )

    assert result.success_count == 0 and result.other_failure_count == 1
    assert result.success_rps == 0 and result.status_counts == {200: 1}
    assert error in result.outcomes[0].error
    assert Thresholds().evaluate(result)


@pytest.mark.asyncio
async def test_cached_warmup_stops_without_measured_load():
    from benchmarks.http_load_benchmark import BenchmarkPreflightError, warm_up

    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "reply": "完整缓存回复",
                "model": "synthetic-model",
                "costTime": 0.01,
                "modelInvoked": False,
                "warnings": ["cached_model_response"],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost") as client:
        with pytest.raises(BenchmarkPreflightError, match="did not invoke"):
            await warm_up(
                client, ScenarioConfig(mode="inference", message="只报告绿筹码；不要统计红筹码。"), 3, "warmup"
            )

    assert len(calls) == 1
    assert calls[0]["message"] == "只报告绿筹码；不要统计红筹码。"


@pytest.mark.asyncio
@pytest.mark.parametrize("invoked,expected", [(True, 0), (False, 1)], ids=["fresh", "cached"])
async def test_cli_result_reports_actual_invocation_and_preserves_input(monkeypatch, tmp_path, invoked, expected):
    from benchmarks import http_load_benchmark as benchmark

    real_client = httpx.AsyncClient
    clients, messages = [], []
    message = "桌上原有4枚绿筹码、2枚红筹码；移走1枚绿筹码后，只问绿筹码还剩几枚，不计算红筹码。"

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/api/generate"
        assert request.headers["Authorization"] == "Bearer synthetic-benchmark-token"
        messages.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"reply": "还剩3枚绿筹码。", "model": "synthetic-model", "costTime": 0.125, "modelInvoked": invoked},
        )

    def factory(*args, **kwargs):
        client = real_client(*args, **kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(benchmark.httpx, "AsyncClient", factory)
    for key in ("BENCH_API_KEY", "BENCH_USERNAME", "BENCH_PASSWORD", "BENCH_BEARER_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    report = tmp_path / "report.json"
    args = benchmark.build_parser().parse_args(
        [
            "--mode",
            "inference",
            "--url",
            "http://localhost",
            "--bearer-token",
            "synthetic-benchmark-token",
            "--warmup-requests",
            "0",
            "--concurrency-steps",
            "1,2",
            "--requests-per-step",
            "2",
            "--step-delay",
            "0",
            "--message",
            message,
            "--json-out",
            str(report),
        ]
    )
    code, results = await benchmark.run_benchmark(args)
    payload = json.loads(report.read_text())

    assert code == expected
    assert [r.success_count for r in results] == [2 if invoked else 0] * 2
    assert len(messages) == 4 and all(m["message"] == message for m in messages)
    assert len({m["sessionId"] for m in messages}) == 4
    assert len({m["traceId"] for m in messages}) == 4
    assert len(clients) == 1 and clients[0].is_closed
    assert [s["success"] for s in payload["steps"]] == [2 if invoked else 0] * 2
    assert bool(payload["gate_failures"]) is (not invoked)
    assert "synthetic-benchmark-token" not in report.read_text()


@pytest.mark.asyncio
async def test_workload_preserves_complete_group_requests_and_observed_models(monkeypatch, tmp_path):
    from benchmarks import http_load_benchmark as benchmark

    requests = [
        {
            "message": "盒中有4枚绿筹码、2枚红筹码。移走1枚绿筹码，只统计剩余绿筹码，不统计红筹码。",
            "sessionType": "group",
            "sessionId": "synthetic-group-a",
            "userId": "synthetic-user-a",
            "loraName": "synthetic-lora",
            "history": [
                {"role": "user", "content": "本题的盒子仅用于筹码计数。"},
                {"role": "assistant", "content": "我会只按给定筹码计数。"},
            ],
        },
        {
            "message": "今天下午才改为喝温水，上午仍喝茶；只复述今天下午的选择，不推断长期偏好。",
            "sessionType": "group",
            "sessionId": "synthetic-group-b",
            "userId": "synthetic-user-b",
            "loraName": "default",
        },
    ]
    source = tmp_path / "complete-workload.json"
    source.write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
    report = tmp_path / "report.json"
    sent, clients = [], []
    real_client = httpx.AsyncClient

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.headers["Authorization"] == "Bearer synthetic-workload-token"
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(
            200,
            json={
                "reply": "完整合成回复。",
                "model": "vllm/synthetic-" + body["loraName"],
                "costTime": 0.125,
                "modelInvoked": True,
            },
        )

    def factory(*args, **kwargs):
        client = real_client(*args, **kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(benchmark.httpx, "AsyncClient", factory)
    for key in ("BENCH_API_KEY", "BENCH_USERNAME", "BENCH_PASSWORD", "BENCH_BEARER_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    args = benchmark.build_parser().parse_args(
        [
            "--mode",
            "inference",
            "--url",
            "http://localhost",
            "--bearer-token",
            "synthetic-workload-token",
            "--workload",
            str(source),
            "--warmup-requests",
            "0",
            "--concurrency-steps",
            "1,2",
            "--requests-per-step",
            "4",
            "--step-delay",
            "0",
            "--json-out",
            str(report),
        ]
    )
    code, results = await benchmark.run_benchmark(args)

    assert code == 0 and [r.success_count for r in results] == [4, 4]
    assert len(sent) == 8
    for index, body in enumerate(sent):
        original = requests[index % len(requests)]
        assert {k: body[k] for k in original} == original
    assert len({body["traceId"] for body in sent}) == 8
    assert len(clients) == 1 and clients[0].is_closed
    payload = json.loads(report.read_text())
    assert [s["model_counts"] for s in payload["steps"]] == [
        {"vllm/synthetic-synthetic-lora": 2, "vllm/synthetic-default": 2}
    ] * 2
    assert source.read_text(encoding="utf-8") == json.dumps(requests, ensure_ascii=False)
    assert "synthetic-workload-token" not in report.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["{", "[]", '[{"history": []}]', '[{"message": "   "}]'],
    ids=["invalid-json", "empty", "missing-message", "blank-message"],
)
async def test_invalid_workload_stops_before_network(monkeypatch, tmp_path, capsys, content):
    from benchmarks import http_load_benchmark as benchmark

    source = tmp_path / "invalid-workload.json"
    source.write_text(content, encoding="utf-8")

    def no_client(*args, **kwargs):
        raise AssertionError("invalid input must not allocate a client or send load")

    monkeypatch.setattr(benchmark.httpx, "AsyncClient", no_client)
    for key in ("BENCH_API_KEY", "BENCH_USERNAME", "BENCH_PASSWORD", "BENCH_BEARER_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    assert (
        await benchmark.async_main(
            ["--mode", "inference", "--bearer-token", "synthetic-workload-token", "--workload", str(source)]
        )
        == 2
    )
    assert "workload" in capsys.readouterr().out.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [["--mode", "api"], ["--mode", "inference", "--lora-name", "another-lora"]],
    ids=["wrong-mode", "conflicting-lora"],
)
async def test_workload_configuration_has_one_source_before_network(monkeypatch, tmp_path, options):
    from benchmarks import http_load_benchmark as benchmark

    source = tmp_path / "valid-workload.json"
    source.write_text('[{"message": "只回复已收到，不增加其他内容。", "loraName": "synthetic-lora"}]', encoding="utf-8")

    def no_client(*args, **kwargs):
        raise AssertionError("conflicting configuration must not create a client")

    monkeypatch.setattr(benchmark.httpx, "AsyncClient", no_client)
    for key in ("BENCH_API_KEY", "BENCH_USERNAME", "BENCH_PASSWORD", "BENCH_BEARER_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    assert (
        await benchmark.async_main([*options, "--bearer-token", "synthetic-workload-token", "--workload", str(source)])
        == 2
    )
