#!/usr/bin/env python
"""Authenticated HTTP load benchmark for the production API entrypoint.

The benchmark intentionally owns one ``httpx.AsyncClient`` for login, warm-up,
and every concurrency step. This keeps connection setup out of the measured
request path and prevents the load generator from exhausting ephemeral ports.

Credentials can be supplied with command-line options or, preferably, with:

* ``BENCH_BEARER_TOKEN``
* ``BENCH_API_KEY``
* ``BENCH_USERNAME`` and ``BENCH_PASSWORD``

Examples:

    python -m benchmarks.http_load_benchmark --mode health
    python -m benchmarks.http_load_benchmark --mode api --concurrency 1,10,50
    python -m benchmarks.http_load_benchmark --mode inference --concurrency 1,2,4

Run inference capacity tests against an isolated staging deployment whose
application rate limits are intentionally aligned with the offered load.
Otherwise, 429 responses correctly measure admission control rather than model
capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx


DEFAULT_BASE_URL = "http://localhost"
DEFAULT_CONCURRENCY_STEPS = {
    "health": (1, 10, 50),
    "api": (1, 10, 50),
    "inference": (1, 2, 4),
}


class BenchmarkConfigurationError(ValueError):
    """The benchmark cannot start because its configuration is invalid."""


class BenchmarkPreflightError(RuntimeError):
    """The target is reachable, but not ready for a meaningful benchmark."""


@dataclass(frozen=True)
class AuthConfig:
    bearer_token: str = ""
    api_key: str = ""
    username: str = ""
    password: str = ""

    def validate(self) -> None:
        methods = sum(
            (
                bool(self.bearer_token.strip()),
                bool(self.api_key.strip()),
                bool(self.username.strip() or self.password),
            )
        )
        if methods > 1:
            raise BenchmarkConfigurationError(
                "Choose exactly one authentication method: bearer token, API key, or username/password."
            )
        if bool(self.username.strip()) != bool(self.password):
            raise BenchmarkConfigurationError(
                "Username and password must be provided together."
            )

    @property
    def configured(self) -> bool:
        return bool(
            self.bearer_token.strip()
            or self.api_key.strip()
            or (self.username.strip() and self.password)
        )


@dataclass(frozen=True)
class ScenarioConfig:
    mode: str
    api_path: str = "/api/stats"
    message: str = "请用一句话回复这条并发压测消息"
    lora_name: str = ""


@dataclass
class RequestOutcome:
    success: bool
    status_code: int
    latency_ms: float
    error: str = ""
    retry_after: str = ""
    server_cost_ms: float | None = None

    @property
    def category(self) -> str:
        if self.success:
            return "success"
        if self.status_code == 429:
            return "rate_limited"
        if self.status_code == 503:
            return "unavailable"
        return "other_failure"


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    """Return the nearest-rank percentile without fabricating small-sample zeros."""

    if not values:
        return 0.0
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


@dataclass
class StepResult:
    mode: str
    concurrency: int
    duration_s: float
    outcomes: list[RequestOutcome] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.outcomes)

    @property
    def success_count(self) -> int:
        return sum(outcome.category == "success" for outcome in self.outcomes)

    @property
    def rate_limited_count(self) -> int:
        return sum(outcome.category == "rate_limited" for outcome in self.outcomes)

    @property
    def unavailable_count(self) -> int:
        return sum(outcome.category == "unavailable" for outcome in self.outcomes)

    @property
    def other_failure_count(self) -> int:
        return sum(outcome.category == "other_failure" for outcome in self.outcomes)

    @property
    def attempted_rps(self) -> float:
        return self.total_count / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def success_rps(self) -> float:
        """Successful throughput; rejected and failed requests are not included."""

        return self.success_count / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self._rate(self.success_count)

    @property
    def rate_limited_rate(self) -> float:
        return self._rate(self.rate_limited_count)

    @property
    def unavailable_rate(self) -> float:
        return self._rate(self.unavailable_count)

    @property
    def other_failure_rate(self) -> float:
        return self._rate(self.other_failure_count)

    @property
    def successful_latencies_ms(self) -> list[float]:
        return [outcome.latency_ms for outcome in self.outcomes if outcome.success]

    @property
    def p50_ms(self) -> float:
        return nearest_rank_percentile(self.successful_latencies_ms, 0.50)

    @property
    def p95_ms(self) -> float:
        return nearest_rank_percentile(self.successful_latencies_ms, 0.95)

    @property
    def p99_ms(self) -> float:
        return nearest_rank_percentile(self.successful_latencies_ms, 0.99)

    @property
    def average_ms(self) -> float:
        latencies = self.successful_latencies_ms
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def status_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(outcome.status_code for outcome in self.outcomes).items()))

    @property
    def error_counts(self) -> dict[str, int]:
        errors = Counter(outcome.error for outcome in self.outcomes if outcome.error)
        return dict(errors.most_common())

    def _rate(self, count: int) -> float:
        return count / self.total_count * 100 if self.total_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "concurrency": self.concurrency,
            "duration_s": round(self.duration_s, 6),
            "total": self.total_count,
            "success": self.success_count,
            "rate_limited_429": self.rate_limited_count,
            "unavailable_503": self.unavailable_count,
            "other_failures": self.other_failure_count,
            "attempted_rps": round(self.attempted_rps, 4),
            "success_rps": round(self.success_rps, 4),
            "success_rate": round(self.success_rate, 4),
            "rate_limited_rate": round(self.rate_limited_rate, 4),
            "unavailable_rate": round(self.unavailable_rate, 4),
            "other_failure_rate": round(self.other_failure_rate, 4),
            "latency_ms": {
                "average": round(self.average_ms, 3),
                "p50": round(self.p50_ms, 3),
                "p95": round(self.p95_ms, 3),
                "p99": round(self.p99_ms, 3),
            },
            "status_counts": self.status_counts,
            "error_counts": self.error_counts,
        }

    def summary(self) -> str:
        return (
            f"mode={self.mode:<9} concurrency={self.concurrency:>3} "
            f"success={self.success_count}/{self.total_count} ({self.success_rate:5.1f}%) "
            f"429={self.rate_limited_count} 503={self.unavailable_count} "
            f"other={self.other_failure_count} "
            f"success_rps={self.success_rps:7.2f} attempted_rps={self.attempted_rps:7.2f} "
            f"p50={self.p50_ms:8.1f}ms p95={self.p95_ms:8.1f}ms p99={self.p99_ms:8.1f}ms"
        )


@dataclass(frozen=True)
class Thresholds:
    min_success_rate: float = 99.0
    min_success_rps: float = 0.0
    max_p95_ms: float = 0.0
    max_rate_limited_rate: float = 0.0
    max_unavailable_rate: float = 0.0
    max_other_failure_rate: float = 0.0

    def validate(self) -> None:
        rate_values = {
            "min_success_rate": self.min_success_rate,
            "max_rate_limited_rate": self.max_rate_limited_rate,
            "max_unavailable_rate": self.max_unavailable_rate,
            "max_other_failure_rate": self.max_other_failure_rate,
        }
        for name, value in rate_values.items():
            if not 0 <= value <= 100:
                raise BenchmarkConfigurationError(f"{name} must be between 0 and 100.")
        if self.min_success_rps < 0 or self.max_p95_ms < 0:
            raise BenchmarkConfigurationError("Throughput and latency thresholds cannot be negative.")

    def evaluate(self, result: StepResult) -> list[str]:
        failures: list[str] = []
        if result.success_rate < self.min_success_rate:
            failures.append(
                f"success rate {result.success_rate:.2f}% < {self.min_success_rate:.2f}%"
            )
        if self.min_success_rps > 0 and result.success_rps < self.min_success_rps:
            failures.append(
                f"success throughput {result.success_rps:.2f} < {self.min_success_rps:.2f} req/s"
            )
        if (
            self.max_p95_ms > 0
            and result.success_count > 0
            and result.p95_ms > self.max_p95_ms
        ):
            failures.append(f"P95 {result.p95_ms:.1f}ms > {self.max_p95_ms:.1f}ms")
        if result.rate_limited_rate > self.max_rate_limited_rate:
            failures.append(
                f"429 rate {result.rate_limited_rate:.2f}% > {self.max_rate_limited_rate:.2f}%"
            )
        if result.unavailable_rate > self.max_unavailable_rate:
            failures.append(
                f"503 rate {result.unavailable_rate:.2f}% > {self.max_unavailable_rate:.2f}%"
            )
        if result.other_failure_rate > self.max_other_failure_rate:
            failures.append(
                f"other failure rate {result.other_failure_rate:.2f}% "
                f"> {self.max_other_failure_rate:.2f}%"
            )
        return failures


def parse_concurrency_steps(raw: str | None, mode: str) -> list[int]:
    if raw is None:
        return list(DEFAULT_CONCURRENCY_STEPS[mode])
    try:
        steps = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise BenchmarkConfigurationError(
            "Concurrency steps must be comma-separated positive integers."
        ) from exc
    if not steps or any(step <= 0 for step in steps):
        raise BenchmarkConfigurationError(
            "Concurrency steps must be comma-separated positive integers."
        )
    if steps != sorted(set(steps)):
        raise BenchmarkConfigurationError(
            "Concurrency steps must be unique and strictly increasing."
        )
    return steps


def _response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)[:200]
    except (ValueError, TypeError):
        pass
    return f"HTTP {response.status_code}"


async def authenticate_client(client: httpx.AsyncClient, auth: AuthConfig) -> None:
    """Configure one shared client without logging or returning credentials."""

    auth.validate()
    if auth.bearer_token.strip():
        client.headers["Authorization"] = f"Bearer {auth.bearer_token.strip()}"
        return
    if auth.api_key.strip():
        client.headers["X-API-Key"] = auth.api_key.strip()
        return
    if not auth.username.strip():
        return

    response = await client.post(
        "/api/auth/login",
        json={"username": auth.username.strip(), "password": auth.password},
    )
    if not 200 <= response.status_code < 300:
        raise BenchmarkPreflightError(
            f"Login failed with HTTP {response.status_code}; no load was generated."
        )

    token = response.cookies.get("access_token")
    if not token:
        try:
            body = response.json()
            if isinstance(body, dict):
                token = body.get("access_token") or body.get("token")
        except (ValueError, TypeError):
            token = None
    if token:
        # Also send a Bearer token so a Secure cookie obtained from an HTTP
        # localhost entrypoint does not silently turn the benchmark into 401s.
        client.headers["Authorization"] = f"Bearer {token}"
    elif not client.cookies:
        raise BenchmarkPreflightError(
            "Login succeeded but returned no reusable authentication credential."
        )


async def _request_once(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    require_inference_reply: bool = False,
) -> RequestOutcome:
    started_at = time.monotonic()
    try:
        response = await client.request(method, path, json=json_body)
    except httpx.TimeoutException as exc:
        return RequestOutcome(
            success=False,
            status_code=0,
            latency_ms=(time.monotonic() - started_at) * 1000,
            error=f"timeout:{type(exc).__name__}",
        )
    except httpx.HTTPError as exc:
        return RequestOutcome(
            success=False,
            status_code=0,
            latency_ms=(time.monotonic() - started_at) * 1000,
            error=f"transport:{type(exc).__name__}",
        )

    latency_ms = (time.monotonic() - started_at) * 1000
    success = 200 <= response.status_code < 300
    error = "" if success else _response_error(response)
    server_cost_ms: float | None = None

    if success and require_inference_reply:
        try:
            payload = response.json()
        except ValueError:
            success = False
            error = "invalid inference JSON response"
        else:
            reply = payload.get("reply") if isinstance(payload, dict) else None
            if not isinstance(reply, str) or not reply.strip():
                success = False
                error = "inference response has no non-empty reply"
            cost_time = payload.get("costTime") if isinstance(payload, dict) else None
            if isinstance(cost_time, (int, float)):
                server_cost_ms = float(cost_time) * 1000

    return RequestOutcome(
        success=success,
        status_code=response.status_code,
        latency_ms=latency_ms,
        error=error,
        retry_after=response.headers.get("Retry-After", ""),
        server_cost_ms=server_cost_ms,
    )


async def issue_scenario_request(
    client: httpx.AsyncClient,
    scenario: ScenarioConfig,
    request_id: str,
) -> RequestOutcome:
    if scenario.mode == "health":
        return await _request_once(client, "GET", "/health")
    if scenario.mode == "api":
        return await _request_once(client, "GET", scenario.api_path)
    if scenario.mode != "inference":
        raise BenchmarkConfigurationError(f"Unsupported mode: {scenario.mode}")

    payload: dict[str, Any] = {
        "message": f"{scenario.message} [{request_id}]",
        "sessionType": "private",
        "sessionId": f"http-load:{request_id}",
        "traceId": request_id.replace(":", "-"),
        "platform": "benchmark",
    }
    if scenario.lora_name:
        payload["loraName"] = scenario.lora_name
    return await _request_once(
        client,
        "POST",
        "/api/generate",
        json_body=payload,
        require_inference_reply=True,
    )


async def run_step(
    client: httpx.AsyncClient,
    scenario: ScenarioConfig,
    *,
    concurrency: int,
    total_requests: int,
    run_id: str,
) -> StepResult:
    if concurrency <= 0 or total_requests <= 0:
        raise BenchmarkConfigurationError("Concurrency and request count must be positive.")

    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(index: int) -> RequestOutcome:
        async with semaphore:
            request_id = f"{run_id}:{concurrency}:{index}"
            return await issue_scenario_request(client, scenario, request_id)

    started_at = time.monotonic()
    outcomes = await asyncio.gather(
        *(limited_request(index) for index in range(total_requests))
    )
    return StepResult(
        mode=scenario.mode,
        concurrency=concurrency,
        duration_s=time.monotonic() - started_at,
        outcomes=list(outcomes),
    )


async def warm_up(
    client: httpx.AsyncClient,
    scenario: ScenarioConfig,
    count: int,
    run_id: str,
) -> None:
    for index in range(count):
        outcome = await issue_scenario_request(
            client,
            scenario,
            f"{run_id}:warmup:{index}",
        )
        if not outcome.success:
            detail = outcome.error or outcome.category
            raise BenchmarkPreflightError(
                f"{scenario.mode} warm-up failed with HTTP {outcome.status_code}: {detail}. "
                "No measured load was generated."
            )


def default_request_count(mode: str, concurrency: int) -> int:
    if mode == "inference":
        return max(6, concurrency * 3)
    return max(20, concurrency * 5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated, connection-reusing HTTP load benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=("health", "api", "inference"),
        default="health",
        help="Scenario to run (default: health).",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("BENCH_URL", DEFAULT_BASE_URL),
        help=f"Production API entrypoint (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--concurrency-steps",
        "--concurrency",
        dest="concurrency_steps",
        help="Strictly increasing comma-separated steps; mode-specific defaults are used when omitted.",
    )
    parser.add_argument(
        "--requests-per-step",
        type=int,
        default=0,
        help="Exact requests per step; 0 uses a safe mode-specific default.",
    )
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--step-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-path", default="/api/stats")
    parser.add_argument("--message", default="请用一句话回复这条并发压测消息")
    parser.add_argument("--lora-name", default="")

    parser.add_argument(
        "--bearer-token",
        default=os.getenv("BENCH_BEARER_TOKEN", ""),
        help="Bearer JWT; prefer BENCH_BEARER_TOKEN to avoid shell history.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("BENCH_API_KEY", ""),
        help="X-API-Key value; prefer BENCH_API_KEY.",
    )
    parser.add_argument("--username", default=os.getenv("BENCH_USERNAME", ""))
    parser.add_argument(
        "--password",
        default=os.getenv("BENCH_PASSWORD", ""),
        help="Login password; prefer BENCH_PASSWORD.",
    )

    parser.add_argument("--min-success-rate", type=float, default=99.0)
    parser.add_argument("--min-success-rps", type=float, default=0.0)
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        default=0.0,
        help="0 disables the latency gate.",
    )
    parser.add_argument("--max-429-rate", type=float, default=0.0)
    parser.add_argument("--max-503-rate", type=float, default=0.0)
    parser.add_argument("--max-other-failure-rate", type=float, default=0.0)
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path for a machine-readable report.",
    )
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.requests_per_step < 0:
        raise BenchmarkConfigurationError("requests-per-step cannot be negative.")
    if args.warmup_requests < 0:
        raise BenchmarkConfigurationError("warmup-requests cannot be negative.")
    if args.step_delay < 0 or args.timeout <= 0:
        raise BenchmarkConfigurationError("step-delay cannot be negative and timeout must be positive.")
    if not args.api_path.startswith("/"):
        raise BenchmarkConfigurationError("api-path must start with '/'.")


async def run_benchmark(args: argparse.Namespace) -> tuple[int, list[StepResult]]:
    _validate_runtime_args(args)
    steps = parse_concurrency_steps(args.concurrency_steps, args.mode)
    auth = AuthConfig(
        bearer_token=args.bearer_token,
        api_key=args.api_key,
        username=args.username,
        password=args.password,
    )
    auth.validate()
    if args.mode != "health" and not auth.configured:
        raise BenchmarkConfigurationError(
            f"{args.mode} mode targets a protected API; provide benchmark credentials."
        )

    thresholds = Thresholds(
        min_success_rate=args.min_success_rate,
        min_success_rps=args.min_success_rps,
        max_p95_ms=args.max_p95_ms,
        max_rate_limited_rate=args.max_429_rate,
        max_unavailable_rate=args.max_503_rate,
        max_other_failure_rate=args.max_other_failure_rate,
    )
    thresholds.validate()

    scenario = ScenarioConfig(
        mode=args.mode,
        api_path=args.api_path,
        message=args.message,
        lora_name=args.lora_name,
    )
    max_connections = max(steps)
    timeout = httpx.Timeout(
        args.timeout,
        connect=min(10.0, args.timeout),
        pool=args.timeout,
    )
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
        keepalive_expiry=30.0,
    )
    run_id = uuid.uuid4().hex[:12]
    results: list[StepResult] = []
    gate_failures: dict[int, list[str]] = {}

    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/"),
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        headers={"User-Agent": "qqchat-http-load-benchmark/1"},
    ) as client:
        health = await issue_scenario_request(
            client,
            ScenarioConfig(mode="health"),
            f"{run_id}:preflight",
        )
        if not health.success:
            raise BenchmarkPreflightError(
                f"Health preflight failed with HTTP {health.status_code}: "
                f"{health.error or health.category}."
            )

        if args.mode != "health":
            await authenticate_client(client, auth)

        await warm_up(client, scenario, args.warmup_requests, run_id)

        print(f"Target: {args.url.rstrip('/')}")
        print(f"Mode: {args.mode}; concurrency steps: {steps}")
        for index, concurrency in enumerate(steps):
            total_requests = (
                args.requests_per_step
                if args.requests_per_step > 0
                else default_request_count(args.mode, concurrency)
            )
            result = await run_step(
                client,
                scenario,
                concurrency=concurrency,
                total_requests=total_requests,
                run_id=run_id,
            )
            results.append(result)
            failures = thresholds.evaluate(result)
            if failures:
                gate_failures[concurrency] = failures
            print(result.summary())
            for failure in failures:
                print(f"  threshold failure: {failure}")
            if index < len(steps) - 1 and args.step_delay:
                await asyncio.sleep(args.step_delay)

    if args.json_out:
        report = {
            "target": args.url.rstrip("/"),
            "mode": args.mode,
            "concurrency_steps": steps,
            "thresholds": asdict(thresholds),
            "steps": [result.to_dict() for result in results],
            "gate_failures": gate_failures,
        }
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return (1 if gate_failures else 0), results


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code, _ = await run_benchmark(args)
        return exit_code
    except (BenchmarkConfigurationError, BenchmarkPreflightError) as exc:
        print(f"Benchmark aborted: {exc}")
        return 2


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
