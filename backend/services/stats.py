"""Application-owned dashboard collection and confirmed, short-lived snapshots."""

from __future__ import annotations

import asyncio
import copy
import math
import socket
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from infra.db_executor import run_db
from infra.observability import count_recent, get_consecutive, snapshot as observability_snapshot


def check_service(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        return connection.connect_ex(("localhost", port)) == 0


def _boolean(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    raise ValueError(f"{name} must be a boolean")


class StatsService:
    def __init__(self, *, database, inference_runtime, env: Mapping[str, str]) -> None:
        if inference_runtime is None:
            raise RuntimeError("Statistics require the application's inference runtime")
        self._database = database
        self._runtime = inference_runtime
        self._env = dict(env)
        self._ttl = float(self._env.get("STATS_CACHE_TTL_SECONDS", "5"))
        if not math.isfinite(self._ttl) or self._ttl < 0:
            raise ValueError("STATS_CACHE_TTL_SECONDS must be finite and nonnegative")
        self._cache: dict[str, Any] | None = None
        self._updated_at = 0.0
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def snapshot(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop, self._lock = loop, asyncio.Lock()
            self._cache, self._updated_at = None, 0.0
        if self._cache is not None and time.monotonic() - self._updated_at < self._ttl:
            return copy.deepcopy(self._cache)
        async with self._lock:
            if self._cache is None or time.monotonic() - self._updated_at >= self._ttl:
                payload = await run_db(self.collect)
                self._cache = payload
                self._updated_at = time.monotonic()
            return copy.deepcopy(self._cache)

    def _query(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self._database.execute_sql(query, params)
        if not isinstance(rows, list):
            raise TypeError("Statistics SELECT must return a row list")
        return rows

    def _enabled(self, config: dict[str, Any], env_key: str, key: str, default: bool = True) -> bool:
        if env_key in self._env:
            return _boolean(self._env[env_key], env_key, default)
        return _boolean(config.get(key), key, default)

    def _gateway(self, config: dict[str, Any]) -> dict[str, Any]:
        port = int(self._env.get("ASTRBOT_PORT", "6185"))
        if not 0 < port < 65536:
            raise ValueError("ASTRBOT_PORT must be between 1 and 65535")
        expected = self._enabled(config, "ASTRBOT_ENABLED", "astrbotEnabled")
        running = check_service(port)
        return {
            "name": "AstrBot Gateway",
            "status": "running" if running else ("degraded" if expected else "stopped"),
            "running": running,
            "expected": expected,
            "port": port,
        }

    def _platforms(self, config: dict[str, Any], gateway: dict[str, Any], now: datetime) -> dict[str, dict[str, Any]]:
        settings = {
            "qq": ("ASTRBOT_QQ_ENABLED", "astrbotQQEnabled", True),
            "telegram": ("ASTRBOT_TELEGRAM_ENABLED", "astrbotTelegramEnabled", True),
            "wecom": ("ASTRBOT_WECOM_ENABLED", "astrbotWecomEnabled", True),
            "wechat_official": ("ASTRBOT_WECHAT_OFFICIAL_ENABLED", "astrbotWechatOfficialEnabled", True),
            "wechat_personal": ("ASTRBOT_WECHAT_PERSONAL_ENABLED", "astrbotWechatPersonalEnabled", False),
        }
        rows = self._query(
            'SELECT platform, MAX("createdAt") AS last_event FROM integration_events GROUP BY platform', {}
        )
        latest = {str(row["platform"]): str(row["last_event"] or "") for row in rows}
        stale = int(self._env.get("PLATFORM_STATUS_STALE_SECONDS", "600"))
        if stale < 0:
            raise ValueError("PLATFORM_STATUS_STALE_SECONDS must be nonnegative")
        result = {}
        for platform, (env_key, key, default) in settings.items():
            enabled = gateway["expected"] and self._enabled(config, env_key, key, default)
            last_event = latest.get(platform, "")
            if not enabled:
                status = "disabled"
            elif gateway["status"] == "degraded":
                status = "degraded"
            elif last_event:
                observed = datetime.fromisoformat(last_event)
                reference = now.astimezone() if observed.tzinfo is not None else now
                status = "connected" if (reference - observed).total_seconds() <= stale else "idle"
            else:
                status = "idle"
            result[platform] = {"enabled": enabled, "status": status, "lastEvent": last_event}
        return result

    def _quantiles(self, table: str, count: int, bounds: dict[str, str]) -> tuple[float, float]:
        if count == 0:
            return 0.0, 0.0
        p95, p99 = (round((count - 1) * fraction) + 1 for fraction in (0.95, 0.99))
        rows = self._query(
            f"SELECT latency, position FROM ("
            f'SELECT COALESCE("costTime", 0) AS latency, '
            f'ROW_NUMBER() OVER (ORDER BY COALESCE("costTime", 0)) AS position '
            f'FROM {table} WHERE "createdAt" >= :start AND "createdAt" < :end'
            f") AS ranked WHERE position IN (:p95, :p99)",
            {**bounds, "p95": p95, "p99": p99},
        )
        values = {int(row["position"]): float(row["latency"]) for row in rows}
        return round(values[p95], 2), round(values[p99], 2)

    def collect(self) -> dict[str, Any]:
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        bounds = {"start": start.isoformat(), "end": (start + timedelta(days=1)).isoformat()}
        (messages,) = self._query(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN reply IS NOT NULL AND reply <> '' THEN 1 ELSE 0 END), 0) AS replies, "
            'COALESCE(AVG(COALESCE("costTime", 0)), 0) AS average, '
            "COUNT(DISTINCT NULLIF(\"sessionId\", '')) AS sessions, "
            'COALESCE(MIN("costTime"), 0) AS minimum, COALESCE(MAX("costTime"), 0) AS maximum '
            'FROM messages WHERE "createdAt" >= :start AND "createdAt" < :end',
            bounds,
        )
        (invocations,) = self._query(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN TRIM(COALESCE(\"errorType\", '')) <> '' THEN 1 ELSE 0 END), 0) AS failures, "
            'COALESCE(SUM(CASE WHEN "usedRag" = 1 THEN 1 ELSE 0 END), 0) AS used_rag, '
            'COALESCE(MIN("costTime"), 0) AS minimum, COALESCE(MAX("costTime"), 0) AS maximum '
            'FROM model_invocations WHERE "createdAt" >= :start AND "createdAt" < :end',
            bounds,
        )
        costs = [float(row[key]) for row in (messages, invocations) for key in ("minimum", "maximum")]
        if any(not math.isfinite(value) or value < 0 for value in costs):
            raise ValueError("Stored response latency must be finite and nonnegative")
        invocation_count = int(invocations["total"])
        table, count = (
            ("model_invocations", invocation_count) if invocation_count else ("messages", int(messages["total"]))
        )
        p95, p99 = self._quantiles(table, count, bounds)
        hourly = self._query(
            'SELECT SUBSTR(CAST("createdAt" AS TEXT), 12, 2) AS hour, COUNT(*) AS messages, '
            "SUM(CASE WHEN reply IS NOT NULL AND reply <> '' THEN 1 ELSE 0 END) AS replies "
            'FROM messages WHERE "createdAt" >= :start AND "createdAt" < :end '
            'GROUP BY SUBSTR(CAST("createdAt" AS TEXT), 12, 2)',
            bounds,
        )
        activity = [{"time": f"{hour:02d}:00", "messages": 0, "replies": 0} for hour in range(0, 24, 2)]
        for row in hourly:
            slot = activity[int(row["hour"]) // 2]
            slot["messages"] += int(row["messages"])
            slot["replies"] += int(row["replies"])
        config = self._database.config
        if not isinstance(config, dict):
            raise TypeError("Statistics configuration must be a dictionary")
        gateway = self._gateway(config)
        rag_failures = count_recent("rag_failures", 86400.0)
        rag_total = int(invocations["used_rag"]) + rag_failures
        metrics = {
            "todayMessages": int(messages["total"]),
            "todayReplies": int(messages["replies"]),
            "avgResponseTime": round(float(messages["average"]), 2),
            "p95ResponseTime": p95,
            "p99ResponseTime": p99,
            "modelFailureRate": round(int(invocations["failures"]) / invocation_count, 4) if invocation_count else 0.0,
            "modelFailures": int(invocations["failures"]),
            "modelInvocations": invocation_count,
            "ragFailureRate": round(rag_failures / rag_total, 4) if rag_total else 0.0,
            "ragFailures": rag_failures,
            "activeSessions": int(messages["sessions"]),
            "queue": self._runtime.stats(),
            "astrBotGateway": gateway,
            "platformStatus": self._platforms(config, gateway, now),
            "observability": observability_snapshot(),
        }
        metrics["queueLength"] = metrics["queue"]["queue_size"]
        metrics["queueMaxSize"] = metrics["queue"]["max_queue_size"]
        metrics["currentInferenceConcurrency"] = metrics["queue"]["active"]
        return {"metrics": metrics, "activity": activity}

    def alerts(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        queue_length = int(metrics["queueLength"])
        queue_max = int(metrics["queueMaxSize"])
        queue_ratio = queue_length / queue_max
        response_threshold = float(self._env.get("ALERT_RESPONSE_P95_SECONDS", "30"))
        queue_threshold = float(self._env.get("ALERT_QUEUE_RATIO", "0.8"))
        auth_threshold = int(self._env.get("ALERT_AUTH_FAILURES_5M", "10"))
        db_threshold = int(self._env.get("ALERT_DB_FAILURES_5M", "1"))
        consecutive_threshold = int(self._env.get("ALERT_MODEL_CONSECUTIVE_FAILURES", "3"))
        if get_consecutive("model_failure") >= consecutive_threshold:
            alerts.append(
                {
                    "severity": "critical",
                    "type": "model_consecutive_failures",
                    "message": "Model has failed consecutively",
                    "value": get_consecutive("model_failure"),
                }
            )
        if count_recent("db_write_failures", 300.0) >= db_threshold:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "database_write_failures",
                    "message": "Database write failures detected",
                    "value": count_recent("db_write_failures", 300.0),
                }
            )
        if metrics["astrBotGateway"]["status"] == "degraded":
            alerts.append(
                {
                    "severity": "critical",
                    "type": "astrbot_degraded",
                    "message": "AstrBot gateway is not reachable",
                    "value": metrics["astrBotGateway"],
                }
            )
        if queue_ratio >= queue_threshold:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "message_backlog",
                    "message": "Inference queue is backing up",
                    "value": {"queueLength": queue_length, "queueMaxSize": queue_max},
                }
            )
        if float(metrics["p95ResponseTime"]) >= response_threshold:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "slow_response",
                    "message": "P95 response time exceeded threshold",
                    "value": metrics["p95ResponseTime"],
                }
            )
        auth_failures = count_recent("integration_auth_failures", 300.0)
        if auth_failures >= auth_threshold:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "frequent_auth_failures",
                    "message": "Integration auth failures are frequent",
                    "value": auth_failures,
                }
            )
        return alerts
