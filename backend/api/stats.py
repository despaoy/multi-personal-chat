"""System statistics, service status, and lightweight alert snapshots."""

from __future__ import annotations

import asyncio
import copy
import os
import socket
import subprocess
import time
from datetime import datetime
from typing import Any

import psutil
from fastapi import APIRouter, Depends

from app.config import service_start_time
from app.dependencies import get_current_admin
from db.adapter import db
from db.schemas import StatsResponse
from infra.concurrency_control import inference_runtime
from infra.observability import count_recent, get_consecutive, snapshot as observability_snapshot

router = APIRouter(prefix="/api/stats", tags=["stats"])

# Dashboard hooks poll several stats endpoints at roughly the same time. The
# underlying payload currently performs multiple synchronous database queries
# and may inspect up to 10k rows from both messages and model_invocations. Keep
# the synchronous builder for compatibility, but refresh it once per short
# window on a worker thread so concurrent polls neither block the event loop nor
# stampede the database.
_METRICS_CACHE_TTL_SECONDS = max(
    0.0, float(os.getenv("STATS_CACHE_TTL_SECONDS", "5"))
)
_metrics_cache: dict[str, Any] | None = None
_metrics_cache_updated_at = 0.0
_metrics_cache_lock: asyncio.Lock | None = None
_metrics_cache_lock_loop: asyncio.AbstractEventLoop | None = None
_RESOURCE_CACHE_TTL_SECONDS = max(
    0.0, float(os.getenv("STATS_RESOURCE_CACHE_TTL_SECONDS", "15"))
)
_resource_cache: dict[str, Any] | None = None
_resource_cache_updated_at = 0.0
_resource_cache_lock: asyncio.Lock | None = None
_resource_cache_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_metrics_cache_lock() -> asyncio.Lock:
    global _metrics_cache_lock, _metrics_cache_lock_loop
    global _metrics_cache, _metrics_cache_updated_at
    loop = asyncio.get_running_loop()
    if _metrics_cache_lock is None or _metrics_cache_lock_loop is not loop:
        _metrics_cache_lock = asyncio.Lock()
        _metrics_cache_lock_loop = loop
        _metrics_cache = None
        _metrics_cache_updated_at = 0.0
    return _metrics_cache_lock


def _get_resource_cache_lock() -> asyncio.Lock:
    global _resource_cache_lock, _resource_cache_lock_loop
    global _resource_cache, _resource_cache_updated_at
    loop = asyncio.get_running_loop()
    if _resource_cache_lock is None or _resource_cache_lock_loop is not loop:
        _resource_cache_lock = asyncio.Lock()
        _resource_cache_lock_loop = loop
        _resource_cache = None
        _resource_cache_updated_at = 0.0
    return _resource_cache_lock


def _reset_metrics_cache() -> None:
    """Clear the process-local snapshot cache (primarily for tests)."""
    global _metrics_cache, _metrics_cache_updated_at, _metrics_cache_lock
    global _metrics_cache_lock_loop
    global _resource_cache, _resource_cache_updated_at, _resource_cache_lock
    global _resource_cache_lock_loop
    _metrics_cache = None
    _metrics_cache_updated_at = 0.0
    _metrics_cache_lock = None
    _metrics_cache_lock_loop = None
    _resource_cache = None
    _resource_cache_updated_at = 0.0
    _resource_cache_lock = None
    _resource_cache_lock_loop = None


async def _metrics_snapshot() -> dict[str, Any]:
    """Return a copy of a single-flight, short-lived metrics snapshot."""
    global _metrics_cache, _metrics_cache_updated_at

    cache_lock = _get_metrics_cache_lock()
    now = time.monotonic()
    if (
        _metrics_cache is not None
        and now - _metrics_cache_updated_at < _METRICS_CACHE_TTL_SECONDS
    ):
        return copy.deepcopy(_metrics_cache)

    async with cache_lock:
        now = time.monotonic()
        if (
            _metrics_cache is None
            or now - _metrics_cache_updated_at >= _METRICS_CACHE_TTL_SECONDS
        ):
            _metrics_cache = await asyncio.to_thread(_metrics_payload)
            _metrics_cache_updated_at = time.monotonic()
        return copy.deepcopy(_metrics_cache)


async def _resource_snapshot() -> dict[str, Any]:
    """Cache expensive host/GPU probes and collapse concurrent refreshes."""
    global _resource_cache, _resource_cache_updated_at

    cache_lock = _get_resource_cache_lock()
    now = time.monotonic()
    if (
        _resource_cache is not None
        and now - _resource_cache_updated_at < _RESOURCE_CACHE_TTL_SECONDS
    ):
        return copy.deepcopy(_resource_cache)

    async with cache_lock:
        now = time.monotonic()
        if (
            _resource_cache is None
            or now - _resource_cache_updated_at >= _RESOURCE_CACHE_TTL_SECONDS
        ):
            system, gpu = await asyncio.gather(
                asyncio.to_thread(get_system_stats), asyncio.to_thread(get_gpu_stats)
            )
            _resource_cache = {"system": system, "gpu": gpu}
            _resource_cache_updated_at = time.monotonic()
        return copy.deepcopy(_resource_cache)



def get_gpu_stats():
    gpu_used = 0.0
    gpu_total = 0.0
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = round(mem_info.used / (1024 ** 3), 1)
            gpu_total = round(mem_info.total / (1024 ** 3), 1)
        pynvml.nvmlShutdown()
        if gpu_total > 0:
            return {"gpu_used": gpu_used, "gpu_total": gpu_total}
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                gpu_total = round(float(parts[0].strip()) / 1024, 1)
                gpu_used = round(float(parts[1].strip()) / 1024, 1)
                return {"gpu_used": gpu_used, "gpu_total": gpu_total}
    except Exception:
        pass

    return {"gpu_used": 0.0, "gpu_total": 0.0}


def get_system_stats():
    cpu_usage = int(psutil.cpu_percent(interval=None))
    memory = psutil.virtual_memory()
    memory_used = round(memory.used / (1024 ** 3), 1)
    memory_total = round(memory.total / (1024 ** 3), 1)

    import platform as _platform
    try:
        disk_path = "C:" if _platform.system() == "Windows" else "/"
        disk = psutil.disk_usage(disk_path)
        disk_used = round(disk.used / (1024 ** 3), 1)
        disk_total = round(disk.total / (1024 ** 3), 1)
    except Exception:
        disk_used = 0.0
        disk_total = 0.0

    return {
        "cpu_usage": cpu_usage,
        "memory_used": memory_used,
        "memory_total": memory_total,
        "disk_used": disk_used,
        "disk_total": disk_total,
    }


def get_service_uptime():
    delta = datetime.now() - service_start_time
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _config() -> dict[str, Any]:
    try:
        return db.config if isinstance(db.config, dict) else {}
    except Exception:
        return {}


def _config_bool(env_key: str, config_key: str, default: bool = True) -> bool:
    env_value = os.getenv(env_key)
    if env_value is not None:
        return _parse_bool(env_value, default)
    return _parse_bool(_config().get(config_key), default)


def _astrbot_expected_enabled() -> bool:
    return _config_bool("ASTRBOT_ENABLED", "astrbotEnabled", True)


def _platform_enabled_map() -> dict[str, bool]:
    global_enabled = _astrbot_expected_enabled()
    settings = {
        "qq": ("ASTRBOT_QQ_ENABLED", "astrbotQQEnabled", True),
        "telegram": ("ASTRBOT_TELEGRAM_ENABLED", "astrbotTelegramEnabled", True),
        "wecom": ("ASTRBOT_WECOM_ENABLED", "astrbotWecomEnabled", True),
        "wechat_official": ("ASTRBOT_WECHAT_OFFICIAL_ENABLED", "astrbotWechatOfficialEnabled", True),
        "wechat_personal": ("ASTRBOT_WECHAT_PERSONAL_ENABLED", "astrbotWechatPersonalEnabled", False),
    }
    return {
        platform: global_enabled and _config_bool(env_key, config_key, default)
        for platform, (env_key, config_key, default) in settings.items()
    }


def _check_service(port: int) -> bool:
    # C8 fix: 用 with 语句确保 socket 在异常时也被关闭，避免描述符泄漏
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            return result == 0
    except Exception:
        return False


def _today_start_iso() -> str:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _query(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    try:
        result = db.execute_sql(query, params or {})
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _first_number(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not rows:
        return default
    value = rows[0].get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], percentile: float) -> float:
    clean = sorted(v for v in values if isinstance(v, (int, float)) and v >= 0)
    if not clean:
        return 0.0
    index = max(0, min(len(clean) - 1, int(round((len(clean) - 1) * percentile))))
    return round(clean[index], 2)


def _today_message_rows() -> list[dict[str, Any]]:
    return _query('SELECT * FROM messages WHERE "createdAt" >= :start ORDER BY "createdAt" DESC LIMIT 10000', {"start": _today_start_iso()})


def _today_invocation_rows() -> list[dict[str, Any]]:
    return _query('SELECT * FROM model_invocations WHERE "createdAt" >= :start ORDER BY "createdAt" DESC LIMIT 10000', {"start": _today_start_iso()})


def _latest_integration_events() -> dict[str, str]:
    rows = _query('SELECT platform, MAX("createdAt") as lastEvent FROM integration_events GROUP BY platform')
    return {str(row.get("platform")): str(row.get("lastEvent") or "") for row in rows}


def _is_recent_iso(value: str, seconds: int) -> bool:
    if not value:
        return False
    try:
        return (datetime.now() - datetime.fromisoformat(value)).total_seconds() <= seconds
    except (TypeError, ValueError):
        return False


def _astrbot_service_status() -> dict[str, Any]:
    astrbot_port = int(os.getenv("ASTRBOT_PORT", "6185"))
    running = _check_service(astrbot_port)
    expected = _astrbot_expected_enabled()
    status = "running" if running else ("degraded" if expected else "stopped")
    return {"name": "AstrBot Gateway", "status": status, "running": running, "expected": expected, "port": astrbot_port}


def _platform_status() -> dict[str, dict[str, Any]]:
    enabled = _platform_enabled_map()
    latest = _latest_integration_events()
    stale_seconds = int(os.getenv("PLATFORM_STATUS_STALE_SECONDS", "600"))
    astrbot = _astrbot_service_status()
    result: dict[str, dict[str, Any]] = {}
    for platform, is_enabled in enabled.items():
        last_event = latest.get(platform, "")
        if not is_enabled:
            status = "disabled"
        elif astrbot["status"] == "degraded":
            status = "degraded"
        elif _is_recent_iso(last_event, stale_seconds):
            status = "connected"
        else:
            status = "idle"
        result[platform] = {"enabled": is_enabled, "status": status, "lastEvent": last_event}
    return result


def _metrics_payload() -> dict[str, Any]:
    messages = _today_message_rows()
    invocations = _today_invocation_rows()
    costs = [float(row.get("costTime") or 0.0) for row in messages]
    invocation_costs = [float(row.get("costTime") or 0.0) for row in invocations]
    all_costs = invocation_costs or costs
    today_messages = len(messages)
    today_replies = sum(1 for row in messages if row.get("reply"))
    avg_response = round(sum(costs) / len(costs), 2) if costs else 0.0
    model_failures = sum(1 for row in invocations if str(row.get("errorType") or "").strip())
    model_failure_rate = round(model_failures / len(invocations), 4) if invocations else 0.0
    used_rag = sum(1 for row in invocations if int(row.get("usedRag") or 0) == 1)
    rag_failures = count_recent("rag_failures", 86400.0)
    rag_total = used_rag + rag_failures
    rag_failure_rate = round(rag_failures / rag_total, 4) if rag_total else 0.0
    queue_stats = inference_runtime.stats()

    return {
        "todayMessages": today_messages,
        "todayReplies": today_replies,
        "avgResponseTime": avg_response,
        "p95ResponseTime": _percentile(all_costs, 0.95),
        "p99ResponseTime": _percentile(all_costs, 0.99),
        "modelFailureRate": model_failure_rate,
        "modelFailures": model_failures,
        "modelInvocations": len(invocations),
        "ragFailureRate": rag_failure_rate,
        "ragFailures": rag_failures,
        "activeSessions": len({row.get("sessionId") for row in messages if row.get("sessionId")}),
        "queueLength": queue_stats.get("queue_size", 0),
        "queueMaxSize": queue_stats.get("max_queue_size", 0),
        "currentInferenceConcurrency": queue_stats.get("active", 0),
        "queue": queue_stats,
        "astrBotGateway": _astrbot_service_status(),
        "platformStatus": _platform_status(),
        "observability": observability_snapshot(),
    }


def _alerts_from_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    queue_length = int(metrics.get("queueLength") or 0)
    queue_max = max(int(metrics.get("queueMaxSize") or 1), 1)
    queue_ratio = queue_length / queue_max
    response_threshold = float(os.getenv("ALERT_RESPONSE_P95_SECONDS", "30"))
    queue_threshold = float(os.getenv("ALERT_QUEUE_RATIO", "0.8"))
    auth_threshold = int(os.getenv("ALERT_AUTH_FAILURES_5M", "10"))
    db_threshold = int(os.getenv("ALERT_DB_FAILURES_5M", "1"))
    consecutive_threshold = int(os.getenv("ALERT_MODEL_CONSECUTIVE_FAILURES", "3"))

    if get_consecutive("model_failure") >= consecutive_threshold:
        alerts.append({"severity": "critical", "type": "model_consecutive_failures", "message": "Model has failed consecutively", "value": get_consecutive("model_failure")})
    if count_recent("db_write_failures", 300.0) >= db_threshold:
        alerts.append({"severity": "warning", "type": "database_write_failures", "message": "Database write failures detected", "value": count_recent("db_write_failures", 300.0)})
    if metrics.get("astrBotGateway", {}).get("status") == "degraded":
        alerts.append({"severity": "critical", "type": "astrbot_degraded", "message": "AstrBot gateway is not reachable", "value": metrics.get("astrBotGateway")})
    if queue_ratio >= queue_threshold:
        alerts.append({"severity": "warning", "type": "message_backlog", "message": "Inference queue is backing up", "value": {"queueLength": queue_length, "queueMaxSize": queue_max}})
    if float(metrics.get("p95ResponseTime") or 0.0) >= response_threshold:
        alerts.append({"severity": "warning", "type": "slow_response", "message": "P95 response time exceeded threshold", "value": metrics.get("p95ResponseTime")})
    auth_failures = count_recent("integration_auth_failures", 300.0)
    if auth_failures >= auth_threshold:
        alerts.append({"severity": "warning", "type": "frequent_auth_failures", "message": "Integration auth failures are frequent", "value": auth_failures})
    return alerts


@router.get("", response_model=StatsResponse)
async def get_stats(current_user: dict = Depends(get_current_admin)):
    metrics, resources = await asyncio.gather(_metrics_snapshot(), _resource_snapshot())
    sys_stats = resources["system"]
    gpu_stats = resources["gpu"]
    return StatsResponse(
        todayMessages=metrics["todayMessages"],
        todayReplies=metrics["todayReplies"],
        avgResponseTime=metrics["avgResponseTime"],
        p95ResponseTime=metrics["p95ResponseTime"],
        p99ResponseTime=metrics["p99ResponseTime"],
        modelFailureRate=metrics["modelFailureRate"],
        ragFailureRate=metrics["ragFailureRate"],
        activeSessions=metrics["activeSessions"],
        modelLoad=min(100, int(metrics["todayReplies"])),
        cpuUsage=sys_stats["cpu_usage"],
        gpuMemory={"used": gpu_stats["gpu_used"], "total": gpu_stats["gpu_total"]},
        memoryUsage={"used": sys_stats["memory_used"], "total": sys_stats["memory_total"]},
        diskUsage={"used": sys_stats["disk_used"], "total": sys_stats["disk_total"]},
        queueLength=metrics["queueLength"],
        currentInferenceConcurrency=metrics["currentInferenceConcurrency"],
        astrBotGateway=metrics["astrBotGateway"],
        platformStatus=metrics["platformStatus"],
    )


@router.get("/metrics")
async def get_metrics(current_user: dict = Depends(get_current_admin)):
    # C16 fix: 含 auth_failures 等安全相关指标，限定 admin
    metrics = await _metrics_snapshot()
    metrics["alerts"] = _alerts_from_metrics(metrics)
    return metrics


@router.get("/alerts")
async def get_alerts(current_user: dict = Depends(get_current_admin)):
    # C16 fix: 告警含认证失败频率等安全信息，限定 admin
    metrics = await _metrics_snapshot()
    alerts = _alerts_from_metrics(metrics)
    return {"alerts": alerts, "count": len(alerts), "metrics": metrics}


@router.get("/activity")
async def get_activity(current_user: dict = Depends(get_current_admin)):
    # C16 fix: 消息活动分布可暴露使用模式，限定 admin
    hours = [f"{h:02d}:00" for h in range(0, 24, 2)]
    activity_map: dict[int, dict[str, int]] = {int(h.split(":")[0]): {"messages": 0, "replies": 0} for h in hours}
    for msg in await asyncio.to_thread(_today_message_rows):
        try:
            msg_time = datetime.fromisoformat(msg["createdAt"])
        except (ValueError, KeyError, TypeError):
            continue
        slot = activity_map.get(msg_time.hour)
        if slot is None:
            continue
        slot["messages"] += 1
        if msg.get("reply"):
            slot["replies"] += 1
    return {"activity": [{"time": h, **activity_map[int(h.split(':')[0])]} for h in hours]}


@router.get("/services")
async def get_services(current_user: dict = Depends(get_current_admin)):
    # C16 fix: 服务状态和 uptime 暴露基础设施拓扑，限定 admin
    uptime_str = get_service_uptime()
    astrbot = await asyncio.to_thread(_astrbot_service_status)
    queue_stats = inference_runtime.stats()
    queue_ratio = queue_stats["queue_size"] / max(queue_stats["max_queue_size"], 1)
    queue_status = "degraded" if queue_ratio >= 0.8 else "running"

    services = [
        {"name": "Backend API", "status": "running", "uptime": uptime_str},
        {"name": "Inference Queue", "status": queue_status, "uptime": uptime_str, "stats": queue_stats},
        {"name": "AstrBot Gateway", "status": astrbot["status"], "uptime": uptime_str if astrbot["running"] else "-", "port": astrbot["port"]},
        {"name": "Model Service", "status": "running", "uptime": uptime_str},
        {"name": "In-Memory DB", "status": "running", "uptime": uptime_str},
    ]
    return {"services": services}
