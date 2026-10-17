"""Authenticated statistics APIs; application data and host resources have separate owners."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Any

import psutil
from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import service_start_time
from app.dependencies import get_current_admin
from app.runtime import get_runtime_container
from db.schemas import StatsResponse
from services.stats import StatsService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stats", tags=["stats"])

# Host hardware is shared by the process; business snapshots belong to each app.
_RESOURCE_CACHE_TTL_SECONDS = max(0.0, float(os.getenv("STATS_RESOURCE_CACHE_TTL_SECONDS", "15")))
_resource_cache: dict[str, Any] | None = None
_resource_cache_updated_at = 0.0
_resource_cache_lock: asyncio.Lock | None = None
_resource_cache_lock_loop: asyncio.AbstractEventLoop | None = None


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


async def _resource_snapshot() -> dict[str, Any]:
    """Cache expensive host/GPU probes and collapse concurrent refreshes."""
    global _resource_cache, _resource_cache_updated_at

    cache_lock = _get_resource_cache_lock()
    now = time.monotonic()
    if _resource_cache is not None and now - _resource_cache_updated_at < _RESOURCE_CACHE_TTL_SECONDS:
        return copy.deepcopy(_resource_cache)

    async with cache_lock:
        now = time.monotonic()
        if _resource_cache is None or now - _resource_cache_updated_at >= _RESOURCE_CACHE_TTL_SECONDS:
            system, gpu = await asyncio.gather(asyncio.to_thread(get_system_stats), asyncio.to_thread(get_gpu_stats))
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
            gpu_used = round(mem_info.used / (1024**3), 1)
            gpu_total = round(mem_info.total / (1024**3), 1)
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
    memory_used = round(memory.used / (1024**3), 1)
    memory_total = round(memory.total / (1024**3), 1)

    import platform as _platform

    try:
        disk_path = "C:" if _platform.system() == "Windows" else "/"
        disk = psutil.disk_usage(disk_path)
        disk_used = round(disk.used / (1024**3), 1)
        disk_total = round(disk.total / (1024**3), 1)
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


def _get_stats_service(request: Request) -> StatsService:
    service = getattr(request.app.state, "stats_service", None)
    if service is None:
        container = get_runtime_container(request.app)
        service = StatsService(
            database=container.db, inference_runtime=container.inference_runtime, env=container.startup_env
        )
        request.app.state.stats_service = service
    return service


async def _stats_snapshot(request: Request) -> dict[str, Any]:
    try:
        return await _get_stats_service(request).snapshot()
    except Exception as exc:
        logger.error("统计数据采集失败: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="统计采集未完成，请检查数据库、运行配置及推理调度器后重试") from exc


@router.get("", response_model=StatsResponse)
async def get_stats(request: Request, current_user: dict = Depends(get_current_admin)):
    snapshot, resources = await asyncio.gather(_stats_snapshot(request), _resource_snapshot())
    metrics, sys_stats, gpu_stats = snapshot["metrics"], resources["system"], resources["gpu"]
    return StatsResponse(
        todayMessages=metrics["todayMessages"],
        todayReplies=metrics["todayReplies"],
        avgResponseTime=metrics["avgResponseTime"],
        p95ResponseTime=metrics["p95ResponseTime"],
        p99ResponseTime=metrics["p99ResponseTime"],
        modelFailureRate=metrics["modelFailureRate"],
        ragFailureRate=metrics["ragFailureRate"],
        activeSessions=metrics["activeSessions"],
        modelLoad=round(100 * metrics["queue"]["active"] / metrics["queue"]["workers"]),
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
async def get_metrics(request: Request, current_user: dict = Depends(get_current_admin)):
    metrics = (await _stats_snapshot(request))["metrics"]
    metrics["alerts"] = _get_stats_service(request).alerts(metrics)
    return metrics


@router.get("/alerts")
async def get_alerts(request: Request, current_user: dict = Depends(get_current_admin)):
    metrics = (await _stats_snapshot(request))["metrics"]
    alerts = _get_stats_service(request).alerts(metrics)
    return {"alerts": alerts, "count": len(alerts), "metrics": metrics}


@router.get("/activity")
async def get_activity(request: Request, current_user: dict = Depends(get_current_admin)):
    return {"activity": (await _stats_snapshot(request))["activity"]}


@router.get("/services")
async def get_services(request: Request, current_user: dict = Depends(get_current_admin)):
    metrics = (await _stats_snapshot(request))["metrics"]
    try:
        health = await request.app.state.readiness_probe.get()
        model_ready, database_ready = health["deps"]["model"], health["deps"]["database"]
    except Exception as exc:
        logger.error("服务状态检查失败: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="服务状态检查未完成，请检查就绪探测后重试") from exc
    uptime = get_service_uptime()
    queue = metrics["queue"]
    gateway = metrics["astrBotGateway"]
    database_name = "PostgreSQL" if get_runtime_container(request.app).is_pg_mode() else "SQLite"
    return {
        "services": [
            {"name": "Backend API", "status": "running", "uptime": uptime},
            {
                "name": "Inference Queue",
                "status": "degraded" if queue["queue_size"] / queue["max_queue_size"] >= 0.8 else "running",
                "uptime": uptime,
                "stats": queue,
            },
            {
                "name": "AstrBot Gateway",
                "status": gateway["status"],
                "uptime": uptime if gateway["running"] else "-",
                "port": gateway["port"],
            },
            {
                "name": "Model Service",
                "status": "running" if model_ready else "stopped",
                "uptime": uptime if model_ready else "-",
            },
            {
                "name": database_name,
                "status": "running" if database_ready else "stopped",
                "uptime": uptime if database_ready else "-",
            },
        ]
    }
