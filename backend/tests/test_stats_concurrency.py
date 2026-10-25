from __future__ import annotations

import asyncio
import threading
import time

import pytest
from stats_fixtures import fresh_observability as fresh_observability


@pytest.mark.asyncio
async def test_metrics_snapshot_single_flights_concurrent_refreshes(monkeypatch):
    from services.stats import StatsService

    calls = 0
    calls_lock = threading.Lock()

    def build_payload():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.03)
        return {"nested": {"value": 1}}

    service = StatsService(database=object(), inference_runtime=object(), env={"STATS_CACHE_TTL_SECONDS": "30"})
    monkeypatch.setattr(service, "collect", build_payload)
    snapshots = await asyncio.gather(*(service.snapshot() for _ in range(20)))

    assert calls == 1
    assert all(item == {"nested": {"value": 1}} for item in snapshots)
    snapshots[0]["nested"]["value"] = 99
    assert (await service.snapshot())["nested"]["value"] == 1


def test_observability_counter_updates_are_thread_safe():
    from infra import observability

    name = "thread_safe_counter_test"
    worker_count = 8
    increments_per_worker = 1_000

    with observability._STATE_LOCK:
        observability._COUNTERS.pop(name, None)
        observability._RECENT.pop(name, None)

    def increment_many():
        for _ in range(increments_per_worker):
            observability.increment(name)

    threads = [threading.Thread(target=increment_many) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observability.get_counter(name) == worker_count * increments_per_worker

    with observability._STATE_LOCK:
        observability._COUNTERS.pop(name, None)
        observability._RECENT.pop(name, None)


@pytest.mark.asyncio
async def test_resource_snapshot_caches_expensive_probes(monkeypatch):
    from api import stats

    calls = {"system": 0, "gpu": 0}

    def system_stats():
        calls["system"] += 1
        time.sleep(0.02)
        return {"cpu_usage": 1}

    def gpu_stats():
        calls["gpu"] += 1
        time.sleep(0.02)
        return {"gpu_used": 2.0, "gpu_total": 3.0}

    monkeypatch.setattr(stats, "get_system_stats", system_stats)
    monkeypatch.setattr(stats, "get_gpu_stats", gpu_stats)
    monkeypatch.setattr(stats, "_RESOURCE_CACHE_TTL_SECONDS", 30.0)
    monkeypatch.setattr(stats, "_resource_cache", None)

    snapshots = await asyncio.gather(*(stats._resource_snapshot() for _ in range(20)))

    assert calls == {"system": 1, "gpu": 1}
    assert all(item["gpu"]["gpu_used"] == 2.0 for item in snapshots)


def test_metrics_snapshot_refreshes_on_a_fresh_event_loop(monkeypatch):
    from services.stats import StatsService

    calls = 0

    def build_payload():
        nonlocal calls
        calls += 1
        return {"generation": calls}

    service = StatsService(database=object(), inference_runtime=object(), env={"STATS_CACHE_TTL_SECONDS": "30"})
    monkeypatch.setattr(service, "collect", build_payload)

    first = asyncio.run(service.snapshot())
    second = asyncio.run(service.snapshot())

    assert first == {"generation": 1}
    assert second == {"generation": 2}



def test_auth_alert_counts_all_recent_events_and_expires_by_time(fresh_observability, monkeypatch):
    from services.stats import StatsService

    observed = fresh_observability
    clock = [1000.0]
    monkeypatch.setattr(observed, "_now", lambda: clock[0])
    service = StatsService(database=object(), inference_runtime=object(),
        env={"ALERT_AUTH_FAILURES_5M": "1500"})
    metrics = {"queueLength": 0, "queueMaxSize": 10, "p95ResponseTime": 0,
        "astrBotGateway": {"status": "stopped"}}
    for _ in range(1501):
        observed.increment("integration_auth_failures")
    assert [(a["type"], a["value"]) for a in service.alerts(metrics)] == [("frequent_auth_failures", 1501)]
    assert observed.snapshot()["recent5m"]["integration_auth_failures"] == 1501
    clock[0] += 300.1
    assert service.alerts(metrics) == []
    assert observed.snapshot()["recent5m"]["integration_auth_failures"] == 0
    observed.increment("integration_auth_failures")
    assert observed.count_recent("integration_auth_failures") == 1
    assert observed.get_counter("integration_auth_failures") == 1502
