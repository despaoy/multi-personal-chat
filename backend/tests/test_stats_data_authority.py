"""Dashboard failures, data ownership, aggregation and live service health."""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from api import stats
from app import dependencies
from services import stats as statistics
from stats_fixtures import StatsRig

PATHS = ["", "/metrics", "/alerts", "/activity", "/services"]


@pytest.fixture
def rig_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(statistics, "check_service", lambda port: False)
    monkeypatch.setattr(
        stats,
        "get_system_stats",
        lambda: {"cpu_usage": 12, "memory_used": 1.0, "memory_total": 8.0, "disk_used": 10.0, "disk_total": 100.0},
    )
    monkeypatch.setattr(stats, "get_gpu_stats", lambda: {"gpu_used": 0.0, "gpu_total": 0.0})
    monkeypatch.setattr(stats, "_resource_cache", None)
    counter = 0

    def make(**kwargs):
        nonlocal counter
        counter += 1
        return StatsRig(tmp_path / f"stats-{counter}.sqlite", **kwargs)

    return make


@pytest.mark.parametrize(
    "stage",
    [
        "messages",
        "invocations",
        "quantiles",
        "activity",
        "events",
        "config",
        "query_shape",
        "aggregate_shape",
        "config_shape",
        "config_boolean",
    ],
)
async def test_collection_failure_is_safe_503_not_successful_zero_metrics(rig_factory, caplog, stage):
    rig = rig_factory(env={})
    rig.seed()
    rig.database.add_model_invocation({"purpose": "chat", "costTime": 1.0})
    rig.db.failure = stage
    async with rig.client() as client:
        response = await client.get("/api/stats/metrics")
    assert response.status_code == 503 and "detail" in response.json()
    assert "todayMessages" not in response.json()
    assert "synthetic-private" not in response.text + caplog.text
    assert "统计数据采集失败" in caplog.text
    rig.probe.get.assert_not_awaited()


@pytest.mark.parametrize("path", ["", "/alerts", "/activity", "/services"])
async def test_every_data_endpoint_propagates_collection_failure(rig_factory, path):
    rig = rig_factory()
    rig.db.failure = "messages"
    async with rig.client() as client:
        response = await client.get("/api/stats" + path)
    assert response.status_code == 503 and "detail" in response.json()


@pytest.mark.parametrize("path", PATHS)
async def test_legitimate_empty_database_keeps_successful_zero_results(rig_factory, path):
    rig = rig_factory()
    async with rig.client() as client:
        response = await client.get("/api/stats" + path)
    assert response.status_code == 200
    if path in {"", "/metrics"}:
        assert response.json()["todayMessages"] == response.json()["todayReplies"] == 0
    elif path == "/activity":
        assert len(response.json()["activity"]) == 12
        assert all(item["messages"] == item["replies"] == 0 for item in response.json()["activity"])
    assert rig.db.config_reads == 1


async def test_applications_own_database_environment_queue_and_cached_metrics(rig_factory):
    first = rig_factory(env={"STATS_CACHE_TTL_SECONDS": "60", "ASTRBOT_ENABLED": "false"})
    second = rig_factory(env={"STATS_CACHE_TTL_SECONDS": "60", "ASTRBOT_ENABLED": "true"})
    first.seed(1)
    second.seed(2)
    second.queue.stats.return_value = {"queue_size": 2, "max_queue_size": 50, "workers": 4, "active": 0}
    async with first.client() as a, second.client() as b:
        ra, rb = await asyncio.gather(a.get("/api/stats/metrics"), b.get("/api/stats/metrics"))
        again = await a.get("/api/stats/metrics")
    assert ra.status_code == rb.status_code == again.status_code == 200
    assert ra.json()["todayMessages"] == again.json()["todayMessages"] == 1
    assert rb.json()["todayMessages"] == 2
    assert ra.json()["queueLength"] == 0 and rb.json()["queueLength"] == 2
    assert ra.json()["astrBotGateway"]["expected"] is False and rb.json()["astrBotGateway"]["expected"] is True
    assert first.db.config_reads == second.db.config_reads == 1


async def test_expired_snapshot_does_not_hide_outage_and_recovers(rig_factory):
    rig = rig_factory(env={"STATS_CACHE_TTL_SECONDS": "0", "ASTRBOT_ENABLED": "false"})
    rig.seed()
    async with rig.client() as client:
        first = await client.get("/api/stats/metrics")
        rig.db.failure = "messages"
        failed = await client.get("/api/stats/metrics")
        rig.db.failure = ""
        rig.seed(2)
        recovered = await client.get("/api/stats/metrics")
    assert first.status_code == 200 and first.json()["todayMessages"] == 1
    assert failed.status_code == 503 and "todayMessages" not in failed.json()
    assert recovered.status_code == 200 and recovered.json()["todayMessages"] == 3


async def test_counts_over_ten_thousand_use_small_aggregates_and_real_queue_load(rig_factory):
    rig = rig_factory()
    rig.seed(10001)
    rig.database.add_model_invocation({"purpose": "chat", "costTime": 1.0})
    async with rig.client() as client:
        response = await client.get("/api/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["todayMessages"] == payload["todayReplies"] == payload["activeSessions"] == 10001
    assert payload["avgResponseTime"] == payload["p95ResponseTime"] == payload["p99ResponseTime"] == 1.0
    assert payload["modelLoad"] == 50
    assert rig.db.returned_rows <= 32 and rig.db.config_reads == 1
    assert "synthetic-private-dialogue-body" not in json.dumps(payload)


async def test_activity_covers_all_twenty_four_hours_and_excludes_tomorrow(rig_factory):
    rig = rig_factory()
    for hour in range(24):
        rig.seed(hour=hour)
    rig.database.add_message(
        {
            "sessionType": "private",
            "sessionId": "future-synthetic",
            "message": "complete future source",
            "reply": "future reply",
            "createdAt": (datetime.now() + timedelta(days=1)).isoformat(),
        }
    )
    async with rig.client() as client:
        response = await client.get("/api/stats/activity")
    assert response.status_code == 200
    activity = response.json()["activity"]
    assert [row["time"] for row in activity] == [f"{hour:02d}:00" for hour in range(0, 24, 2)]
    assert all(row["messages"] == row["replies"] == 2 for row in activity)


@pytest.mark.parametrize("model,database", [(True, True), (False, True), (True, False)])
async def test_service_status_uses_live_readiness_instead_of_fixed_success(rig_factory, model, database):
    rig = rig_factory()
    rig.probe.get.return_value = {"ready": model and database, "deps": {"model": model, "database": database}}
    async with rig.client() as client:
        response = await client.get("/api/stats/services")
    assert response.status_code == 200
    services = {row["name"]: row for row in response.json()["services"]}
    assert services["Model Service"]["status"] == ("running" if model else "stopped")
    assert services["SQLite"]["status"] == ("running" if database else "stopped")
    assert "In-Memory DB" not in services
    rig.probe.get.assert_awaited_once_with()


async def test_readiness_check_failure_is_safe_503(rig_factory, caplog):
    rig = rig_factory()
    rig.probe.get.side_effect = OSError("synthetic-private-probe-location")
    async with rig.client() as client:
        response = await client.get("/api/stats/services")
    assert response.status_code == 503
    assert "synthetic-private" not in response.text + caplog.text


@pytest.mark.parametrize("identity,expected", [(None, 401), ("member", 403)])
@pytest.mark.parametrize("path", PATHS)
async def test_authorization_precedes_database_collection(rig_factory, monkeypatch, identity, expected, path):
    rig = rig_factory()
    rig.app.dependency_overrides.clear()
    if identity is not None:

        async def member(request):
            return {"role": "user", "auth_type": "managed_api_key", "username": "synthetic-member"}

        monkeypatch.setattr(dependencies, "get_current_user", member)
    async with rig.client() as client:
        response = await client.get("/api/stats" + path)
    assert response.status_code == expected
    assert rig.db.config_reads == 0 and rig.db.queries == []
    rig.probe.get.assert_not_awaited()


@pytest.mark.parametrize("ttl", ["-1", "nan", "inf"])
async def test_invalid_cache_configuration_is_not_clamped(rig_factory, ttl):
    rig = rig_factory(env={"STATS_CACHE_TTL_SECONDS": ttl})
    async with rig.client() as client:
        response = await client.get("/api/stats/metrics")
    assert response.status_code == 503 and rig.db.queries == []


async def test_collection_cancellation_keeps_its_original_type(rig_factory, monkeypatch):
    rig = rig_factory()
    failure = asyncio.CancelledError()
    monkeypatch.setattr(statistics.StatsService, "snapshot", AsyncMock(side_effect=failure))
    async with rig.client() as client:
        with pytest.raises(asyncio.CancelledError) as caught:
            await client.get("/api/stats/metrics")
    assert caught.value is failure



async def test_generation_percentiles_never_substitute_message_history(rig_factory):
    rig = rig_factory(env={"STATS_CACHE_TTL_SECONDS": "0", "ASTRBOT_ENABLED": "false"})
    rig.seed(3, cost=99.0)
    async with rig.client() as client:
        empty = await client.get("/api/stats/metrics")
        assert empty.status_code == 200
        assert empty.json()["p95ResponseTime"] == empty.json()["p99ResponseTime"] == 0.0
        assert empty.json()["chatGenerations"] == 0
        for cost in [2.0, 4.0]:
            rig.database.add_model_invocation({"purpose": "chat", "costTime": cost})
        rig.database.add_model_invocation({"purpose": "memory_judgement", "costTime": 9000.0, "errorType": "synthetic"})
        measured = await client.get("/api/stats/metrics")
        basic = await client.get("/api/stats")
    assert measured.status_code == basic.status_code == 200
    data = measured.json()
    assert data["todayMessages"] == 3 and data["avgResponseTime"] == 99.0
    assert data["p95ResponseTime"] == data["p99ResponseTime"] == 4.0
    assert data["chatGenerations"] == 2 and data["chatGenerationFailures"] == 0
    assert data["chatGenerationFailureRate"] == basic.json()["chatGenerationFailureRate"] == 0.0
    assert not {"modelFailureRate", "modelFailures", "modelInvocations"} & data.keys()
