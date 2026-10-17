"""A failed resource probe or invalid alert setting cannot become healthy data."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from api import stats
from services import stats as statistics
from stats_fixtures import StatsRig


def system_metrics():
    return {"cpu_usage": 12, "memory_used": 1.0, "memory_total": 8.0, "disk_used": 10.0, "disk_total": 100.0}


@pytest.mark.parametrize(
    "output,expected",
    [
        ("24576, 1024\n", {"gpu_total": 24.0, "gpu_used": 1.0}),
        ("24576, 0\n12288, 1024\n", {"gpu_total": 24.0, "gpu_used": 0.0}),
        ("", {"gpu_total": 0.0, "gpu_used": 0.0}),
    ],
)
def test_gpu_uses_one_successful_driver_query_and_preserves_first_device(monkeypatch, output, expected):
    query = Mock(return_value=subprocess.CompletedProcess(["nvidia-smi"], 0, stdout=output, stderr=""))
    monkeypatch.setattr(stats.subprocess, "run", query)
    assert stats.get_gpu_stats() == expected
    query.assert_called_once_with(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("synthetic-private-driver-location"),
        subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="synthetic-private-driver-detail"),
        subprocess.TimeoutExpired(["nvidia-smi"], 5),
    ],
)
def test_driver_failures_propagate_without_zero_or_another_tool(monkeypatch, failure):
    query = Mock(side_effect=failure)
    monkeypatch.setattr(stats.subprocess, "run", query)
    with pytest.raises(type(failure)) as caught:
        stats.get_gpu_stats()
    assert caught.value is failure
    query.assert_called_once()


@pytest.mark.parametrize("output", ["invalid", "100, 1, 2", "0, 0", "100, -1", "100, 101", "inf, 1", "100, nan"])
def test_malformed_or_impossible_gpu_measurement_is_an_error(monkeypatch, output):
    monkeypatch.setattr(
        stats.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess(["nvidia-smi"], 0, stdout=output, stderr="")),
    )
    with pytest.raises(ValueError):
        stats.get_gpu_stats()


def test_disk_failure_is_not_replaced_by_zero(monkeypatch):
    failure = OSError("synthetic-private-disk-location")
    monkeypatch.setattr(stats.psutil, "cpu_percent", lambda interval: 12)
    monkeypatch.setattr(stats.psutil, "virtual_memory", lambda: SimpleNamespace(used=1024**3, total=8 * 1024**3))
    monkeypatch.setattr(stats.psutil, "disk_usage", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        stats.get_system_stats()
    assert caught.value is failure


@pytest.fixture
def resource_rig(tmp_path, monkeypatch):
    rig = StatsRig(tmp_path / "resources.sqlite")
    monkeypatch.setattr(statistics, "check_service", lambda port: False)
    monkeypatch.setattr(stats, "get_system_stats", system_metrics)
    monkeypatch.setattr(stats, "get_gpu_stats", lambda: {"gpu_used": 1.0, "gpu_total": 24.0})
    monkeypatch.setattr(stats, "_resource_cache", None)
    monkeypatch.setattr(stats, "_RESOURCE_CACHE_TTL_SECONDS", 0.0)
    return rig


@pytest.mark.parametrize("component", ["get_system_stats", "get_gpu_stats"])
async def test_resource_outage_is_safe_503_and_does_not_reuse_expired_data(
    resource_rig, monkeypatch, caplog, component
):
    async with resource_rig.client() as client:
        before = await client.get("/api/stats")
        assert before.status_code == 200
        collector = stats.get_system_stats if component == "get_system_stats" else stats.get_gpu_stats
        failure = OSError("synthetic-private-probe-location")
        monkeypatch.setattr(stats, component, Mock(side_effect=failure))
        failed = await client.get("/api/stats")
        assert failed.status_code == 503 and "detail" in failed.json()
        assert "gpuMemory" not in failed.json() and "diskUsage" not in failed.json()
        assert "synthetic-private" not in failed.text + caplog.text
        monkeypatch.setattr(stats, component, collector)
        recovered = await client.get("/api/stats")
    assert recovered.status_code == 200 and recovered.json()["gpuMemory"] == {"used": 1.0, "total": 24.0}


@pytest.mark.parametrize("path", ["/metrics", "/alerts", "/activity", "/services"])
async def test_independent_data_endpoints_do_not_probe_hardware(resource_rig, monkeypatch, path):
    forbidden = Mock(side_effect=AssertionError("data-only endpoint must not probe hardware"))
    monkeypatch.setattr(stats, "get_gpu_stats", forbidden)
    monkeypatch.setattr(stats, "get_system_stats", forbidden)
    async with resource_rig.client() as client:
        response = await client.get("/api/stats" + path)
    assert response.status_code == 200
    forbidden.assert_not_called()


async def test_resource_cancellation_is_not_converted_to_http_failure():
    failure = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await stats._resource_probe("GPU", Mock(side_effect=failure))


@pytest.mark.parametrize(
    "name,value",
    [
        ("ALERT_RESPONSE_P95_SECONDS", "-1"),
        ("ALERT_RESPONSE_P95_SECONDS", "nan"),
        ("ALERT_RESPONSE_P95_SECONDS", "inf"),
        ("ALERT_RESPONSE_P95_SECONDS", "synthetic-private-alert-setting"),
        ("ALERT_QUEUE_RATIO", "-0.1"),
        ("ALERT_QUEUE_RATIO", "1.1"),
        ("ALERT_QUEUE_RATIO", "nan"),
        ("ALERT_AUTH_FAILURES_5M", "-1"),
        ("ALERT_DB_FAILURES_5M", "1.5"),
        ("ALERT_MODEL_CONSECUTIVE_FAILURES", "synthetic-private-alert-setting"),
    ],
)
async def test_invalid_alert_setting_is_named_safe_and_fails_before_collection(
    tmp_path, monkeypatch, caplog, name, value
):
    rig = StatsRig(tmp_path / "bad-alert.sqlite", env={name: value, "ASTRBOT_ENABLED": "false"})
    monkeypatch.setattr(statistics, "check_service", lambda port: False)
    async with rig.client() as client:
        response = await client.get("/api/stats/metrics")
    assert response.status_code == 503 and name in response.json()["detail"]
    assert "synthetic-private" not in response.text + caplog.text
    assert rig.db.queries == [] and rig.db.config_reads == 0


def test_zero_thresholds_preserve_immediate_alert_behavior(monkeypatch):
    names = [
        "ALERT_RESPONSE_P95_SECONDS",
        "ALERT_QUEUE_RATIO",
        "ALERT_AUTH_FAILURES_5M",
        "ALERT_DB_FAILURES_5M",
        "ALERT_MODEL_CONSECUTIVE_FAILURES",
    ]
    service = statistics.StatsService(database=object(), inference_runtime=object(), env={name: "0" for name in names})
    monkeypatch.setattr(statistics, "count_recent", lambda *args: 0)
    monkeypatch.setattr(statistics, "get_consecutive", lambda *args: 0)
    metrics = {"queueLength": 0, "queueMaxSize": 100, "p95ResponseTime": 0, "astrBotGateway": {"status": "stopped"}}
    types = {alert["type"] for alert in service.alerts(metrics)}
    assert types == {
        "model_consecutive_failures",
        "database_write_failures",
        "message_backlog",
        "slow_response",
        "frequent_auth_failures",
    }


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "synthetic-private-resource-setting", "0"])
def test_resource_cache_setting_is_checked_during_real_module_import(tmp_path, value):
    program = """
import json,logging
logging.disable(logging.CRITICAL)
from infra.environment import ConfigurationError
try:
 from api import stats
except ConfigurationError as exc:
 print(json.dumps(dict(invalid=True,name=exc.name,private_value_hidden='synthetic-private' not in str(exc))))
else:
 print(json.dumps(dict(invalid=False,ttl=stats._RESOURCE_CACHE_TTL_SECONDS)))
"""
    env = {**os.environ, "STATS_RESOURCE_CACHE_TTL_SECONDS": value, "DATABASE_PATH": str(tmp_path / "import.sqlite")}
    result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if value == "0":
        assert payload == {"invalid": False, "ttl": 0.0}
    else:
        assert payload == {"invalid": True, "name": "STATS_RESOURCE_CACHE_TTL_SECONDS", "private_value_hidden": True}
