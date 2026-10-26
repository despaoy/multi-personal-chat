"""Executor configuration is explicit and every admitted wait budget is finite."""

import base64
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
from unittest.mock import Mock

import pytest

from infra import bounded_executor
from infra.environment import read_positive_float, read_positive_int


def test_missing_defaults_and_valid_configured_values_remain_exact():
    assert read_positive_int({}, "WORKERS", 8) == 8
    assert read_positive_float({}, "TIMEOUT", 30.0) == 30.0
    assert read_positive_int({"WORKERS": " +3 "}, "WORKERS", 8) == 3
    assert read_positive_float({"TIMEOUT": "0.125"}, "TIMEOUT", 30.0) == 0.125
    assert read_positive_float({"TIMEOUT": "1e-3"}, "TIMEOUT", 30.0) == 0.001


@pytest.mark.parametrize("value", ["", "controlled-private-number", "0", "-3", "1.5"])
def test_invalid_integer_config_is_not_defaulted_or_clamped(value):
    with pytest.raises(RuntimeError, match="WORKERS.*positive integer") as raised:
        read_positive_int({"WORKERS": value}, "WORKERS", 8)
    assert "controlled-private-number" not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize("value", ["", "controlled-private-number", "0", "-1", "nan", "inf"])
def test_invalid_float_config_names_the_setting_without_exposing_its_value(value):
    with pytest.raises(RuntimeError, match="TIMEOUT.*positive number.*finite") as raised:
        read_positive_float({"TIMEOUT": value}, "TIMEOUT", 30.0)
    assert "controlled-private-number" not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_default_deadline_is_rejected_before_executor_allocation(monkeypatch, value):
    create = Mock(side_effect=AssertionError("Invalid timeout must not allocate a worker pool"))
    monkeypatch.setattr(bounded_executor, "ThreadPoolExecutor", create)
    with pytest.raises(ValueError, match="default_timeout.*positive"):
        bounded_executor.BoundedThreadExecutor(name="numeric-test", max_workers=1, max_pending=1, default_timeout=value)
    create.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
async def test_invalid_override_deadline_does_not_admit_work_or_consume_capacity(value):
    executor = bounded_executor.BoundedThreadExecutor(
        name="numeric-test", max_workers=1, max_pending=1, default_timeout=1.0
    )
    work = Mock(return_value="must not run")
    try:
        with pytest.raises(ValueError, match="timeout.*positive"):
            await executor.run(work, timeout=value)
        work.assert_not_called()
        assert executor.stats()["submitted"] == executor.stats()["pending"] == 0
        assert await executor.run(lambda: "healthy", timeout=0.5) == "healthy"
    finally:
        executor.shutdown()


BASELINE = {
    "ASYNC_DB_WORKERS": "8",
    "ASYNC_DB_MAX_PENDING": "32",
    "ASYNC_DB_TIMEOUT_SECONDS": "30",
    "AUTH_DATABASE_WORKERS": "8",
    "AUTH_DATABASE_MAX_PENDING": "32",
    "AUTH_DATABASE_TIMEOUT_SECONDS": "5",
    "AUTH_PASSWORD_WORKERS": "2",
    "AUTH_PASSWORD_MAX_PENDING": "8",
    "AUTH_PASSWORD_TIMEOUT_SECONDS": "10",
    "MODEL_INFERENCE_TIMEOUT": "180",
    "DB_WRITE_TIMEOUT": "3",
    "RAG_TIMEOUT": "8",
    "RAG_COLD_START_TIMEOUT": "60",
    "INTEGRATION_AUTH_TIMEOUT": "3",
    "INTEGRATION_DB_TIMEOUT": "3",
    "TRAINING_EXPORT_MAX_CONCURRENCY": "2",
    "TRAINING_EXPORT_ACQUIRE_TIMEOUT": "1",
}


def child_environment(tmp_path):
    env = dict(
        os.environ,
        ENVIRONMENT="test",
        USE_POSTGRESQL="false",
        DATABASE_PATH=str(tmp_path / "isolated.sqlite"),
        DATABASE_URL="postgresql+asyncpg://unused@127.0.0.1:1/unused",
        JWT_SECRET="executor-config-synthetic-secret-more-than-32-characters",
        ENCRYPTION_KEY=base64.urlsafe_b64encode(b"n" * 32).decode(),
        BOOTSTRAP_ADMIN_USERNAME="synthetic-admin",
        BOOTSTRAP_ADMIN_PASSWORD="synthetic-password",
    )
    env.update(BASELINE)
    return env


@pytest.mark.parametrize(
    ("module", "setting", "value"),
    [
        ("infra.db_executor", "ASYNC_DB_WORKERS", "controlled-private-number"),
        ("infra.db_executor", "ASYNC_DB_WORKERS", "-3"),
        ("infra.db_executor", "ASYNC_DB_MAX_PENDING", "0"),
        ("infra.db_executor", "ASYNC_DB_TIMEOUT_SECONDS", "nan"),
        ("infra.auth_work", "AUTH_DATABASE_TIMEOUT_SECONDS", "inf"),
        ("api.auth", "AUTH_PASSWORD_TIMEOUT_SECONDS", "nan"),
        ("api.auth", "AUTH_PASSWORD_WORKERS", "controlled-private-number"),
        ("infra.auth_work", "AUTH_DATABASE_MAX_PENDING", "controlled-private-number"),
        ("api.generate", "MODEL_INFERENCE_TIMEOUT", "inf"),
        ("api.generate", "DB_WRITE_TIMEOUT", "nan"),
        ("services.model_usage", "DB_WRITE_TIMEOUT", "0"),
        ("services.chat_answer", "RAG_TIMEOUT", "inf"),
        ("services.chat_answer", "RAG_COLD_START_TIMEOUT", "-1"),
        ("api.integrations", "INTEGRATION_AUTH_TIMEOUT", "inf"),
        ("api.integrations", "INTEGRATION_DB_TIMEOUT", "nan"),
        ("api.training", "TRAINING_EXPORT_MAX_CONCURRENCY", "0"),
        ("api.training", "TRAINING_EXPORT_ACQUIRE_TIMEOUT", "nan"),
    ],
)
def test_configured_consumers_fail_during_import_with_safe_setting_name(tmp_path, module, setting, value):
    source = """import importlib,json,logging,sys,traceback
logging.disable(logging.CRITICAL)
from app import env
env.load_backend_env=lambda *args,**kwargs:None
try:importlib.import_module(sys.argv[1])
except Exception as error:
 print(json.dumps(dict(kind=type(error).__name__,setting_named=sys.argv[2] in str(error),
  private_hidden='controlled-private-number' not in ''.join(traceback.format_exception(error)))))
else:print(json.dumps(dict(kind='accepted',setting_named=False,private_hidden=True)))
"""
    env = child_environment(tmp_path)
    env[setting] = value
    result = subprocess.run(
        [sys.executable, "-c", source, module, setting], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"kind": "RuntimeError", "setting_named": True, "private_hidden": True}


@pytest.mark.parametrize(
    ("module", "attribute", "expected"),
    [
        ("infra.db_executor", "db_executor", [8, 32, 30.0]),
        ("infra.auth_work", "auth_database_executor", [8, 32, 5.0]),
        ("api.auth", "_password_executor", [2, 8, 10.0]),
    ],
)
def test_existing_consumer_defaults_remain_unchanged_when_settings_are_missing(tmp_path, module, attribute, expected):
    source = """import importlib,json,logging,sys
logging.disable(logging.CRITICAL)
from app import env
env.load_backend_env=lambda *args,**kwargs:None
executor=getattr(importlib.import_module(sys.argv[1]),sys.argv[2])
print(json.dumps([executor.max_workers,executor.max_pending,executor.default_timeout]))
executor.shutdown()
"""
    env = child_environment(tmp_path)
    for name in BASELINE:
        env.pop(name)
    result = subprocess.run(
        [sys.executable, "-c", source, module, attribute], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("entry", ["run.py", "main.py"])
@pytest.mark.parametrize("invalid", [False, True])
def test_both_launchers_use_the_shared_positive_config_contract(monkeypatch, entry, invalid):
    import app.env
    import uvicorn

    monkeypatch.setattr(app.env, "load_backend_env", lambda *args, **kwargs: None)
    monkeypatch.setenv("BACKEND_WORKERS", "1")
    monkeypatch.setenv("BACKEND_LIMIT_CONCURRENCY", "5")
    monkeypatch.setenv("BACKEND_BACKLOG", "0" if invalid else "25")
    monkeypatch.setenv("BACKEND_KEEPALIVE_TIMEOUT", "3")
    path = Path(__file__).resolve().parents[1] / entry
    monkeypatch.setattr(sys, "argv", [str(path)])
    launch = Mock()
    monkeypatch.setattr(uvicorn, "run", launch)
    if invalid:
        with pytest.raises(RuntimeError, match="BACKEND_BACKLOG.*positive integer"):
            runpy.run_path(str(path), run_name="__main__")
        launch.assert_not_called()
    else:
        runpy.run_path(str(path), run_name="__main__")
        launch.assert_called_once()
        assert launch.call_args.kwargs["limit_concurrency"] == 5
        assert launch.call_args.kwargs["backlog"] == 25
        assert launch.call_args.kwargs["timeout_keep_alive"] == 3
        assert launch.call_args.kwargs["workers"] == 1



@pytest.mark.parametrize("configured", [False, True])
def test_chat_deadlines_keep_defaults_and_explicit_positive_values(tmp_path, configured):
    source = """import json,logging
logging.disable(logging.CRITICAL)
from app import env
env.load_backend_env=lambda *args,**kwargs:None
from api import generate,integrations
from services import chat_answer,model_usage
print(json.dumps([generate._MODEL_INFERENCE_TIMEOUT,generate._DB_WRITE_TIMEOUT,
 chat_answer._RAG_TIMEOUT,chat_answer._RAG_COLD_START_TIMEOUT,model_usage._WRITE_TIMEOUT,
 integrations._AUTH_TIMEOUT,integrations._DB_TIMEOUT,integrations._MODEL_TIMEOUT]))
"""
    env = child_environment(tmp_path)
    names = ("MODEL_INFERENCE_TIMEOUT", "DB_WRITE_TIMEOUT", "RAG_TIMEOUT", "RAG_COLD_START_TIMEOUT",
             "INTEGRATION_AUTH_TIMEOUT", "INTEGRATION_DB_TIMEOUT")
    for name in names:
        env.pop(name)
    if configured:
        env.update(dict.fromkeys(names, "0.125"))
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert json.loads(result.stdout) == ([.125] * 8 if configured else [180., 3., 8., 60., 3., 3., 3., 180.])



@pytest.mark.parametrize(
    "settings,expected",
    [
        ({}, [2, 1.0]),
        ({"TRAINING_EXPORT_MAX_CONCURRENCY": "3", "TRAINING_EXPORT_ACQUIRE_TIMEOUT": "0.001"}, [3, 0.001]),
    ],
)
def test_training_export_defaults_and_configured_limits_remain_exact(tmp_path, settings, expected):
    source = """import json,logging
logging.disable(logging.CRITICAL)
from app import env
env.load_backend_env=lambda *args,**kwargs:None
from api import training
print(json.dumps([training._DATASET_EXPORT_MAX_CONCURRENCY,training._DATASET_EXPORT_ACQUIRE_TIMEOUT]))
"""
    env = child_environment(tmp_path)
    for name in ["TRAINING_EXPORT_MAX_CONCURRENCY", "TRAINING_EXPORT_ACQUIRE_TIMEOUT"]:
        env.pop(name)
    env.update(settings)
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert json.loads(result.stdout) == expected
