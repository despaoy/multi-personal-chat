"""Tests that formal experiment APIs never degrade into unlabelled mock data."""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest
from fastapi import HTTPException

from api import experiments
from db.schemas import ExperimentStartRequest


class _RecordingDatabase:
    def __init__(self, *, fail_insert: bool = False):
        self.fail_insert = fail_insert
        self.inserts = []
        self.updates = []

    def execute_sql_insert(self, query, params=None):
        if self.fail_insert:
            raise RuntimeError("database unavailable")
        self.inserts.append((query, params))
        return 1

    def execute_sql(self, query, params=None):
        self.updates.append((query, params))
        return []


def test_experiment_creation_fails_closed_when_database_is_unavailable(monkeypatch):
    monkeypatch.setattr(experiments, "db", _RecordingDatabase(fail_insert=True))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            experiments.start_lora_ablation(
                ExperimentStartRequest(mock=True),
                current_user={"role": "admin"},
            )
        )

    assert exc_info.value.status_code == 503


def test_missing_quantization_module_cannot_turn_real_run_into_mock(
    monkeypatch,
):
    database = _RecordingDatabase()
    monkeypatch.setattr(experiments, "db", database)
    monkeypatch.setitem(sys.modules, "experiments.quantization_benchmark", None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            experiments.start_quantization_benchmark(
                ExperimentStartRequest(mock=False),
                current_user={"role": "admin"},
            )
        )

    assert exc_info.value.status_code == 503
    assert database.inserts
    assert database.updates
    assert database.updates[-1][1]["id"].startswith("quant_bench_")
    assert '"mock": false' in database.updates[-1][1]["r"]

class _QueryDatabase:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.thread_ids = []

    def execute_sql(self, query, params=None):
        self.thread_ids.append(threading.get_ident())
        if self.fail:
            raise RuntimeError("postgres://user:secret@private-host/database")
        if "COUNT(*)" in query:
            return [{"cnt": 0}]
        return []


def test_experiment_queries_leave_the_event_loop_thread(monkeypatch):
    database = _QueryDatabase()
    monkeypatch.setattr(experiments, "db", database)
    caller_thread = threading.get_ident()

    result = asyncio.run(
        experiments.list_experiments(current_user={"role": "admin"})
    )

    assert result == {"success": True, "experiments": [], "total": 0}
    assert database.thread_ids
    assert all(thread_id != caller_thread for thread_id in database.thread_ids)


def test_experiment_query_errors_do_not_expose_database_details(monkeypatch):
    monkeypatch.setattr(experiments, "db", _QueryDatabase(fail=True))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            experiments.get_experiment(
                "private-experiment",
                current_user={"role": "admin"},
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "实验记录暂时不可用"
    assert "secret" not in exc_info.value.detail
