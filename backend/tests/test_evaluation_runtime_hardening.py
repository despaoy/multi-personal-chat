"""Regression tests for evaluation task lifecycle and API contracts."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import evaluation as evaluation_api
from app.dependencies import get_current_admin
from db.schemas import EvalRunRequest, FeedbackCreate
from evaluation import runtime_runner


class _ListDatabase:
    def execute_sql(self, query, params=None):
        if "COUNT(*)" in query:
            return [{"cnt": 5}]
        return [
            {
                "id": "eval_1",
                "run_at": "2026-01-01T00:00:00+00:00",
                "adapter_name": None,
                "model_label": "model",
                "total_prompts": 1,
                "metrics": json.dumps({"mock": False}),
                "notes": "completed",
            }
        ]


class _FailingInsertDatabase:
    def execute_sql_insert(self, query, params=None):
        raise RuntimeError("database unavailable")


def _evaluation_client(monkeypatch, database) -> TestClient:
    monkeypatch.setattr(evaluation_api, "db", database)
    app = FastAPI()
    app.include_router(evaluation_api.router)
    app.dependency_overrides[get_current_admin] = lambda: {
        "user_id": 1,
        "username": "admin",
        "role": "admin",
    }
    return TestClient(app)


def test_evaluation_lock_restarts_on_a_fresh_event_loop(monkeypatch):
    monkeypatch.setattr(runtime_runner, "_evaluation_lock", None)
    monkeypatch.setattr(runtime_runner, "_evaluation_lock_loop", None)

    async def use_lock():
        lock = runtime_runner._get_evaluation_lock()
        async with lock:
            pass
        return lock

    first = asyncio.run(use_lock())
    second = asyncio.run(use_lock())

    assert second is not first


def test_evaluation_shutdown_cancels_and_joins_tasks(monkeypatch):
    async def scenario():
        runtime_runner._evaluation_tasks.clear()
        started = asyncio.Event()

        async def wait_forever(run_id, options, database):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            runtime_runner,
            "execute_generation_evaluation",
            wait_forever,
        )
        task = runtime_runner.schedule_generation_evaluation("eval_test", {}, object())
        await asyncio.wait_for(started.wait(), timeout=1)

        assert task in runtime_runner._evaluation_tasks
        await runtime_runner.shutdown_generation_evaluations()

        assert task.cancelled()
        assert not runtime_runner._evaluation_tasks

    asyncio.run(scenario())


def test_update_run_uses_cross_database_named_parameters():
    calls = []

    class Database:
        def execute_sql(self, query, params=None):
            calls.append((query, params))

    runtime_runner._update_run(
        Database(),
        "eval_1",
        metrics={"mock": False},
        total=2,
        breakdown={"persona": 2},
        note="completed",
    )

    query, params = calls[0]
    assert "?" not in query
    assert ":metrics" in query
    assert params["run_id"] == "eval_1"
    assert json.loads(params["breakdown"]) == {"persona": 2}


def test_evaluation_run_returns_503_when_record_cannot_be_created(monkeypatch):
    with _evaluation_client(monkeypatch, _FailingInsertDatabase()) as client:
        response = client.post("/api/evaluation/run", json={"mock": False})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


def test_evaluation_runs_returns_sql_total(monkeypatch):
    with _evaluation_client(monkeypatch, _ListDatabase()) as client:
        response = client.get("/api/evaluation/runs", params={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 5
    assert len(payload["runs"]) == 1


def test_evaluation_and_feedback_inputs_are_bounded():
    with pytest.raises(ValidationError):
        EvalRunRequest(max_prompts=51)
    with pytest.raises(ValidationError):
        EvalRunRequest(split="training")
    with pytest.raises(ValidationError):
        FeedbackCreate(rating="unknown")
    with pytest.raises(ValidationError):
        FeedbackCreate(rating="thumbs_up", detail="x" * 10001)


def test_runtime_evaluation_defaults_to_current_kisaki_development_set():
    request = EvalRunRequest()
    dataset = runtime_runner.load_runtime_dataset(request.dataset_id)
    assert request.dataset_id == "kisaki_v21"
    assert dataset["evaluation_role"] == "development_only"
    assert all(item.get("persona", "kisaki") == "kisaki" for item in dataset["prompts"])


def test_runtime_evaluation_preserves_multiturn_user_context():
    item = runtime_runner.load_runtime_dataset("kisaki_v21")["prompts"][60]
    assert item["category"] == "multiturn"
    assert runtime_runner.conversation_turns(item) == [
        message["content"] for message in item["conversation"]
    ]
