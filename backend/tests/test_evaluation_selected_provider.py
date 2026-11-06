"""Selected-provider evaluation through the actual API, compiler, transport and store."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from evaluation import runtime_runner
from fastapi import FastAPI

from api import evaluation
from app.dependencies import get_current_admin
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from inference import model_manager, request_context, vllm_client

FIRST = "完整前提：甲盒原有4枚绿筹码，没有红筹码；后来增加3枚绿筹码，其他数量不变。请给出两种筹码的数量。"
SECOND = "沿用同一甲盒，移走2枚绿筹码。请分别给出现在的绿筹码和红筹码数量。"
ANSWERS = ["绿筹码7枚，红筹码0枚。", "绿筹码5枚，红筹码0枚。"]


@pytest.fixture
async def lab(tmp_path, monkeypatch):
    from api import generate as chat_api

    database = SQLiteDB(tmp_path / "evaluation.sqlite")
    state = SimpleNamespace(
        db=database,
        requests=[],
        tasks=[],
        manager_databases=[],
        mode="complete",
        entered=asyncio.Event(),
        acquires=0,
        releases=0,
    )
    database.update_config(
        {
            "openaiCompatBaseUrl": "https://api.deepseek.com/v1",
            "openaiCompatApiKey": "synthetic-evaluation-key",
            "openaiCompatModel": "synthetic-selected-model",
        }
    )
    provider = model_manager.OpenAICompatProvider(database=database)
    state.provider = state.selected = provider
    state.dataset = {
        "status": "isolated",
        "evaluation_role": "complete synthetic input",
        "prompts": [
            {"id": "complete-multi", "category": "complete", "turns": [FIRST, SECOND]},
            {"id": "independent", "category": "complete", "prompt": "完整题设：每张票2元，3张票共多少元？"},
        ],
    }

    async def handle(request):
        body = json.loads(request.content)
        state.requests.append(dict(url=str(request.url), authorization=request.headers.get("Authorization"), body=body))
        state.entered.set()
        if state.mode == "wait":
            await asyncio.Event().wait()
        if state.mode == "auth":
            return httpx.Response(401, json={"error": {"message": "synthetic denied"}})
        index = len(state.requests) - 1
        if state.mode == "mutate" and index == 0:
            provider.base_url, provider.model, provider.api_key = "https://changed.test", "changed-model", "changed-key"
        return httpx.Response(
            200,
            json={
                "id": "synthetic-response",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ANSWERS[index % 2]},
                    }
                ],
                "usage": {"prompt_tokens": 11 + index, "completion_tokens": 5 + index, "total_tokens": 16 + 2 * index},
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            state.acquires += 1
            try:
                yield http
            finally:
                state.releases += 1

    state.pool = Pool()
    manager = SimpleNamespace(get_current_provider=lambda: state.selected)

    def get_manager(*, database):
        state.manager_databases.append(database)
        assert database is state.db
        return manager

    forbidden = AsyncMock(side_effect=AssertionError("Selecting cloud must not acquire local vLLM"))
    state.forbidden = forbidden
    monkeypatch.setattr(chat_api, "get_vllm_client", forbidden)
    monkeypatch.setattr(vllm_client, "get_enabled_vllm_client", forbidden)
    monkeypatch.setattr(runtime_runner, "get_enabled_vllm_client", forbidden, raising=False)
    monkeypatch.setattr(runtime_runner, "get_model_manager", get_manager, raising=False)
    monkeypatch.setattr(request_context, "text_token_counter", lambda *args: len)
    monkeypatch.setattr(runtime_runner, "load_runtime_dataset", lambda _: state.dataset)
    monkeypatch.setenv("OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS", "65536")
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8192")
    schedule = runtime_runner.schedule_generation_evaluation

    def track(*args, **kwargs):
        task = schedule(*args, **kwargs)
        state.tasks.append(task)
        return task

    monkeypatch.setattr(runtime_runner, "schedule_generation_evaluation", track)
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(
        db=database, is_pg_mode=lambda: False, startup_env={}, http_client_pool=state.pool
    )
    app.include_router(evaluation.router)
    app.dependency_overrides[get_current_admin] = lambda: {"user_id": 1, "username": "synthetic-admin", "role": "admin"}
    state.api = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://isolated-evaluation.test")

    async def start(**options):
        response = await state.api.post(
            "/api/evaluation/run", json={"dataset_id": "legacy_general", "max_prompts": 1, "mock": False, **options}
        )
        assert response.status_code == 200 and response.json()["status"] == "queued"
        state.run_id = response.json()["run_id"]
        return state.tasks[-1]

    def result():
        row = database.execute_sql(
            "SELECT notes,metrics,total_prompts,category_breakdown FROM gold_eval_runs WHERE id=:id",
            {"id": state.run_id},
        )[0]
        return row, json.loads(row["metrics"])

    state.start, state.result = start, result
    try:
        yield state
    finally:
        await runtime_runner.shutdown_generation_evaluations()
        await state.api.aclose()
        await http.aclose()
        database.close_connection()


async def test_cloud_job_pins_selected_model_keeps_history_and_records_actual_usage(lab):
    lab.mode = "mutate"
    await (await lab.start())
    row, result = lab.result()
    assert row["notes"] == "completed" and result["mock"] is False
    assert result["provider"] == "openai_compat" and result["model"] == "synthetic-selected-model"
    assert result["samples"][0]["turn_responses"] == ANSWERS and result["scored_responses"] == 1
    assert len(lab.requests) == 2 and lab.acquires == lab.releases == 2
    assert lab.manager_databases == [lab.db]
    for request in lab.requests:
        assert request["authorization"] == "Bearer synthetic-evaluation-key"
        assert request["url"] == "https://api.deepseek.com/v1/chat/completions"
        assert request["body"]["model"] == "synthetic-selected-model"
        assert request["body"]["top_p"] == 1 and request["body"]["thinking"] == {"type": "disabled"}
        assert request["body"]["temperature"] == 0 and request["body"]["max_tokens"] == 256
    second = lab.requests[1]["body"]["messages"]
    assert any(m["role"] == "user" and FIRST in m["content"] for m in second)
    assert {"role": "assistant", "content": ANSWERS[0]} in second and SECOND in second[-1]["content"]
    rows = lab.db.execute_sql("SELECT * FROM model_invocations ORDER BY id")
    assert len(rows) == 2 and [r["totalTokens"] for r in rows] == [16, 18]
    assert all(
        r["usageSource"] == "provider"
        and r["purpose"] == "generation_evaluation"
        and r["modelName"] == "synthetic-selected-model"
        for r in rows
    )
    lab.forbidden.assert_not_awaited()


async def test_auth_fault_stops_remaining_cases_without_mock_or_local(lab):
    lab.mode = "auth"
    await (await lab.start(max_prompts=2))
    row, result = lab.result()
    assert row["notes"] == "failed" and result["mock"] is False
    assert len(lab.requests) == 1 and result["total_prompts"] == 1 and result["unattempted_prompts"] == 1
    assert result["generation_errors"] == 1 and result["scored_responses"] == 0
    assert result["samples"][0]["detail"]["kind"] == "provider_auth_error"
    records = lab.db.execute_sql("SELECT usageSource,totalTokens,errorType FROM model_invocations")
    assert records == [{"usageSource": "unavailable", "totalTokens": None, "errorType": "ProviderResponseError"}]
    lab.forbidden.assert_not_awaited()


async def test_usage_failure_retains_received_turn_but_does_not_score_or_retry(lab, monkeypatch):
    writes = []

    def fail(record):
        writes.append(record)
        raise OSError("synthetic-private-ledger-failure")

    monkeypatch.setattr(lab.db, "add_model_invocation", fail)
    await (await lab.start(max_prompts=2))
    row, result = lab.result()
    assert row["notes"] == "failed" and len(writes) == len(lab.requests) == 1
    assert result["unattempted_prompts"] == 1 and result["scored_responses"] == 0
    sample = result["samples"][0]
    assert sample["turn_responses"] == ANSWERS[:1] and sample["response"] == ""
    assert sample["detail"]["invocationRecorded"] is None and sample["error_type"] == "UsageRecordingError"
    assert "synthetic-private" not in sample["error"]


async def test_selected_mock_provider_cannot_become_real_evaluation(lab):
    lab.selected = SimpleNamespace(name="mock", model="synthetic-mock")
    await (await lab.start())
    row, result = lab.result()
    assert row["notes"] == "failed" and result["mock"] is False
    assert "does not support selected provider" in result["error"]
    assert not lab.requests and lab.manager_databases == [lab.db]
    lab.forbidden.assert_not_awaited()


async def test_explicit_demo_is_labelled_and_does_not_resolve_real_provider(lab):
    await (await lab.start(mock=True))
    row, result = lab.result()
    assert row["notes"] == "completed" and result["mock"] is True
    assert not lab.requests and not lab.manager_databases
    lab.forbidden.assert_not_awaited()


async def test_selected_vllm_preserves_adapter_contract_and_unknown_token_usage(lab, monkeypatch):
    lab.selected = SimpleNamespace(name="vllm", model="synthetic-local-model")
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return ANSWERS[(len(calls) - 1) % 2]

    async def client():
        return SimpleNamespace(generate=generate)

    from api import generate as chat_api

    monkeypatch.setattr(chat_api, "get_vllm_client", client)
    monkeypatch.setattr(runtime_runner, "get_enabled_vllm_client", client, raising=False)
    await (await lab.start(adapter_name="controlled-adapter"))
    row, result = lab.result()
    assert row["notes"] == "completed" and result["provider"] == "vllm"
    assert len(calls) == 2 and all(c["lora_name"] == "controlled-adapter" and c["top_p"] == 0.9 for c in calls)
    records = lab.db.execute_sql("SELECT usageSource,totalTokens,usedLora FROM model_invocations")
    assert len(records) == 2 and all(
        r["usageSource"] == "unavailable" and r["totalTokens"] is None and r["usedLora"] == 1 for r in records
    )
    assert not lab.requests


async def test_cancelled_job_records_cancellation_and_preserves_usage(lab):
    lab.mode = "wait"
    task = await lab.start()
    await asyncio.wait_for(lab.entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row, result = lab.result()
    assert row["notes"] == "cancelled" and result["error_type"] == "CancelledError"
    assert lab.acquires == lab.releases == 1
    records = lab.db.execute_sql("SELECT usageSource,errorType FROM model_invocations")
    assert records == [{"usageSource": "unavailable", "errorType": "CancelledError"}]


async def test_result_publication_failure_propagates_without_regeneration(lab, monkeypatch):
    execute = lab.db.execute_sql

    def fail_update(query, params=None):
        if query.startswith("UPDATE gold_eval_runs"):
            raise OSError("synthetic-publication-failure")
        return execute(query, params)

    monkeypatch.setattr(lab.db, "execute_sql", fail_update)
    task = await lab.start()
    with pytest.raises(OSError, match="synthetic-publication-failure"):
        await task
    assert (
        len(lab.requests) == 2
        and lab.db.execute_sql("SELECT COUNT(*) AS count FROM model_invocations")[0]["count"] == 2
    )
    assert lab.result()[0]["notes"] == "queued"


async def test_held_out_filter_is_not_ignored_for_legacy_unlabelled_data(lab):
    await (await lab.start(split="held_out"))
    row, result = lab.result()
    assert row["notes"] == "failed" and "no evaluation prompts matched" in result["error"]
    assert not lab.requests and not lab.manager_databases
