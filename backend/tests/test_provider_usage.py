"""Provider-reported counts, including failed answers and concurrent requests."""

from services import chat_answer
import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from test_memory_source_inheritance import source_database as source_database
from test_compiled_cloud_transport import installed as installed

from inference.model_manager import OpenAICompatProvider
from inference.usage import begin_completion, collect_usage, invocation_usage


def payload(prompt=1000, completion=100):
    return {"usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}


def provider(monkeypatch, response):
    instance = OpenAICompatProvider({"openaiCompatBaseUrl": "https://fixture.invalid", "openaiCompatApiKey": "synthetic-only", "openaiCompatModel": "fixture"})
    @asynccontextmanager
    async def acquire(timeout):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
            yield client
    monkeypatch.setattr(instance, "_acquire_http_client", acquire)
    return instance


@pytest.mark.asyncio
async def test_two_actual_adapter_responses_preserve_exact_usage(monkeypatch):
    response = {**payload(), "choices": [{"finish_reason": "stop", "message": {"content": "Synthetic complete answer."}}]}
    adapter = provider(monkeypatch, httpx.Response(200, json=response))
    with collect_usage():
        for _ in range(2):
            reply, elapsed = await adapter.async_complete([{"role": "user", "content": "Use supplied synthetic facts."}], temperature=0, max_tokens=200)
            assert reply == "Synthetic complete answer." and elapsed >= 0
        assert invocation_usage() == {"promptTokens": 2000, "completionTokens": 200, "totalTokens": 2200, "usageSource": "provider"}


@pytest.mark.asyncio
async def test_truncated_answer_keeps_received_usage(monkeypatch):
    response = {**payload(), "choices": [{"finish_reason": "length", "message": {"content": "Partial"}}]}
    adapter = provider(monkeypatch, httpx.Response(200, json=response))
    with collect_usage():
        with pytest.raises(ValueError, match="finish"):
            await adapter.async_complete([{"role": "user", "content": "Synthetic facts"}], temperature=0, max_tokens=200)
        assert invocation_usage()["totalTokens"] == 1100


@pytest.mark.asyncio
async def test_failed_followup_preserves_known_counts_as_partial(monkeypatch):
    adapter = provider(monkeypatch, httpx.Response(503))
    with collect_usage():
        begin_completion().read(payload())
        with pytest.raises(RuntimeError):
            await adapter.async_complete([{"role": "user", "content": "Synthetic facts"}], temperature=0, max_tokens=200)
        assert invocation_usage() == {"promptTokens": 1000, "completionTokens": 100, "totalTokens": 1100, "usageSource": "partial"}


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2}, {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0}, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3}])
def test_missing_or_malformed_usage_is_not_estimated(usage):
    with collect_usage():
        with pytest.raises((KeyError, TypeError, ValueError)):
            begin_completion().read({"usage": usage})
        assert invocation_usage() == {"promptTokens": None, "completionTokens": None, "totalTokens": None, "usageSource": "unavailable"}


@pytest.mark.asyncio
async def test_parallel_requests_and_nested_collection_are_isolated():
    async def request(count):
        with collect_usage():
            attempt = begin_completion()
            await asyncio.sleep(0)
            attempt.read(payload(count, 2))
            with collect_usage():
                begin_completion().read(payload(999, 1))
                assert invocation_usage()["totalTokens"] == 1000
            return invocation_usage()["totalTokens"]
    assert await asyncio.gather(request(10), request(20)) == [12, 22]
    assert invocation_usage()["usageSource"] == "unavailable"
    assert begin_completion() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,expected_calls", [("stop", 2), ("length", 1)])
async def test_api_records_actual_usage_even_if_answer_rejected(installed, monkeypatch, tmp_path, finish, expected_calls):
    from api import generate
    from db.database import SQLiteDB
    from db.schemas import MessageRequest
    from fastapi import HTTPException
    from types import SimpleNamespace
    database = SQLiteDB(tmp_path / "measured.sqlite")
    monkeypatch.setattr(generate, "db", SimpleNamespace(config=installed.config, loras=[], add_model_invocation=database.add_model_invocation))
    monkeypatch.setattr(chat_answer, "public_knowledge_db", generate.db)
    installed.state.finish = finish
    compiled = chat_answer.generate_chat_answer
    async def twice(*args, **kwargs):
        await compiled(*args, **kwargs)
        return await compiled(*args, **kwargs)
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', twice)
    request = MessageRequest(message="Synthetic facts: project A is approved; B is rejected. Which is approved?", loraName="default", traceId="usage-fixture")
    if finish == "stop":
        await generate._generate_reply_impl(request, persist_message=False)
    else:
        with pytest.raises(HTTPException):
            await generate._generate_reply_impl(request, persist_message=False)
    rows = database.execute_sql("SELECT * FROM model_invocations")
    assert len(rows) == 1 and len(installed.calls) == expected_calls
    assert rows[0]["promptTokens"] == 100 * expected_calls
    assert rows[0]["completionTokens"] == 20 * expected_calls
    assert rows[0]["totalTokens"] == 120 * expected_calls
    assert rows[0]["usageSource"] == "provider"
    assert bool(rows[0]["errorType"]) == (finish == "length")


def test_source_migration_keeps_legacy_values_and_is_idempotent(monkeypatch):
    import importlib.util
    from pathlib import Path
    from sqlalchemy import create_engine, text
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    spec = importlib.util.spec_from_file_location("usage_migration", Path(__file__).parents[1] / "alembic/versions/015_invocation_usage_source.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE model_invocations ("totalTokens" INTEGER)'))
        connection.execute(text('INSERT INTO model_invocations VALUES (17)'))
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        migration.upgrade()
        assert connection.execute(text('SELECT "totalTokens", "usageSource" FROM model_invocations')).one() == (17, "legacy_unverified")
    engine.dispose()


def test_detached_background_context_does_not_charge_parent_request():
    from inference.request_context import detached_request_context
    with collect_usage():
        begin_completion().read(payload(10, 2))
        context = detached_request_context()
        assert context.run(begin_completion) is None
        assert context.run(invocation_usage)["usageSource"] == "unavailable"
        assert invocation_usage()["totalTokens"] == 12


@pytest.mark.parametrize("source,counts", [("provider", (2000, 200, 2200)), ("partial", (1000, 100, 1100)), ("unavailable", (None, None, None))])
def test_both_databases_preserve_measurement_and_unknown(source_database, source, counts):
    import uuid
    trace = "usage-" + uuid.uuid4().hex
    source_database.add_model_invocation({"traceId": trace, "promptTokens": counts[0], "completionTokens": counts[1], "totalTokens": counts[2], "usageSource": source})
    rows = source_database.execute_sql('SELECT * FROM model_invocations WHERE "traceId" = :trace', {"trace": trace})
    assert len(rows) == 1
    assert tuple(rows[0][key] for key in ("promptTokens", "completionTokens", "totalTokens")) == counts
    assert rows[0]["usageSource"] == source
