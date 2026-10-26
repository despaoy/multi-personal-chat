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
from services.chat_answer import ChatAnswer
from stats_fixtures import fresh_observability as fresh_observability


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
@pytest.mark.parametrize('finish,expected_calls', [('stop', 2), ('length', 1)])
async def test_api_records_actual_usage_even_if_answer_rejected(installed, monkeypatch, tmp_path, finish, expected_calls):
    from api import generate
    from db.database import SQLiteDB
    from db.schemas import MessageRequest
    from fastapi import HTTPException
    from types import SimpleNamespace
    database = SQLiteDB(tmp_path / 'measured.sqlite')
    monkeypatch.setattr(generate, 'db', SimpleNamespace(config=installed.config, loras=[], add_model_invocation=database.add_model_invocation))
    installed.state.finish = finish
    compiled = chat_answer.generate_chat_answer

    async def twice(*args, **kwargs):
        await compiled(*args, **kwargs)
        return await compiled(*args, **kwargs)
    monkeypatch.setattr(chat_answer, 'generate_chat_answer', twice)
    request = MessageRequest(message='Synthetic facts: project A is approved; B is rejected. Which is approved?', loraName='default', traceId='usage-fixture')
    if finish == 'stop':
        await generate._generate_reply_impl(request, persist_message=False, message_db=generate.db)
    else:
        with pytest.raises(HTTPException):
            await generate._generate_reply_impl(request, persist_message=False, message_db=generate.db)
    rows = database.execute_sql('SELECT * FROM model_invocations')
    assert len(rows) == 1 and len(installed.calls) == expected_calls
    assert rows[0]['promptTokens'] == 100 * expected_calls
    assert rows[0]['completionTokens'] == 20 * expected_calls
    assert rows[0]['totalTokens'] == 120 * expected_calls
    assert rows[0]['usageSource'] == 'provider'
    assert bool(rows[0]['errorType']) == (finish == 'length')
    from services.stats import StatsService
    monkeypatch.setattr('services.stats.check_service', lambda _port: False)
    recorded = StatsService(database=database, inference_runtime=generate.inference_runtime, env={'ASTRBOT_ENABLED': 'false'}).collect()['metrics']
    assert recorded['chatGenerations'] == 1
    assert recorded['chatGenerationFailures'] == int(finish == 'length')
    assert recorded['chatGenerationFailureRate'] == float(finish == 'length')



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


@pytest.fixture
def publication_database(installed, monkeypatch, tmp_path):
    from api import generate
    from db.database import SQLiteDB

    database = SQLiteDB(tmp_path / "result-publication.sqlite")
    database.update_config(dict(installed.config, useKnowledgeBase=False))
    monkeypatch.setattr(generate, "db", database)
    try:
        yield database
    finally:
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("confidence", [{"invalid": "source confidence must be numeric"}, float("nan")])
async def test_invalid_result_stops_before_archive_and_accounts_once(installed, publication_database, monkeypatch, confidence):
    from fastapi import HTTPException
    from pydantic import ValidationError
    from api import generate
    from db.schemas import MessageRequest
    from test_compiled_cloud_transport import QUERY

    compiled = chat_answer.generate_chat_answer

    async def malformed(*args, **kwargs):
        _chat_result_metadata = await compiled(*args, **kwargs)
        reply, used_rag, metadata = _chat_result_metadata.reply, _chat_result_metadata.used_rag, _chat_result_metadata.metadata
        return ChatAnswer(reply=reply, used_rag=used_rag, model_invoked=_chat_result_metadata.model_invoked, metadata=dict(metadata, confidence=confidence))

    monkeypatch.setattr(chat_answer, "generate_chat_answer", malformed)
    request = MessageRequest(message=QUERY, loraName="default", platform="web", adapter="result-fixture",
        userId="synthetic-owner", senderId="synthetic-owner", sessionId="synthetic-result",
        conversationId="synthetic-result", sourceMessageId="synthetic-source", traceId="synthetic-result-trace")
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(request, enable_rag=False, message_db=publication_database)
    rows = publication_database.execute_sql('SELECT * FROM model_invocations')
    messages = publication_database.execute_sql('SELECT id FROM messages')
    actual = (caught.value.status_code, len(installed.calls), len(rows),
              sum(row["totalTokens"] for row in rows), len(messages))
    assert actual == (503, 1, 1, 120, 0)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert caught.value.detail["stage"] == "generation_result"
    assert caught.value.detail["reason"] == "invalid_generation_result"
    assert caught.value.detail["messageSaved"] is False
    assert caught.value.detail["fields"] == ["confidence"]
    assert rows[0]["errorType"] == "ValidationError" and rows[0]["usageSource"] == "provider"
    assert "source confidence" not in str(caught.value.detail)


@pytest.mark.asyncio
async def test_post_archive_error_does_not_count_usage_again(installed, publication_database, monkeypatch, fresh_observability):
    from fastapi import HTTPException
    from api import generate
    from db.schemas import MessageRequest
    from test_compiled_cloud_transport import QUERY, ANSWER

    original = generate.log_event
    failure = OSError("synthetic-private-publication-error")

    def event(name, **kwargs):
        if name == "message_generated":
            raise failure
        return original(name, **kwargs)

    monkeypatch.setattr(generate, "log_event", event)
    request = MessageRequest(message=QUERY, loraName="default", platform="web", adapter="result-fixture",
        userId="synthetic-owner", senderId="synthetic-owner", sessionId="synthetic-result",
        conversationId="synthetic-result", sourceMessageId="synthetic-source", traceId="synthetic-result-trace")
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(request, enable_rag=False, message_db=publication_database)
    rows = publication_database.execute_sql('SELECT * FROM model_invocations')
    messages = publication_database.execute_sql('SELECT * FROM messages')
    assert caught.value.status_code == 500 and caught.value.__cause__ is failure
    assert caught.value.detail["stage"] == "generation" and caught.value.detail["messageSaved"] is True
    assert "已存档" in caught.value.detail["message"]
    assert len(installed.calls) == len(rows) == len(messages) == 1
    assert rows[0]["totalTokens"] == 120 and rows[0]["errorType"] == ""
    assert messages[0]["message"] == QUERY and messages[0]["reply"] == ANSWER
    assert "synthetic-private" not in str(caught.value.detail)
    assert fresh_observability.get_counter("model_failures") == 0
    assert fresh_observability.get_consecutive("model_failure") == 0


@pytest.mark.asyncio
async def test_invalid_no_model_result_does_not_invent_an_invocation(installed, publication_database, monkeypatch):
    from fastapi import HTTPException
    from api import generate
    from db.schemas import MessageRequest

    async def unavailable(*args, **kwargs):
        return ChatAnswer(reply='当前没有可核对的第五条公告原文。', used_rag=False, model_invoked=False, metadata={'answerMode': 'source_unavailable', 'confidence': {'invalid': True}})

    monkeypatch.setattr(chat_answer, "generate_chat_answer", unavailable)
    request = MessageRequest(message="当前只提供了两条公告，未提供第五条。请读取第五条原文；没有原文就明确说明。",
        loraName="default", platform="web", adapter="result-fixture", senderId="synthetic-owner",
        conversationId="synthetic-result", sessionId="synthetic-result")
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(request, enable_rag=False, message_db=publication_database)
    assert caught.value.status_code == 503 and caught.value.detail["messageSaved"] is False
    assert not installed.calls
    assert publication_database.execute_sql('SELECT id FROM model_invocations') == []
    assert publication_database.execute_sql('SELECT id FROM messages') == []


@pytest.mark.asyncio
async def test_invalid_review_only_result_accounts_for_actual_provider(installed, publication_database, monkeypatch):
    from fastapi import HTTPException
    from api import generate
    from db.schemas import MessageRequest
    from inference.request_context import get_request_context
    from test_compiled_cloud_transport import QUERY

    async def reviewed(*args, **kwargs):
        await get_request_context().provider.async_complete(
            [{"role": "user", "content": "合成审核：只核对已提供的公告资料，不生成给用户的最终回答。" + QUERY}],
            temperature=0, max_tokens=173, top_p=1.0,
        )
        return ChatAnswer(reply='没有第五条公告原文。', used_rag=False, model_invoked=False, metadata={'answerMode': 'source_unavailable', 'confidence': {'invalid': True}})

    monkeypatch.setattr(chat_answer, "generate_chat_answer", reviewed)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message=QUERY, loraName="default"),
            enable_rag=False, message_db=publication_database)
    rows = publication_database.execute_sql('SELECT * FROM model_invocations')
    assert caught.value.status_code == 503 and caught.value.detail["messageSaved"] is False
    assert len(installed.calls) == len(rows) == 1
    assert rows[0]["modelName"] == "fixture-model" and rows[0]["totalTokens"] == 120
    assert rows[0]["errorType"] == "ValidationError"
    assert publication_database.execute_sql('SELECT id FROM messages') == []


@pytest.mark.asyncio
@pytest.mark.parametrize("last_failure", ["provider", "metadata"])
async def test_model_alert_tracks_actual_generation_failure(
    installed, publication_database, monkeypatch, fresh_observability, last_failure,
):
    from api import generate
    from db.schemas import MessageRequest
    from fastapi import HTTPException
    from services.stats import StatsService
    from test_compiled_cloud_transport import QUERY

    compiled = chat_answer.generate_chat_answer

    async def malformed(*args, **kwargs):
        _chat_result_metadata = await compiled(*args, **kwargs)
        reply, used_rag, metadata = _chat_result_metadata.reply, _chat_result_metadata.used_rag, _chat_result_metadata.metadata
        return ChatAnswer(reply=reply, used_rag=used_rag, model_invoked=_chat_result_metadata.model_invoked, metadata=dict(metadata, confidence={'invalid': True}))

    # Two real adapter failures establish a pending outage; the third request
    # distinguishes a continued outage from invalid metadata after recovery.
    for index in range(3):
        installed.state.status = 200 if index == 2 and last_failure == "metadata" else 503
        if installed.state.status == 200:
            monkeypatch.setattr(chat_answer, "generate_chat_answer", malformed)
        request = MessageRequest(message=QUERY, loraName="default", platform="web", adapter="alert-fixture",
            userId="synthetic-owner", senderId="synthetic-owner", sessionId="synthetic-alert",
            conversationId="synthetic-alert", traceId=f"synthetic-alert-{index}")
        with pytest.raises(HTTPException) as caught:
            await generate._generate_reply_impl(request, enable_rag=False, message_db=publication_database)
        if installed.state.status == 200:
            assert caught.value.status_code == 503 and caught.value.detail["stage"] == "generation_result"
        else:
            assert isinstance(caught.value.__cause__, RuntimeError)

    metrics = {"queueLength": 0, "queueMaxSize": 10, "p95ResponseTime": 0,
        "astrBotGateway": {"status": "running"}}
    alerts = StatsService(database=publication_database, inference_runtime=generate.inference_runtime, env={}).alerts(metrics)
    model_alerts = [alert for alert in alerts if alert["type"] == "model_consecutive_failures"]
    expected_failures = 3 if last_failure == "provider" else 2
    expected_consecutive = 3 if last_failure == "provider" else 0
    assert fresh_observability.get_counter("model_failures") == expected_failures
    assert fresh_observability.get_consecutive("model_failure") == expected_consecutive
    assert [alert["value"] for alert in model_alerts] == ([3] if last_failure == "provider" else [])
    assert len(installed.calls) == len(publication_database.execute_sql('SELECT id FROM model_invocations')) == 3
    assert publication_database.execute_sql('SELECT id FROM messages') == []
    monkeypatch.setattr("services.stats.check_service", lambda _port: False)
    recorded = StatsService(database=publication_database, inference_runtime=generate.inference_runtime,
        env={"ASTRBOT_ENABLED": "false"}).collect()["metrics"]
    assert recorded["chatGenerations"] == recorded["chatGenerationFailures"] == 3
    assert recorded["chatGenerationFailureRate"] == 1.0
    assert not {"modelFailureRate", "modelFailures", "modelInvocations"} & recorded.keys()



@pytest.mark.asyncio
async def test_incomplete_chat_result_cannot_publish_assumed_model_success(
    installed, publication_database, monkeypatch, fresh_observability,
):
    from api import generate
    from db.schemas import MessageRequest
    from fastapi import HTTPException
    from test_compiled_cloud_transport import ANSWER, QUERY

    async def incomplete(*args, **kwargs):
        # Deliberately omit the required invocation provenance at the service boundary.
        return ANSWER, False, {}

    monkeypatch.setattr(chat_answer, "generate_chat_answer", incomplete)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_reply_impl(MessageRequest(message=QUERY, loraName="default"),
            enable_rag=False, message_db=publication_database)
    assert isinstance(caught.value.__cause__, AttributeError)
    assert caught.value.detail["messageSaved"] is False
    assert not installed.calls and publication_database.execute_sql('SELECT id FROM messages') == []
    assert fresh_observability.get_counter("model_failures") == 0
