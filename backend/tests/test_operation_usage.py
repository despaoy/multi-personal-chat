"""Independent operation accounting with real adapters, stores and source evidence."""

import asyncio
import importlib.util
import json
import uuid
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine, text
from alembic.migration import MigrationContext
from alembic.operations import Operations

from api import ask
from app.dependencies import get_current_user
from app.main import global_exception_handler
from app.runtime import RuntimeContainer
from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, OpenAICompatibleMemoryCompletion
from inference.usage import begin_completion, collect_usage, invocation_usage
from services.model_usage import UsageRecordingError, account_usage
from services.stats import StatsService
from stats_fixtures import StatsRig
from test_ask_request_context import ask_rig as ask_rig, request, USER, ANSWER, decode_event
from test_memory_retry_policy import delivered_memory as delivered_memory, proposal
from test_memory_source_inheritance import source_database as source_database

pytestmark = pytest.mark.usefixtures('literal_grounded_query')


def usage(prompt=100, completion=20):
    return {'usage': {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt+completion}}


@pytest.mark.parametrize('failure', [False, True])
async def test_operation_totals_and_failures_are_preserved_on_both_databases(source_database, failure):
    trace = 'operation-' + uuid.uuid4().hex
    async def run():
        async with account_usage(source_database, {'traceId': trace, 'purpose': 'memory_judgement', 'modelName': 'synthetic-model'}):
            begin_completion().read(usage())
            begin_completion().read(usage(200, 40))
            if failure:
                raise ValueError('synthetic invalid model answer')
    if failure:
        with pytest.raises(ValueError, match='invalid model answer'):
            await run()
    else:
        await run()
    rows = source_database.execute_sql('SELECT * FROM model_invocations WHERE "traceId" = :trace', {'trace': trace})
    assert len(rows) == 1
    assert (rows[0]['promptTokens'], rows[0]['completionTokens'], rows[0]['totalTokens']) == (300, 60, 360)
    assert rows[0]['usageSource'] == 'provider' and rows[0]['purpose'] == 'memory_judgement'
    assert rows[0]['errorType'] == ('ValueError' if failure else '')


async def test_ask_records_actual_completion_once_and_cache_hit_does_not_add_usage(ask_rig):
    rig = ask_rig
    first = await ask.ask(request(), current_user=USER, database=rig.database)
    cached = await ask.ask(request(), current_user=USER, database=rig.database)
    assert first.answer == cached.answer == ANSWER
    assert first.modelInvoked and cached.cacheHit and not cached.modelInvoked
    rows = rig.database.execute_sql('SELECT * FROM model_invocations')
    assert len(rows) == 1 and len(rig.calls) == 1
    assert rows[0]['totalTokens'] == 120 and rows[0]['usageSource'] == 'provider'
    assert rows[0]['purpose'] == 'knowledge_answer' and rows[0]['modelName'] == first.model
    assert rows[0]['sessionId'] == 'ask:1'


async def test_actual_fastapi_dependency_selects_the_application_database(ask_rig):
    rig = ask_rig
    application = FastAPI()
    application.state.runtime_container = RuntimeContainer(db=rig.database, is_pg_mode=lambda: False)
    application.include_router(ask.router)
    application.dependency_overrides[get_current_user] = lambda: USER
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://isolated.test') as client:
        result = await client.post('/api/ask', json=request().model_dump())
    assert result.status_code == 200 and result.json()['answer'] == ANSWER
    rows = rig.database.execute_sql('SELECT * FROM model_invocations')
    assert len(rows) == 1 and rows[0]['totalTokens'] == 120


async def test_stream_is_recorded_before_the_done_receipt_is_emitted(ask_rig):
    rig = ask_rig
    response = await ask.ask_stream(request(), rig.http_request, current_user=USER, database=rig.database)
    events = []
    async for chunk in response.body_iterator:
        event = decode_event(chunk)
        events.append(event)
        if event['type'] == 'done':
            rows = rig.database.execute_sql('SELECT * FROM model_invocations')
            assert len(rows) == 1 and rows[0]['totalTokens'] == 120
    assert events[-1]['type'] == 'done' and all(event['type'] != 'error' for event in events)


@pytest.mark.parametrize('streaming', [False, True])
async def test_record_failure_cannot_return_an_overall_success(ask_rig, monkeypatch, streaming):
    rig = ask_rig
    write = Mock(side_effect=OSError('synthetic-private-record-path'))
    monkeypatch.setattr(rig.database, 'add_model_invocation', write)
    if streaming:
        response = await ask.ask_stream(request(), rig.http_request, current_user=USER, database=rig.database)
        events = [decode_event(chunk) async for chunk in response.body_iterator]
        assert events[-1]['type'] == 'error' and events[-1]['kind'] == 'usage_record_failed'
        assert all(event['type'] != 'done' for event in events)
        assert 'synthetic-private' not in str(events)
    else:
        with pytest.raises(UsageRecordingError) as caught:
            await ask.ask(request(), current_user=USER, database=rig.database)
        response = await global_exception_handler(None, caught.value)
        assert response.status_code == 503
        assert json.loads(response.body)['kind'] == 'usage_record_failed'
        assert b'synthetic-private' not in response.body
    assert rig.service.cache.stats()['entries'] == 0
    assert len(rig.calls) == write.call_count == 1
    assert rig.database.execute_sql('SELECT * FROM model_invocations') == []


@pytest.mark.parametrize('write_failure', [False, True])
async def test_cancelled_ask_accounts_unknown_usage_and_still_cancels(ask_rig, monkeypatch, write_failure):
    rig = ask_rig
    rig.hold = True
    if write_failure:
        monkeypatch.setattr(rig.database, 'add_model_invocation', Mock(side_effect=OSError('synthetic write failure')))
    task = asyncio.create_task(ask.ask(request(), current_user=USER, database=rig.database))
    await asyncio.wait_for(rig.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    rows = rig.database.execute_sql('SELECT * FROM model_invocations')
    if write_failure:
        assert isinstance(caught.value.__cause__, UsageRecordingError) and rows == []
    else:
        assert len(rows) == 1 and rows[0]['totalTokens'] is None
        assert rows[0]['usageSource'] == 'unavailable' and rows[0]['errorType'] == 'CancelledError'


@pytest.mark.parametrize('outcome', ['saved', 'invalid_json', 'http503', 'record_failure', 'invalid_kind', 'invalid_qualifier'])
async def test_background_memory_records_its_own_usage_without_charging_the_parent(delivered_memory, monkeypatch, outcome):
    rig = delivered_memory
    await rig.scheduler.shutdown()
    config = MemoryLlmConfig(True, 'https://api.deepseek.com', 'deepseek-flash', api_key='synthetic-only',
                             queue_size=2, idle_seconds=0)
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        if outcome == 'http503':
            return httpx.Response(503, json={'error': {'message': 'synthetic failure'}})
        content = '{broken' if outcome == 'invalid_json' else proposal(rig.body)
        if outcome in {'invalid_kind', 'invalid_qualifier'}:
            batch = json.loads(content)
            invalid = dict(batch['memories'][0])
            invalid.update(kind='user_fact') if outcome == 'invalid_kind' else invalid.update(qualifiers={'type': 'undeclared'})
            batch['memories'].append(invalid)
            content = json.dumps(batch)

        return httpx.Response(200, json={**usage(), 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})
    completion = OpenAICompatibleMemoryCompletion(config)
    completion._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    rig.completion = completion
    rig.scheduler = MemoryEnrichmentScheduler(config=config, completion=completion,
        embedding_provider=rig.scheduler._embedding_provider)
    if outcome == 'record_failure':
        monkeypatch.setattr(rig.database, 'add_model_invocation', Mock(side_effect=OSError('synthetic record failure')))
    with collect_usage():
        begin_completion().read(usage(10, 2))
        response = await rig.step()
        assert invocation_usage()['totalTokens'] == 12
    receipt = response['context']['memory_completion']['semantic_receipt']
    rows = rig.database.execute_sql('SELECT * FROM model_invocations')
    assert len(calls) == 1
    if outcome == 'record_failure':
        assert rows == [] and receipt['status'] == 'failed' and receipt['persisted'] == 0
        assert receipt['stage'] == 'model_invocation_record' and not receipt['retryable']
    else:
        assert len(rows) == 1 and rows[0]['purpose'] == 'memory_judgement'
        assert rows[0]['modelName'] == 'deepseek-flash'
        assert rows[0]['totalTokens'] == (None if outcome == 'http503' else 120)
        assert rows[0]['usageSource'] == ('unavailable' if outcome == 'http503' else 'provider')
        assert (receipt['status'], receipt['persisted']) == (('saved', 1) if outcome == 'saved' else ('failed', 0))
        if outcome in {'invalid_json', 'invalid_kind', 'invalid_qualifier'}:
            assert rows[0]['errorType'] == 'MemoryResponseError'
            field = {'invalid_json': 'response', 'invalid_kind': 'kind', 'invalid_qualifier': 'qualifiers'}[outcome]
            assert receipt['reason'] == 'invalid_model_response' and receipt['field'] == field
            assert receipt['stage'] == 'proposal_validation' and not receipt['retryable']
            assert await rig.repo.list_memory_records(rig.snapshot.character_id, rig.snapshot.user_scope,
                                                      include_inactive=True) == []
            assert not rig.selectable()



def test_background_and_knowledge_records_do_not_change_chat_metrics(tmp_path):
    rig = StatsRig(tmp_path/'statistics.sqlite')
    rig.database.add_model_invocation({'costTime': 2.0})
    for purpose in ['memory_judgement', 'knowledge_answer']:
        rig.database.add_model_invocation({'purpose': purpose, 'costTime': 9000.0, 'errorType': 'synthetic'})
    metrics = StatsService(database=rig.database, inference_runtime=rig.queue, env={'ASTRBOT_ENABLED': 'false'}).collect()['metrics']
    assert metrics['chatGenerations'] == 1 and metrics['chatGenerationFailures'] == 0
    assert metrics['chatGenerationFailureRate'] == 0 and metrics['p95ResponseTime'] == metrics['p99ResponseTime'] == 2.0
    assert len(rig.database.execute_sql('SELECT * FROM model_invocations')) == 3


def test_purpose_migration_preserves_existing_usage_and_can_run_twice(monkeypatch):
    path = Path(__file__).parents[1]/'alembic/versions/016_invocation_purpose.py'
    spec = importlib.util.spec_from_file_location('operation_usage_migration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = create_engine('sqlite://')
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE model_invocations ("totalTokens" INTEGER)'))
        connection.execute(text('INSERT INTO model_invocations VALUES (37)'))
        monkeypatch.setattr(module, 'op', Operations(MigrationContext.configure(connection)))
        module.upgrade()
        module.upgrade()
        assert connection.execute(text('SELECT "totalTokens", purpose FROM model_invocations')).one() == (37, 'chat')
    engine.dispose()


async def test_source_changed_during_confirmation_is_not_published_or_cached(tmp_path, monkeypatch):
    from test_grounded_source_lifecycle import arrange_grounded_sources, BODY_B
    from knowledge.curated_sources import CuratedSourceChangedError

    rig = arrange_grounded_sources(tmp_path, monkeypatch)
    async def confirm():
        await asyncio.sleep(0)
        rig.path.write_bytes(BODY_B.encode('utf-8'))
    with pytest.raises(CuratedSourceChangedError):
        await rig.service.answer(rig.query, domain_id=rig.domain_id, top_k=1,
                                 generate=rig.generate, model_id='synthetic', before_publish=confirm)
    assert len(rig.generate_calls) == 1 and rig.service.cache.stats()['entries'] == 0


async def test_timed_out_record_is_not_retried_and_late_commit_remains_single(tmp_path, monkeypatch):
    import threading
    from db.database import SQLiteDB
    from services import model_usage

    database = SQLiteDB(tmp_path/'late-record.sqlite')
    writer = database.add_model_invocation
    release, committed = threading.Event(), threading.Event()
    attempts = []
    def delayed(record):
        attempts.append(record)
        assert release.wait(2)
        writer(record)
        committed.set()
    monkeypatch.setattr(database, 'add_model_invocation', delayed)
    monkeypatch.setattr(model_usage, '_WRITE_TIMEOUT', .01)
    try:
        with pytest.raises(UsageRecordingError) as caught:
            async with account_usage(database, {'traceId': 'late-synthetic', 'purpose': 'knowledge_answer'}) as confirm:
                begin_completion().read(usage())
                await confirm()
        assert caught.value.public_detail()['invocationRecorded'] is None
        assert len(attempts) == 1
    finally:
        release.set()
    assert await asyncio.to_thread(committed.wait, 2)
    rows = database.execute_sql('SELECT * FROM model_invocations')
    assert len(rows) == len(attempts) == 1 and rows[0]['totalTokens'] == 120
