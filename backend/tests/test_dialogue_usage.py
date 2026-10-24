"""Dialogue batches keep their full inputs and confirm accounting before publication."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from api import training
from app.dependencies import get_current_admin
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from inference import model_manager
from test_dialogue_generation_batch import complete_prompt_batch, mapped_messages, PLAN_80
from test_memory_source_inheritance import source_database as source_database

DESCRIPTION = '完全虚构角色青汐研究员；2031年以前不适用，仅限预约者，不能免除原件。'
RESEARCH = '完整合成资料：虚构青汐研究员只在2031年以后协助预约者，不得免除原件；雨天暂停办理。'
ENDPOINT = '/api/training/generate-dialogues'


@pytest.fixture
async def dialogue_usage(tmp_path, monkeypatch):
    database = SQLiteDB(tmp_path/'dialogue-usage.sqlite')
    config = {'openaiCompatBaseUrl': 'https://api.deepseek.com', 'openaiCompatApiKey': 'synthetic-only',
              'openaiCompatModel': 'deepseek-flash', 'temperature': .2, 'maxTokens': 512}
    monkeypatch.setattr(model_manager, '_get_db_config', lambda: dict(config))
    provider = model_manager.OpenAICompatProvider()
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda: SimpleNamespace(get_current_provider=lambda: provider))
    state = dict(is_generating=False, cancel_requested=False, progress=0, total=0, batch_num=0,
                 total_batches=0, generated_count=0, new_dialogues=[], all_generated_dialogues=[], started_at=None)
    monkeypatch.setattr(training, 'generation_state', state)
    lock = asyncio.Lock()
    monkeypatch.setattr(training, 'get_generation_state_lock', lambda: lock)
    monkeypatch.setattr(training, '_dialogue_generation_task', None)
    search = AsyncMock(return_value=RESEARCH)
    monkeypatch.setattr(training, 'search_character_info', search)
    rig = SimpleNamespace(database=database, state=state, requests=[], batches=[], mode='success',
                          entered=asyncio.Event(), release=asyncio.Event(), tasks=[], search=search)
    async def respond(request):
        wire = json.loads(request.content)
        prompt = wire['messages'][-1]['content']
        assert DESCRIPTION in prompt and RESEARCH in prompt
        assert wire['model'] == 'deepseek-flash' and wire['temperature'] == .2
        ordinal = len(rig.requests)+1
        count, turns, batch = complete_prompt_batch(prompt, ordinal)
        rig.requests.append({'wire': wire, 'count': count, 'turns': turns,
                             'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        rig.batches.append(batch)
        if ordinal == 2 and rig.mode == 'hold_second':
            rig.entered.set()
            await rig.release.wait()
        if ordinal == 2 and rig.mode == 'http_second':
            return httpx.Response(503, json={'error': {'message': 'synthetic private rejection'}})
        if ordinal == 2 and rig.mode == 'invalid_second':
            batch[0]['conversations'].pop()
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(batch,ensure_ascii=False)}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})
    @asynccontextmanager
    async def acquire(_self, timeout):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client
    monkeypatch.setattr(model_manager.OpenAICompatProvider, '_acquire_http_client', acquire)
    application = FastAPI()
    application.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False)
    application.include_router(training.router)
    application.dependency_overrides[get_current_admin] = lambda: {'user_id': 901, 'username': 'synthetic', 'role': 'admin'}
    rig.app = application
    async def post(count=2):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://isolated.test') as client:
            return await client.post(ENDPOINT, json={'character_description': DESCRIPTION,'num_dialogues': count,
                'style': '合成谨慎风格','custom_prompt': '保留时间、否定与雨天例外。'})
    rig.post = post
    try:
        yield rig
    finally:
        rig.release.set()
        task = training._dialogue_generation_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_eighty_complete_dialogues_keep_the_plan_and_every_measured_batch(dialogue_usage):
    rig = dialogue_usage
    response = await rig.post(80)
    assert response.status_code == 200 and response.json()['total'] == 80
    assert tuple((call['count'], call['turns']) for call in rig.requests) == PLAN_80
    expected = [messages for batch in rig.batches for messages in mapped_messages(batch)]
    assert [row['conversations'] for row in response.json()['dialogues']] == expected
    rows = rig.database.execute_sql('SELECT * FROM model_invocations ORDER BY id')
    assert len(rows) == len(rig.requests) == 16
    assert all(row['purpose'] == 'dialogue_generation' and row['totalTokens'] == 120 for row in rows)
    assert sum(row['totalTokens'] for row in rows) == 1920
    assert all(row['usageSource'] == 'provider' and row['modelName'] == 'deepseek-flash' for row in rows)
    assert len({row['traceId'] for row in rows}) == 16 and len({row['sessionId'] for row in rows}) == 1


@pytest.mark.parametrize('mode,kind,measured', [('invalid_second','dialogue_batch_invalid',120), ('http_second','provider_upstream_error',None)])
async def test_failed_batch_retains_usage_and_publishes_only_completed_batches(dialogue_usage, mode, kind, measured):
    rig = dialogue_usage
    rig.mode = mode
    response = await rig.post()
    assert response.status_code == 502
    detail = response.json()['detail']
    assert detail.get('error', detail.get('kind')) == kind
    assert detail['partial_result']['total'] == 1 and rig.state['generated_count'] == 1
    assert rig.state['all_generated_dialogues'] == detail['partial_result']['dialogues']
    rows = rig.database.execute_sql('SELECT * FROM model_invocations ORDER BY id')
    assert len(rows) == 2 and rows[0]['totalTokens'] == 120 and rows[1]['totalTokens'] == measured
    assert rows[1]['usageSource'] == ('provider' if measured else 'unavailable') and rows[1]['errorType']
    if mode == 'http_second':
        assert detail['upstreamStatus'] == 503 and rows[1]['errorType'] == 'ProviderResponseError'
    assert 'synthetic private' not in response.text


async def test_record_failure_does_not_publish_the_unconfirmed_batch(dialogue_usage, monkeypatch):
    rig = dialogue_usage
    original = rig.database.add_model_invocation
    def write(row):
        if len(rig.requests) == 2:
            raise OSError('synthetic private record path')
        original(row)
    attempts = Mock(side_effect=write)
    monkeypatch.setattr(rig.database, 'add_model_invocation', attempts)
    response = await rig.post()
    assert response.status_code == 503
    detail = response.json()['detail']
    assert detail['kind'] == 'usage_record_failed' and detail['partial_result']['total'] == 1
    assert rig.state['generated_count'] == 1 and rig.state['progress'] == 50
    assert len(rig.state['all_generated_dialogues']) == 1 and len(rig.requests) == attempts.call_count == 2
    assert len(rig.database.execute_sql('SELECT * FROM model_invocations')) == 1
    assert 'synthetic private' not in response.text


@pytest.mark.parametrize('user_cancel', [False, True])
async def test_cancel_keeps_confirmed_batches_and_records_the_inflight_attempt(dialogue_usage, user_cancel):
    rig = dialogue_usage
    rig.mode = 'hold_second'
    task = asyncio.create_task(rig.post())
    try:
        await asyncio.wait_for(rig.entered.wait(), 3)
        if user_cancel:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url='http://isolated.test') as client:
                cancelled = await client.post(ENDPOINT+'/cancel')
            assert cancelled.status_code == 200
            response = await task
            assert response.status_code == 200 and response.json()['cancelled'] and not response.json()['success']
            assert response.json()['total'] == 1
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        rows = rig.database.execute_sql('SELECT * FROM model_invocations ORDER BY id')
        assert len(rows) == 2 and rows[0]['totalTokens'] == 120 and rows[1]['totalTokens'] is None
        assert rows[1]['usageSource'] == 'unavailable' and rows[1]['errorType'] == 'CancelledError'
        assert rig.state['generated_count'] == 1 and not rig.state['is_generating']
    finally:
        rig.release.set()
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_batch_records_use_the_actual_application_database(dialogue_usage, source_database):
    rig = dialogue_usage
    rig.app.state.runtime_container.db = source_database
    maximum = source_database.execute_sql('SELECT COALESCE(MAX(id),0) AS maximum FROM model_invocations')[0]['maximum']
    response = await rig.post()
    assert response.status_code == 200 and response.json()['total'] == 2
    rows = source_database.execute_sql('SELECT * FROM model_invocations WHERE id > :maximum ORDER BY id', {'maximum': maximum})
    assert len(rows) == 2 and all(row['purpose'] == 'dialogue_generation' and row['totalTokens'] == 120 for row in rows)
    assert len({row['sessionId'] for row in rows}) == 1
    assert rig.database.execute_sql('SELECT * FROM model_invocations') == []


async def test_initial_progress_uses_the_same_plan_before_research_finishes(dialogue_usage, monkeypatch):
    rig = dialogue_usage
    entered = asyncio.Event()
    async def research(description):
        assert description == DESCRIPTION
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(training, 'search_character_info', research)
    task = asyncio.create_task(rig.post(80))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rig.app), base_url='http://isolated.test') as client:
            response = await client.get(ENDPOINT+'/progress')
        assert response.status_code == 200
        progress = response.json()
        assert progress['is_generating'] and progress['total'] == 80
        assert progress['total_batches'] == 16 and progress['batch_num'] == 0
        assert rig.requests == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert rig.database.execute_sql('SELECT * FROM model_invocations') == []
