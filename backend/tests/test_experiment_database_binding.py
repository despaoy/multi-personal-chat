"""Application-owned experiments use real execution contracts and explicit failures."""
import asyncio
import json
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from api import experiments
from app.config import create_access_token
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from db.schemas import ExperimentStartRequest
from experiments import rag_ablation

RealAblation = rag_ablation.RAGAblation


@pytest.fixture
def stores(tmp_path, monkeypatch):
    dataset = dict(status='frozen', formal_use_allowed=True, questions=[dict(id='complete',
        question='完整资料：文档42规定课程12月20日开课，12月18日19点前确认。查这份安排。',
        expected_doc_ids=['42'], question_type='single_evidence')])
    path = tmp_path / 'frozen-complete.json'
    path.write_text(json.dumps(dataset, ensure_ascii=False))
    databases = [SQLiteDB(tmp_path / f'experiments-{i}.sqlite') for i in range(2)]
    for i, database in enumerate(databases):
        database.add_user('synthetic-admin', 'unused-signed-token-hash')
        database.execute_sql("INSERT INTO experiment_runs(id,experiment_type,hypothesis,status,started_at,results) "
            "VALUES ('shared','lora_ablation',:owner,'completed','2026-10-18',:results)",
            {'owner': str(i), 'results': json.dumps({'owner': str(i), 'source': dataset['questions'][0]['question']})})
    from db.adapter import db as global_database
    forbidden = Mock(side_effect=AssertionError('experiment used the global store'))
    monkeypatch.setattr(experiments, 'db', SimpleNamespace(execute_sql=forbidden, execute_sql_insert=forbidden), raising=False)
    monkeypatch.setattr(global_database, 'execute_sql', forbidden)
    monkeypatch.setattr(global_database, 'execute_sql_insert', forbidden)
    retrieve = Mock(return_value=[{'id': '42', 'content': '12月20日开课，12月18日19点前确认。'}])
    formal = []

    def make_rag(*args, formal=False, **kwargs):
        runner = RealAblation(str(path), formal=formal)
        runner._variant = lambda variant: retrieve
        return runner

    def tracked_rag(*args, **kwargs):
        formal.append(kwargs.get('formal', False))
        return make_rag(*args, **kwargs)

    class ControlledBenchmark:
        DEFAULT_CONFIGS = ['controlled']

        async def run_comparison(self, configs, mock=False):
            assert configs == self.DEFAULT_CONFIGS and mock
            return [SimpleNamespace(to_dict=lambda: {'controlled': True, 'mock': True})]

    quant = ModuleType('experiments.quantization_benchmark')
    quant.QuantizationBenchmark = ControlledBenchmark
    monkeypatch.setitem(sys.modules, 'experiments.quantization_benchmark', quant)
    monkeypatch.setattr(rag_ablation, 'RAGAblation', tracked_rag)
    try:
        yield SimpleNamespace(databases=databases, forbidden=forbidden, retrieve=retrieve, formal=formal,
                              path=path, dataset=dataset, make_rag=make_rag)
    finally:
        for database in databases:
            database.close_connection()


def client(database):
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=database, is_pg_mode=lambda: False, startup_env={})
    app.include_router(experiments.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://isolated',
        headers={'Authorization': 'Bearer ' + create_access_token('synthetic-admin', 1, 'admin')})


async def test_two_applications_keep_complete_experiments_and_reports_in_own_store(stores):
    for i, database in enumerate(stores.databases):
        async with client(database) as api:
            listed = await api.get('/api/experiments/')
            detail = await api.get('/api/experiments/shared')
            report = await api.get('/api/experiments/shared/report')
            assert listed.status_code == detail.status_code == report.status_code == 200
            assert listed.json()['experiments'][0]['results']['owner'] == detail.json()['experiment']['hypothesis'] == str(i)
            assert str(i) in report.json()['report']
            for route in ['lora-ablation', 'rag-ablation', 'quantization-benchmark']:
                response = await api.post('/api/experiments/' + route, json={'mock': route != 'rag-ablation'})
                assert response.status_code == 200 and response.json()['status'] == 'completed'
                identity = response.json()['experiment_id']
                record = database.execute_sql('SELECT status FROM experiment_runs WHERE id=:id', {'id': identity})[0]
                assert record['status'] == 'completed'
            assert database.execute_sql('SELECT count(*) AS n FROM experiment_runs')[0]['n'] == 4
    assert stores.formal == [True, True]
    stores.forbidden.assert_not_called()


@pytest.mark.parametrize('operation', ['read', 'insert'])
async def test_experiment_bound_store_fault_never_uses_global_database(stores, monkeypatch, operation):
    database = stores.databases[0]
    method = 'execute_sql' if operation == 'read' else 'execute_sql_insert'
    monkeypatch.setattr(database, method, Mock(side_effect=OSError('synthetic-bound-store-down')))
    async with client(database) as api:
        response = await api.get('/api/experiments/') if operation == 'read' else await api.post('/api/experiments/rag-ablation', json={'mock': False})
    assert response.status_code == 503
    assert stores.formal == []
    stores.forbidden.assert_not_called()


@pytest.mark.parametrize('stored', ['{broken-json', '', 'null', '1'])
async def test_invalid_experiment_results_cannot_be_read_as_success(stores, stored):
    database = stores.databases[0]
    database.execute_sql('UPDATE experiment_runs SET results=:value', {'value': stored})
    async with client(database) as api:
        for url in ['/api/experiments/', '/api/experiments/shared', '/api/experiments/shared/report']:
            assert (await api.get(url)).status_code == 503


async def test_legacy_result_arrays_nullable_fields_and_empty_lists_remain_valid(stores):
    database = stores.databases[0]
    async with client(database) as api:
        for value, expected in [(None, {}), ('[]', []), ('[{"source":"完整旧实验结果","score":1.0}]', [{'source': '完整旧实验结果', 'score': 1.0}])]:
            database.execute_sql('UPDATE experiment_runs SET results=:value', {'value': value})
            listed = await api.get('/api/experiments/')
            detail = await api.get('/api/experiments/shared')
            report = await api.get('/api/experiments/shared/report')
            assert listed.status_code == detail.status_code == report.status_code == 200
            assert listed.json()['experiments'][0]['results'] == detail.json()['experiment']['results'] == expected
        database.execute_sql('DELETE FROM experiment_runs')
        empty = await api.get('/api/experiments/')
        assert empty.status_code == 200 and empty.json() == {'success': True, 'experiments': [], 'total': 0}
        assert (await api.get('/api/experiments/missing')).status_code == 404


@pytest.mark.parametrize('defect', ['retrieval', 'missing_questions'])
async def test_actual_ablation_failure_marks_record_failed_and_keeps_original_error(stores, defect):
    error = OSError('synthetic-retrieval-down')
    if defect == 'retrieval':
        stores.retrieve.side_effect = error
    else:
        stores.path.write_text(json.dumps({'status': 'frozen', 'formal_use_allowed': True}))
    database = stores.databases[0]
    with pytest.raises(HTTPException) as caught:
        await experiments.start_rag_ablation(ExperimentStartRequest(mock=False), current_user={'role': 'admin'}, database=database)
    assert caught.value.status_code == 500
    if defect == 'retrieval':
        assert caught.value.__cause__ is error
    else:
        assert isinstance(caught.value.__cause__, KeyError)
    assert database.execute_sql("SELECT status FROM experiment_runs WHERE id!='shared'") == [{'status': 'failed'}]


async def test_actual_formal_ablation_rejects_unreviewed_dataset(stores):
    stores.path.write_text(json.dumps(dict(stores.dataset, status='candidate', formal_use_allowed=False)))
    database = stores.databases[0]
    async with client(database) as api:
        response = await api.post('/api/experiments/rag-ablation', json={'mock': False})
    assert response.status_code == 409 and 'reviewed and frozen' in response.json()['detail']
    stores.retrieve.assert_not_called()
    assert database.execute_sql("SELECT status FROM experiment_runs WHERE id!='shared'") == [{'status': 'failed'}]


def test_genuine_empty_retrieval_is_zero_score_not_a_system_failure(stores):
    stores.retrieve.return_value = []
    result = stores.make_rag(formal=True).run_variant('vector_only')
    assert result.recall_at_1 == result.recall_at_5 == result.mrr == 0
    assert result.failed_queries == 0 and result.per_question[0]['error'] == ''


async def test_cancellation_is_not_converted_to_an_empty_result_or_http_error(stores):
    error = asyncio.CancelledError('controlled-cancellation')
    stores.retrieve.side_effect = error
    with pytest.raises(asyncio.CancelledError) as caught:
        await experiments.start_rag_ablation(ExperimentStartRequest(mock=False), current_user={'role': 'admin'}, database=stores.databases[0])
    assert caught.value is error
    row = stores.databases[0].execute_sql("SELECT status,results FROM experiment_runs WHERE id!='shared'")[0]
    assert row['status'] == 'failed' and json.loads(row['results'])['error'] == 'cancelled'
    assert json.loads(row['results'])['mock'] is False


@pytest.mark.parametrize('stage', ['work', 'work-error', 'insert', 'commit'])
async def test_http_cancellation_joins_work_and_publishes_only_confirmed_state(stores, monkeypatch, stage):
    database = stores.databases[0]
    original_execute = database.execute_sql
    original_insert = database.execute_sql_insert
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    fault = OSError('synthetic-worker-fault-after-cancellation')

    def hold(func, *args, **kwargs):
        started.set()
        assert release.wait(3), 'controlled operation was not released'
        try:
            if stage == 'work-error':
                raise fault
            return func(*args, **kwargs)
        finally:
            finished.set()

    if stage in {'work', 'work-error'}:
        stores.retrieve.side_effect = lambda *args: hold(lambda: [{'id': '42', 'content': '12月20日开课，12月18日19点前确认。'}])
    elif stage == 'insert':
        monkeypatch.setattr(database, 'execute_sql_insert', lambda *args, **kwargs: hold(original_insert, *args, **kwargs))
    else:
        def committing(query, params=None):
            if "SET status='completed'" in query:
                return hold(original_execute, query, params)
            return original_execute(query, params)
        monkeypatch.setattr(database, 'execute_sql', committing)
    request = None
    try:
        async with client(database) as api:
            request = asyncio.create_task(api.post('/api/experiments/rag-ablation', json={'mock': False}))
            assert await asyncio.wait_for(asyncio.to_thread(started.wait, 2), 3)
            request.cancel('controlled-user-cancel')
            await asyncio.wait({request}, timeout=.05)
            assert not request.done() and not finished.is_set()
            if stage == 'work':
                request.cancel('controlled-second-cancel')
                await asyncio.wait({request}, timeout=.05)
                assert not request.done()
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await request
            assert caught.value.args == ('controlled-user-cancel',) and finished.is_set()
            if stage == 'work-error':
                assert caught.value.__cause__ is fault
        row = original_execute("SELECT status,results FROM experiment_runs WHERE id!='shared'")[0]
        if stage == 'commit':
            assert row['status'] == 'completed' and json.loads(row['results'])['formal']
        else:
            assert row['status'] == 'failed' and json.loads(row['results']) == {'mock': False, 'error': 'cancelled'}
        if stage in {'work', 'work-error'}:
            assert stores.retrieve.call_count == 1
        if stage == 'insert':
            assert stores.formal == []
    finally:
        release.set()
        if request is not None:
            await asyncio.gather(request, return_exceptions=True)


async def test_cancellation_cleanup_write_failure_remains_an_explicit_error(stores, monkeypatch):
    database = stores.databases[0]
    original_execute = database.execute_sql
    cancellation = asyncio.CancelledError('controlled-worker-cancel')
    failure = OSError('synthetic-cancellation-record-unavailable')
    stores.retrieve.side_effect = cancellation

    def fail_cancel_record(query, params=None):
        if "AND status='running'" in query:
            raise failure
        return original_execute(query, params)

    monkeypatch.setattr(database, 'execute_sql', fail_cancel_record)
    with pytest.raises(OSError) as caught:
        await experiments.start_rag_ablation(ExperimentStartRequest(mock=False), current_user={'role': 'admin'}, database=database)
    assert caught.value is failure and caught.value.__context__ is cancellation
    assert original_execute("SELECT status FROM experiment_runs WHERE id!='shared'") == [{'status': 'running'}]
