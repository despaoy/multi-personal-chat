"""A complete dialogue task keeps its selected resource across awaited work."""

from infra.resource_pool import HttpClientPool

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import Mock

import httpx
import pytest
from app.runtime import RuntimeContainer
from db.database import SQLiteDB
from fastapi import FastAPI
from test_dialogue_generation_batch import complete_prompt_batch, mapped_messages

from api import training
from app.dependencies import get_current_admin
from inference import model_manager
from inference.request_context import get_request_context

DESCRIPTION = "完全虚构角色青汐研究员；2031年以前不适用，仅限预约者，不能免除原件。"


@pytest.mark.asyncio
@pytest.mark.parametrize('selected,switch_at,upstream_failure', [('openai_compat', 'research', False), ('openai_compat', 'first_batch', False), ('ollama', 'research', False), ('openai_compat', 'first_batch', True)])
async def test_dialogue_task_keeps_owner_and_reports_real_failure(monkeypatch, tmp_path, selected, switch_at, upstream_failure):
    config = dict(openaiCompatBaseUrl='https://api.deepseek.com', openaiCompatModel='deepseek-flash', openaiCompatApiKey='synthetic-dialogue-original-key', temperature=0.2, maxTokens=512)
    monkeypatch.setattr(model_manager, '_get_db_config', lambda _owner_database=None: dict(config))
    cloud = model_manager.OpenAICompatProvider()
    snapshots = Mock(wraps=cloud.snapshot)
    monkeypatch.setattr(cloud, 'snapshot', snapshots)
    calls, batches = ([], [])
    manager = model_manager.ModelManager.__new__(model_manager.ModelManager)
    manager._current_provider = model_manager.ModelProvider(selected)

    def change_default():
        manager.set_provider(model_manager.ModelProvider.OLLAMA if selected == 'openai_compat' else model_manager.ModelProvider.OPENAI_COMPAT)
        config.update(openaiCompatBaseUrl='https://changed.invalid/v1', openaiCompatModel='changed-model', openaiCompatApiKey='synthetic-dialogue-new-key', temperature=0.9)

    def completed(name, prompt):
        assert DESCRIPTION in prompt
        assert '未经独立核验的外部资料' in prompt
        count, turns, batch = complete_prompt_batch(prompt, len(calls) + 1)
        calls.append(dict(provider=name, count=count, turns=turns))
        batches.append(batch)
        return (json.dumps(batch, ensure_ascii=False), turns)

    def respond(request):
        body = json.loads(request.content)
        raw, turns = completed('openai_compat', body['messages'][-1]['content'])
        calls[-1].update(url=str(request.url), model=body['model'], auth=request.headers['Authorization'], temperature=body['temperature'])
        assert body['max_tokens'] == (2048 if turns <= 2 else 4096 if turns <= 5 else 6144)
        if switch_at == 'first_batch' and len(calls) == 1:
            change_default()
        if upstream_failure and len(calls) == 2:
            return httpx.Response(503, text='synthetic-private-upstream-detail')
        return httpx.Response(200, json={'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}, 'choices': [{'finish_reason': 'stop', 'message': {'content': raw}}]})

    @asynccontextmanager
    async def acquire(_self, timeout, **_resource_options):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client
    monkeypatch.setattr(model_manager.OpenAICompatProvider, '_acquire_http_client', acquire)

    class Local(model_manager.BaseProvider):

        def __init__(self):
            super().__init__('ollama')

        def generate(self, *_args, **_kwargs):
            raise AssertionError('This task requires asynchronous generation')

        async def async_generate(self, prompt, session_history, max_tokens_override):
            assert session_history == []
            raw, turns = completed('ollama', prompt)
            assert max_tokens_override == (2048 if turns <= 2 else 4096 if turns <= 5 else 6144)
            return (raw, 0.0)
    manager._providers = {model_manager.ModelProvider.OPENAI_COMPAT: cloud, model_manager.ModelProvider.OLLAMA: Local()}
    reads = Mock(wraps=manager.get_current_provider)
    monkeypatch.setattr(manager, 'get_current_provider', reads)
    legacy = Mock(wraps=manager.async_generate)
    monkeypatch.setattr(manager, 'async_generate', legacy)
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: manager)

    async def research(description):
        assert description == DESCRIPTION
        await asyncio.sleep(0)
        if switch_at == 'research':
            change_default()
        return '合成资料：2031年以前不适用，仅限预约者，不能免除原件。'
    state = dict(is_generating=False, cancel_requested=False, progress=0, total=0, batch_num=0, total_batches=0, generated_count=0, new_dialogues=[], all_generated_dialogues=[], started_at=None)
    monkeypatch.setattr(training, 'generation_state', state)
    lock = asyncio.Lock()
    monkeypatch.setattr(training, 'get_generation_state_lock', lambda: lock)
    monkeypatch.setattr(training, '_dialogue_generation_task', None)
    monkeypatch.setattr(training, 'search_character_info', research)
    app = FastAPI()
    app.state.runtime_container = RuntimeContainer(db=SQLiteDB(tmp_path / 'dialogue-accounting.sqlite'), is_pg_mode=lambda: False, http_client_pool=HttpClientPool(max_connections=1))
    app.include_router(training.router)
    app.dependency_overrides[get_current_admin] = lambda: {'id': 901, 'role': 'admin'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture.invalid') as client:
        response = await client.post('/api/training/generate-dialogues', json=dict(character_description=DESCRIPTION, num_dialogues=2, style='合成谨慎风格'))
    if upstream_failure:
        assert response.status_code == 502
        detail = response.json()['detail']
        assert detail['upstreamStatus'] == 503 and detail['partial_result']['total'] == 1
        output = detail['partial_result']['dialogues']
        assert 'synthetic-private' not in response.text
    else:
        assert response.status_code == 200 and response.json()['success']
        assert response.json()['total'] == 2 and response.json()['cancelled'] is False
        output = response.json()['dialogues']
    assert [row['conversations'] for row in output] == [mapped_messages(batch)[0] for batch in batches[:len(output)]]
    assert [row['provider'] for row in calls] == [selected, selected]
    if selected == 'openai_compat':
        assert snapshots.call_count == 1
        assert all((row['model'] == 'deepseek-flash' and row['temperature'] == 0.2 and (row['auth'] == 'Bearer synthetic-dialogue-original-key') and row['url'].startswith('https://api.deepseek.com/') for row in calls))
    else:
        snapshots.assert_not_called()
    assert reads.call_count == 1 and legacy.call_count == 0
    assert state['generated_count'] == len(output) and (not state['is_generating'])
    assert training._dialogue_generation_task.done() and get_request_context() is None
