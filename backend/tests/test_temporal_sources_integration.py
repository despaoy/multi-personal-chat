"""Real Temporal execution with controlled indexing and authoritative source storage."""

import asyncio
import os
from pathlib import Path

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from character.library import CharacterSettings
from db.database import SQLiteDB
from repositories.character_sources import CharacterSourceRepository
from services import temporal_sources
from workflows.source_indexing import SourceIndexWorkflow


@pytest.mark.skipif(os.getenv('RUN_TEMPORAL_INTEGRATION') != '1', reason='requires local Temporal server')
def test_source_workflow_failure_retry_restart_and_history_privacy(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / 'sources.db')
    repo = CharacterSourceRepository(db)
    char = repo.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
    corpus = repo.publish('owner', char['id'], 1, [{'id': 'chapter', 'title': 'PRIVATE-TITLE', 'content': 'PRIVATE-ORIGINAL'}])
    calls = []

    async def controlled(owner, corpus_id, *, database, expected_attempt):
        calls.append(expected_attempt)
        if expected_attempt == 1:
            raise RuntimeError('PRIVATE-PROVIDER-EXCEPTION')
        repo.mark_indexed(owner, corpus_id, {'corpus_digest': corpus['content_digest'], 'index_name': 'test-index',
                                           'embedding_identity': 'test-model', 'chunk_count': 1}, expected_attempt=expected_attempt)

    monkeypatch.setattr(temporal_sources, 'publish_configured_source_index', controlled)

    async def exercise():
        cache = Path(__file__).resolve().parents[2] / '.uv-cache' / 'temporal'
        cache.mkdir(parents=True, exist_ok=True)
        async with await WorkflowEnvironment.start_local(download_dest_dir=str(cache), ui=False) as env:
            queue = 'source-integration'
            # Dispatch without an active worker; the source remains durably submitted.
            assert await temporal_sources.dispatch_sources(db, env.client, queue) == 1
            # Simulate a lost database acknowledgement after Temporal accepted the workflow.
            db.execute_sql("UPDATE character_source_corpora SET index_status='pending' WHERE id=:id", {'id': corpus['id']})
            assert await temporal_sources.dispatch_sources(db, env.client, queue) == 1
            assert await temporal_sources.dispatch_sources(db, env.client, queue) == 0
            activities = temporal_sources.SourceIndexActivities(db)
            async with Worker(env.client, task_queue=queue, workflows=[SourceIndexWorkflow],
                              activities=[activities.index, activities.fail]):
                handle = env.client.get_workflow_handle(f"character-source-index-{corpus['id']}-1")
                assert await asyncio.wait_for(handle.result(), 90) == 'failed'
                assert repo.get('owner', corpus['id'])['index_status'] == 'failed'
                assert 'PRIVATE-' not in (await handle.fetch_history()).to_json()
            repo.retry_index_job('owner', corpus['id'], 1)
            assert await temporal_sources.dispatch_sources(db, env.client, queue) == 1
            # A replacement worker consumes the retry without repeating the old attempt.
            restarted = temporal_sources.SourceIndexActivities(db)
            async with Worker(env.client, task_queue=queue, workflows=[SourceIndexWorkflow],
                              activities=[restarted.index, restarted.fail]):
                handle = env.client.get_workflow_handle(f"character-source-index-{corpus['id']}-2")
                assert await asyncio.wait_for(handle.result(), 30) == 'completed'
                assert 'PRIVATE-' not in (await handle.fetch_history()).to_json()
                assert repo.require_indexed('owner', corpus['id'], 'test-index', 'test-model')['index_attempt'] == 2
                assert await restarted.index({'owner': 'owner', 'corpus_id': corpus['id'], 'attempt': 1}) == 'obsolete'
            assert calls == [1, 1, 1, 2]
    try:
        asyncio.run(exercise())
    finally:
        db.close_connection()
