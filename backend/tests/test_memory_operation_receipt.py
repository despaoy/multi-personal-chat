import asyncio
import json
from types import SimpleNamespace

import pytest

from character.memory_llm import (
    MemoryEnrichmentScheduler,
    MemoryLlmConfig,
    classify_memory_write_mode,
    is_memory_erasure_request,
    parse_llm_proposals,
)
from character.models import UserScope
from db.memory_claim_guard import MemoryClaimConflict
from db.memory_source import ClaimSourceRevokedError


@pytest.mark.parametrize('source', [
    '请把我那只猫的名字从记忆里删除，也不要再引用那条原话。',
    '把我的住处从长期记忆里面清除。',
    '请将那条记录从记忆中彻底删掉。',
    '忘掉我的工作地点。',
    '请把你记住的住址彻底删掉。',
])
def test_natural_erasure_routes_to_same_hot_path_and_admission(source):
    assert is_memory_erasure_request(source)
    assert classify_memory_write_mode(source) == 'hot'
    assert len(parse_llm_proposals(response(source), source_message=source, existing_memories=(record(),))) == 1


@pytest.mark.parametrize('source', [
    '不要忘掉我的住址。', '别从记忆里删除我的猫名。',
    '如果要从记忆里删除那条记录，该怎么做？',
    '他说请忘掉我的住址。', '“请忘掉我的住址”是示例命令。',
    '我不希望你把这条记忆删除。',
])
def test_non_authorizing_text_cannot_erase_even_with_model_erase_proposal(source):
    assert not is_memory_erasure_request(source)
    assert not parse_llm_proposals(response(source), source_message=source, existing_memories=(record(),))


def record():
    return dict(id=7, memory_key='fact_pet', memory_type='user_fact', content='用户的猫叫米粒', status='active')


def response(source):
    return json.dumps({'memories': [dict(kind='other_user_fact', value='', content='',
        evidence=source, operation='ERASE', target_memory_id='7', target_memory_key='fact_pet', confidence=.99)]})


class Repo:
    def __init__(self, *, fail=False):
        self.deleted = []
        self.fail = fail

    async def list_memory_records(self, *args, **kwargs):
        return [record()]

    async def erase_memory(self, character, scope, **kwargs):
        if self.fail:
            raise RuntimeError('storage unavailable')
        self.deleted.append(scope.sender_id)
        return 1


class Completion:
    def __init__(self, gate=None):
        self.gate = gate

    async def complete(self, messages):
        if self.gate:
            await self.gate.wait()
        payload = json.loads(messages[-1]['content'])
        return response(payload['current_user_message'])

    async def close(self):
        pass


def scheduler(gate=None):
    return MemoryEnrichmentScheduler(config=MemoryLlmConfig(enabled=True, base_url='http://unused', model='stub'),
        completion=Completion(gate), embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1., 0.] for _ in texts]))


def job(repo, user):
    return dict(repository=repo, character_id='role', user_scope=UserScope('web', 'test', user, user, 'private'),
        message='请把那条资料从记忆里删除。', rule_hints=[], source_message_id='source-' + user)


async def test_receipts_follow_exact_jobs_not_global_last_status():
    worker = scheduler()
    good, bad = Repo(), Repo(fail=True)
    try:
        first, second = await asyncio.gather(worker.schedule_and_wait(**job(good, 'a')),
                                             worker.schedule_and_wait(**job(bad, 'b')))
        assert first['status'] == 'erased' and first['persisted'] == 1
        assert second['status'] == 'failed' and second['persisted'] == 0
        assert first['source_message_id'] == 'source-a'
        assert second['source_message_id'] == 'source-b'
        assert good.deleted == ['a'] and bad.deleted == []
    finally:
        await worker.shutdown(timeout=1)


async def test_raw_source_search_failure_does_not_disable_claim_erasure():
    class SourceFailureRepo(Repo):
        async def search_sources(self, *args, **kwargs):
            raise RuntimeError('source search unavailable')

        async def list_sources(self, *args, **kwargs):
            return []

    worker, repo = scheduler(), SourceFailureRepo()
    try:
        result = await worker.schedule_and_wait(**job(repo, 'a'))
        assert result['status'] == 'erased'
        assert result['source_candidate_coverage']['status'] == 'retrieval_error'
        assert repo.deleted == ['a']
    finally:
        await worker.shutdown(timeout=1)


async def test_timeout_is_pending_and_does_not_cancel_accepted_work():
    gate = asyncio.Event()
    worker, repo = scheduler(gate), Repo()
    try:
        result = await worker.schedule_and_wait(**job(repo, 'a'), timeout_seconds=.01)
        assert result['status'] == 'pending' and result['persisted'] == 0
        assert repo.deleted == []
        gate.set()
        assert await worker.flush_memory(timeout=1)
        assert repo.deleted == ['a']
    finally:
        gate.set()
        await worker.shutdown(timeout=1)


async def test_closed_scheduler_does_not_return_old_success():
    worker = scheduler()
    await worker.shutdown(timeout=1)
    result = await worker.schedule_and_wait(**job(Repo(), 'a'))
    assert result['status'] == 'not_scheduled' and result['persisted'] == 0


@pytest.mark.parametrize('outcomes,expected', [
    (['saved', ClaimSourceRevokedError('revoked')], 'partial'),
    (['erased', MemoryClaimConflict('changed')], 'partial'),
    ([MemoryClaimConflict('changed')], 'conflict'),
    ([ClaimSourceRevokedError('revoked')], 'skipped'),
    (['saved', 'no_change'], 'saved'),
])
async def test_receipt_does_not_hide_partial_or_conflicted_operations(monkeypatch, outcomes, expected):
    from character.memory_llm import ValidatedMemoryProposal

    worker = scheduler()
    monkeypatch.setattr('character.memory_llm.parse_llm_proposals',
                        lambda *a, **kw: [ValidatedMemoryProposal(operation='ADD') for _ in outcomes])
    pending = iter(outcomes)

    async def persist(*args):
        value = next(pending)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(worker, '_persist_proposal', persist)
    try:
        result = await worker.schedule_and_wait(**job(Repo(), 'a'))
        assert result['status'] == expected
        assert result['persisted'] == sum(value in ('saved', 'erased') for value in outcomes)
        assert result['conflicts'] == sum(isinstance(value, MemoryClaimConflict) for value in outcomes)
        assert len(result['operation_outcomes']) == len(outcomes)
        assert worker.status.failed == 0
    finally:
        await worker.shutdown(timeout=1)


async def test_shutdown_resolves_active_and_queued_receipts():
    gate = asyncio.Event()
    worker, repo = scheduler(gate), Repo()
    first = asyncio.create_task(worker.schedule_and_wait(**job(repo, 'a')))
    second = asyncio.create_task(worker.schedule_and_wait(**job(repo, 'b')))
    await asyncio.sleep(0)
    await worker.shutdown(timeout=.01)
    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert all(result['status'] == 'cancelled' for result in results)
    assert worker.status.queued == worker.status.processing == 0
    assert repo.deleted == []


async def test_cancellation_preserves_already_confirmed_commits(monkeypatch):
    from character.memory_llm import ValidatedMemoryProposal

    worker = scheduler()
    monkeypatch.setattr('character.memory_llm.parse_llm_proposals',
                        lambda *a, **kw: [ValidatedMemoryProposal(operation='ADD')] * 2)
    second_started = asyncio.Event()
    calls = 0

    async def persist(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 'saved'
        second_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, '_persist_proposal', persist)
    task = asyncio.create_task(worker.schedule_and_wait(**job(Repo(), 'a')))
    try:
        await asyncio.wait_for(second_started.wait(), timeout=1)
        await worker.shutdown(timeout=.01)
        result = await task
        assert result['status'] == 'cancelled'
        assert result['persisted'] == 1
        assert result['operation_outcomes'] == ('saved',)
        assert worker.status.last_outcome == 'cancelled'
    finally:
        await worker.shutdown(timeout=.01)


async def test_actual_sqlite_receipt_requires_claim_and_source_removal(tmp_path):
    from datetime import datetime, timezone

    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    database = SQLiteDB(tmp_path / 'erase.sqlite')
    repo = DatabaseCharacterMemoryRepository(database)
    scope = dict(character_id='role', platform='web', adapter='test', sender_id='a',
                 conversation_type='private', conversation_id='a')
    database.capture_memory_source(**scope, source_message_id='original', body='我的猫叫米粒',
                                   observed_at=datetime.now(timezone.utc))
    saved = database.append_character_memory_claim(**scope, memory_type='user_fact', memory_key='fact_pet',
        content='用户的猫叫米粒', source_message_id='original')

    class RealTargetCompletion(Completion):
        async def complete(self, messages):
            raw = json.loads(await super().complete(messages))
            raw['memories'][0]['target_memory_id'] = str(saved['id'])
            return json.dumps(raw)

    worker = scheduler()
    worker._completion = RealTargetCompletion()
    try:
        result = await worker.schedule_and_wait(**job(repo, 'a'))
        assert result['status'] == 'erased' and result['persisted'] == 1
        user_scope = job(repo, 'a')['user_scope']
        assert await repo.list_memory_records('role', user_scope, include_inactive=True) == []
        assert await repo.list_sources('role', user_scope) == []
        assert result['source_capture'] == 'erase_request'
    finally:
        await worker.shutdown(timeout=1)


async def test_real_database_conflict_does_not_replay_stale_model_proposal(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from character.memory_extractor import extract_memories
    from character.memory_llm import ValidatedMemoryProposal
    from character.models import MemoryItem
    from db.database import SQLiteDB
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    repo = DatabaseCharacterMemoryRepository(SQLiteDB(tmp_path / 'conflict.sqlite'))
    scope = job(repo, 'a')['user_scope']
    original = await repo.append_claim('role', scope, MemoryItem('', 'user_fact', '用户住在南宁', .8),
                                       memory_key='user_location', observed_at='2025-01-01T00:00:00+00:00')
    item = extract_memories('我住在桂林。')[0]
    proposal = ValidatedMemoryProposal(operation='SUPERSEDE', memory=item,
        target_memory_id=str(original['id']), evidence=item.evidence, confidence=.99)
    monkeypatch.setattr('character.memory_llm.parse_llm_proposals', lambda *a, **kw: [proposal])

    class InterleavedCompletion(Completion):
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            await repo.append_claim('role', scope, MemoryItem('', 'user_fact', '用户住在洛阳', .8),
                memory_key='user_location', relation_type='SUPERSEDE', supersedes_memory_id=original['id'],
                observed_at=datetime.now(timezone.utc).isoformat())
            return '{}'

    worker = scheduler()
    worker._completion = InterleavedCompletion()
    try:
        result = await worker.schedule_and_wait(**{**job(repo, 'a'), 'message': item.evidence})
        assert result['status'] == 'conflict' and result['persisted'] == 0
        assert result['operation_outcomes'] == ('conflict',)
        assert worker._completion.calls == 1  # no hidden re-inference/replay
        active = await repo.list_memory_records('role', scope)
        assert [row['content'] for row in active] == ['用户住在洛阳']
        assert len(await repo.list_memory_records('role', scope, include_inactive=True)) == 2
        assert len(await repo.list_sources('role', scope)) == 1  # original speech retained
    finally:
        await worker.shutdown(timeout=1)
