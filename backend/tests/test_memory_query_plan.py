"""Field coverage and diagnostic contracts; no model or example-specific answers."""
import pytest

from character.memory_query import plan_memory_query
from character.memory_service import CharacterMemoryService
from character.models import UserScope


@pytest.mark.parametrize(('query', 'fields', 'excluded'), [
    ('你还记得我的专业和工作地点吗？', {'major', 'workplace'}, set()),
    ('我的专业和你的工作地点有什么联系？', {'major'}, {'workplace'}),
    ('我朋友的专业是什么？', set(), {'major'}),
    ('你的个人资料呢？', set(), {'name', 'origin', 'residence', 'major', 'workplace', 'study_stage'}),
    ('我的基本信息有哪些？', {'name', 'origin', 'residence', 'major', 'workplace', 'study_stage'}, set()),
    ('我的饮食禁忌呢？', {'constraints'}, set()),
    ('今天天气怎么样？', set(), set()),
])
def test_field_plan(query, fields, excluded):
    plan = plan_memory_query(query)
    assert set(plan.fields) == fields
    assert set(plan.excluded_fields) == excluded


class Repo:
    def __init__(self, rows=(), error=None):
        self.rows = rows
        self.error = error

    async def list_memory_records(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.rows


def row(key, content, **extra):
    return dict(id=key, memory_key=key, memory_type='user_fact', content=content, **extra)


async def recall(rows, query, **kwargs):
    service = CharacterMemoryService(Repo(rows, **kwargs), semantic_enabled=False)
    return await service.recall_with_diagnostics('role', UserScope('web', 'test', 'u', 'c', 'private'), query)


@pytest.mark.asyncio
async def test_compound_fields_cover_actual_records_not_collateral_evidence():
    rows = [row('user_major', '用户学习天文学'), row('user_workplace', '用户在观测站工作')]
    rows += [row(f'preference_{i}', '用户喜欢工作地点附近的散步路线') for i in range(12)]
    items, _, trace = await recall(rows, '我的专业和工作地点是什么？')
    assert {i.memory_id for i in items} >= {'user_major', 'user_workplace'}
    assert trace['covered_fields'] == ['major', 'workplace']
    assert trace['missing_fields'] == []


@pytest.mark.asyncio
async def test_profile_reports_partial_coverage_without_inventing_missing_slots():
    rows = [row('user_name', '用户叫阿禾'), row('user_origin', '用户来自绍兴'),
            row('user_residence', '用户现住南昌')]
    items, _, trace = await recall(rows, '我的基本信息有哪些？')
    assert {i.memory_id for i in items} == {r['id'] for r in rows}
    assert trace['covered_fields'] == ['name', 'origin', 'residence']
    assert trace['missing_fields'] == ['major', 'workplace', 'study_stage']


@pytest.mark.asyncio
@pytest.mark.parametrize(('rows', 'status'), [
    ([], 'no_records_returned'),
    ([row('user_major', '用户学习天文学', status='retracted')], 'all_filtered'),
    ([row('preference_x', '用户喜欢攀岩')], 'no_relevant_candidates'),
    ([row('user_major', '用户学习天文学')], 'selected'),
])
async def test_distinguishable_retrieval_outcomes(rows, status):
    _, _, trace = await recall(rows, '我的专业是什么？')
    assert trace['status'] == status
    assert trace['elapsed_ms'] >= 0


@pytest.mark.asyncio
async def test_repository_failure_is_not_empty_database():
    items, count, trace = await recall([], '我的专业是什么？', error=RuntimeError('unavailable'))
    assert items == () and count == 0
    assert trace['status'] == 'retrieval_error'
    assert trace['error_type'] == 'RuntimeError'
    assert trace['stage'] == 'read'


@pytest.mark.asyncio
async def test_other_person_fields_are_not_user_fields():
    items, _, trace = await recall([row('user_major', '用户学习天文学')], '我朋友的专业是什么？')
    assert items == ()
    assert trace['status'] == 'all_filtered'


@pytest.mark.asyncio
async def test_semantic_failure_preserves_field_recall_and_reports_fallback(monkeypatch):
    service = CharacterMemoryService(Repo([row('user_major', '用户学习天文学')]), semantic_enabled=True)

    def broken(*args):
        raise RuntimeError('embedding unavailable')

    monkeypatch.setattr(service, '_semantic_similarities', broken)
    items, _, trace = await service.recall_with_diagnostics(
        'role', UserScope('web', 'test', 'u', 'c', 'private'), '我的专业是什么？')
    assert len(items) == 1
    assert trace['status'] == 'selected'
    assert trace['semantic_status'] == 'fallback'


@pytest.mark.asyncio
async def test_cancellation_is_never_swallowed_as_empty_memory():
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        await recall([], '我的专业是什么？', error=asyncio.CancelledError())
