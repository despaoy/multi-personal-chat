from dataclasses import replace

import pytest
from test_complete_memory_read import missing_context
from test_memory_response import context, packets

from inference.generation_request import GenerationRequest, generate_character_response
from inference.memory_response import render_absent_memory_response, render_complete_memory_read, render_memory_response


@pytest.mark.parametrize('field', ['episodic_reference_context', 'conversation_reference_context'])
def test_empty_storage_is_not_absence_of_visible_evidence(field):
    ctx = replace(missing_context(), **{field: '原始话语：我来自昆明。'})
    assert render_complete_memory_read('我来自哪里？', ctx, ()) is None
    assert render_absent_memory_response('我来自哪里？', ctx) is None


@pytest.mark.parametrize('field', ['episodic_reference_context', 'conversation_reference_context'])
def test_optional_task_splitter_cannot_drop_unresolved_evidence(field):
    from test_task_execution import request

    from inference.task_execution import prepare_independent_tasks

    original = request()
    ctx = replace(missing_context(), **{field: '我的专业是地质学。'})
    assert prepare_independent_tasks(replace(original, history=(), character_context=ctx)) is None


def test_raw_correction_does_not_get_overridden_by_old_projection():
    original = context(packets('我现在住在厦门。'))
    expected = render_memory_response('我住哪里？', original)
    assert expected is not None and '厦门' in expected
    ctx = replace(original,
                  episodic_reference_context='后来搬到了海口。')
    assert render_memory_response('我住哪里？', ctx) is None
    # No automatic promotion or overwrite of the original stored record.
    assert render_memory_response('我住哪里？', replace(ctx, episodic_reference_context='')) == expected


@pytest.mark.asyncio
async def test_generation_receives_raw_evidence_instead_of_zero_call_unknown():
    ctx = replace(missing_context(), episodic_reference_context='原始话语：我来自昆明。')
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return '这是测试模型返回，不是质量验收。'

    result = await generate_character_response(GenerationRequest(message='我来自哪里？', character_context=ctx), model)
    assert result.model_invoked and len(calls) == 1
    assert '我来自昆明' in calls[0]['messages'][-1]['content']
    assert ctx.memory_status == 'no_match'


def test_saved_memory_status_remains_storage_only_with_unsaved_raw_fact():
    ctx = replace(missing_context(), episodic_reference_context='我来自昆明。')
    answer = render_memory_response('你保存了我的籍贯吗？', ctx)
    assert answer is not None and '没有' in answer
