from dataclasses import replace

import pytest
from test_memory_response import context, packets
from test_task_execution import request

from inference.generation_request import generate_character_response
from inference.memory_response import read_memory_fields, render_absent_memory_response
from inference.task_execution import prepare_independent_tasks


@pytest.mark.parametrize('state,expected', [
    ('absent', 'absent'), ('error', 'unavailable'), ('unchecked', 'unavailable'),
    ('unknown', 'unknown'), ('clipped', 'unadmitted'), ('conflict', 'conflicting'),
    ('invalid', 'unverified'), ('known', 'known'),
])
def test_saved_memory_states_are_distinct(state, expected):
    compiled = context(packets('我的专业是化学。'))
    if state in {'absent', 'unknown', 'clipped', 'unchecked'}:
        compiled = replace(compiled, memory_packets=(), used_memory_ids=(), memory_status='no_match',
                           memory_field_presence=() if state == 'unknown' else (('major', state == 'clipped'),))
        if state == 'unchecked':
            compiled = replace(compiled, memory_status='not_checked')
    elif state == 'error':
        compiled = replace(compiled, memory_status='retrieval_error')
    elif state == 'invalid':
        compiled = replace(compiled, memory_packets=(replace(compiled.memory_packets[0], evidence=()),))
    elif state == 'conflict':
        other = replace(packets('我的专业是数学。')[0], memory_id='other')
        compiled = replace(compiled, memory_packets=(*compiled.memory_packets, other),
                           used_memory_ids=(*compiled.used_memory_ids, 'other'))
    result, = read_memory_fields('我的专业是什么？', compiled)
    assert result.status == expected
    assert (render_absent_memory_response('我的专业是什么？', compiled) is not None) is (state == 'absent')
    assert (result.value is not None) is (state == 'known')


@pytest.mark.asyncio
async def test_proven_absence_keeps_user_ownership_and_identity_task():
    original = request()
    original = replace(original, history=(), character_context=replace(
        original.character_context, memory_status='no_match', memory_packets=(), used_memory_ids=(),
        memory_field_presence=(('major', False),)))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        assert '我的专业' not in kwargs['messages'][-1]['content']
        return '林远是学生。'

    result = await generate_character_response(original, model)
    assert len(calls) == 1
    assert result.reply == '我暂时没有找到你的专业记录。\n\n林远是学生。'
    assert result.task_results[0]['mode'] == 'memory_not_found'
    assert result.task_results[1]['mode'] == 'generated'
    for history in [({'role': 'user', 'content': '我的专业是生物学。'},),
                    ({'role': 'assistant', 'content': '你刚说你的专业是生物学。'},)]:
        assert prepare_independent_tasks(replace(original, history=history)) is None


def test_absence_of_one_field_cannot_hide_an_unresolved_field():
    compiled = replace(context(()), memory_field_presence=(('major', False),))
    assert render_absent_memory_response('我的专业和名字是什么？', compiled) is None
    assert read_memory_fields('你的专业是什么？', compiled) == ()


@pytest.mark.parametrize('state,presence,expected', [
    ('no_match', False, '未找到当前可用记录'),
    ('no_match', True, '存在记录但未进入本轮记忆证据'),
    ('retrieval_error', False, '读取失败'),
])
def test_query_result_is_data_and_preserves_full_task_history_and_system(state, presence, expected):
    from inference.generation_request import build_generation_request

    original = request()
    original = replace(original, character_context=replace(
        original.character_context, memory_packets=(), used_memory_ids=(), reference_context='',
        memory_status=state, memory_field_presence=(('major', presence),)))
    plan = build_generation_request(original)
    control = build_generation_request(replace(original, character_context=replace(
        original.character_context, memory_field_presence=())))
    assert plan.messages[0] == control.messages[0]
    assert original.message in plan.messages[-1]['content']
    assert '旧回复' in str(plan.messages)
    assert expected in plan.messages[-1]['content']
    assert '不包含当前消息及聊天历史' in plan.messages[-1]['content']
    assert '当前对话者（用户）' in plan.messages[-1]['content']
    assert all('<memory_query_result' not in message['content'] for message in plan.messages[:-1])


@pytest.mark.parametrize('message', ['我朋友的专业是什么？', '请翻译“我的专业是什么？”',
                                     '我的专业是数学。', '你好'])
def test_query_result_does_not_invent_queries_from_other_owners_or_data(message):
    from inference.memory_response import memory_query_result

    compiled = replace(context(()), memory_field_presence=(('major', False),))
    assert memory_query_result(message, compiled) == ''


def test_diagnostic_state_is_never_serialized_as_a_field_value():
    import json

    from inference.memory_response import memory_query_result

    compiled = replace(context(()), memory_field_presence=(('major', False),))
    result = json.loads(memory_query_result('我的专业是什么？', compiled))
    field, = result['字段结果']
    assert field['字段'] == '专业' and field['查询状态'] == 'absent'
    assert field['可用值'] is None


@pytest.mark.parametrize('conflict', [False, True])
def test_unverified_and_conflicting_projection_does_not_publish_candidate_values(conflict):
    import json

    from inference.memory_response import memory_query_result

    compiled = context(packets('我的专业是化学。'))
    if conflict:
        other = replace(packets('我的专业是数学。')[0], memory_id='other')
        compiled = replace(compiled, memory_packets=(*compiled.memory_packets, other),
                           used_memory_ids=(*compiled.used_memory_ids, 'other'))
    else:
        compiled = replace(compiled, memory_packets=(replace(compiled.memory_packets[0], evidence=()),))
    compiled = replace(compiled, memory_status='matched')
    data = memory_query_result('我的专业是什么？', compiled)
    field, = json.loads(data)['字段结果']
    assert field['查询状态'] == ('conflicting' if conflict else 'unverified')
    assert field['可用值'] is None
    assert '化学' not in data and '数学' not in data


def test_partial_read_projects_only_unresolved_field_without_removing_known_packet():
    import json

    from inference.memory_response import memory_query_result

    compiled = replace(context(packets('我叫许澄。')), memory_status='matched',
                       memory_field_presence=(('name', True), ('major', False)))
    reads = read_memory_fields('我的名字和专业分别是什么？', compiled)
    assert [(item.field, item.status) for item in reads] == [('name', 'known'), ('major', 'absent')]
    field, = json.loads(memory_query_result('我的名字和专业分别是什么？', compiled))['字段结果']
    assert field['字段'] == '专业' and field['可用值'] is None
    assert compiled.memory_packets[0].memory_id in compiled.used_memory_ids
