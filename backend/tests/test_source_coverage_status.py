import pytest

from character.models import CompiledCharacterContext
from character.source_memory import SourceRecall, attach_sources, compile_sources
from inference.memory_response import read_memory_fields, render_memory_response


@pytest.mark.parametrize('status', ['budget_omitted', 'retrieval_error'])
def test_unread_sources_cannot_prove_a_missing_personal_field(status):
    base = CompiledCharacterContext('', '', '', memory_status='no_match',
                                    memory_field_presence=(('major', False), ('name', True)))
    context = attach_sources(base, SourceRecall(diagnostics={'status': status}))
    assert dict(context.memory_field_presence) == {'major': None, 'name': True}
    assert read_memory_fields('我的专业是什么？', context)[0].status == 'unverified'
    reply = render_memory_response('你保存了我的专业吗？', context)
    assert '没有你的专业记录' not in reply
    assert '原话' in reply and '不能确认' in reply
    assert '有你的姓名记录' in render_memory_response('你保存了我的姓名吗？', context)


def test_real_source_packet_budget_failure_propagates_to_storage_response():
    result = compile_sources([{'source_message_id': 's', 'observed_at': '2026-09-27T00:00:00Z',
                               'body': '我的专业是档案学。' + '完整内容' * 700}])
    assert result.diagnostics['status'] == 'budget_omitted' and not result.context
    context = attach_sources(CompiledCharacterContext('', '', '', memory_status='no_match',
        memory_field_presence=(('major', False),)), result)
    assert dict(context.memory_field_presence)['major'] is None
    assert not context.memory_packets
    assert '不能确认' in render_memory_response('你保存了我的专业吗？', context)


def test_empty_refresh_removes_old_source_packet_without_inventing_absence():
    first = attach_sources(CompiledCharacterContext('', '', '', memory_field_presence=(('major', False),)),
                           SourceRecall('旧原话', {'status': 'available'}))
    refreshed = attach_sources(first, SourceRecall(diagnostics={'status': 'retrieval_error'}))
    assert refreshed.episodic_reference_context == ''
    assert dict(refreshed.memory_field_presence)['major'] is None


def test_no_match_does_not_change_a_fresh_typed_absence():
    context = attach_sources(CompiledCharacterContext('', '', '', memory_field_presence=(('major', False),)),
                             SourceRecall(diagnostics={'status': 'no_match'}))
    assert dict(context.memory_field_presence)['major'] is False
