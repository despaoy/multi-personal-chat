"""Offline selector contracts; these do not assert answer correctness."""
from dataclasses import replace

import pytest

from evaluation.episodic_recall_audit import Episode, packet_text, select_evidence

SCOPE = ('web', 'audit', 'alice', 'kisaki', 'private')


def episode(key, text, *, session='s1', reply='收到。'):
    return Episode(key, SCOPE, session, '2026-09-26T00:00:' + key.zfill(2), text, reply)


def test_full_session_recovers_nonlexical_correction():
    rows = [episode('1', '我周六要去参观美术馆。'), episode('2', '改了，那天待在家里修书架。')]
    turn = select_evidence(rows, '周六做什么？', SCOPE, mode='turn', max_chars=2000)
    session = select_evidence(rows, '周六做什么？', SCOPE, mode='session', max_chars=2000)
    assert turn['source_ids'] == ['1']
    assert session['source_ids'] == ['1', '2']
    assert all(row.message in session['packet'] for row in rows)


def test_cross_session_correction_remains_an_explicit_coverage_gap():
    rows = [episode('1', '我周六要去参观美术馆。'), episode('2', '改了，那天待在家里修书架。', session='s2')]
    result = select_evidence(rows, '周六做什么？', SCOPE, mode='session', max_chars=2000)
    assert result['source_ids'] == ['1']  # Diagnostic gap, NOT safe current-state evidence.


@pytest.mark.parametrize('field', range(len(SCOPE)))
def test_scope_filter_happens_before_grouping(field):
    wrong = list(SCOPE)
    wrong[field] = 'other'
    row = episode('1', '周六修书架。')
    other = replace(episode('2', '周六改去露营。'), scope=tuple(wrong))
    assert select_evidence([row, other], '周六', SCOPE, mode='session', max_chars=2000)['source_ids'] == ['1']


def test_assistant_text_cannot_create_recall_hit():
    row = episode('1', '你好。', reply='你周六要去露营。')
    assert not select_evidence([row], '周六', SCOPE, mode='session', max_chars=2000)['source_ids']


def test_additional_hits_keep_later_unmatched_correction_and_scope():
    rows = [episode('1', '我现在住在厦门。'), episode('2', '搬家了。', session='s2')]
    result = select_evidence(rows, '我住哪里？', SCOPE, mode='suffix', max_chars=2000, additional_hit_ids=('1',))
    assert result['source_ids'] == ['1', '2']
    with pytest.raises(ValueError):
        select_evidence(rows, '我住哪里？', SCOPE, mode='suffix', max_chars=2000, additional_hit_ids=('other-user',))
    blocked = [*rows, episode('3', '不要保存修改。', session='s3')]
    assert not select_evidence(blocked, '我住哪里？', SCOPE, mode='suffix', max_chars=2000,
                               additional_hit_ids=('1',))['source_ids']


@pytest.mark.parametrize('blocked', ['不要保存这件事。', '密码是测试占位符。'])
def test_filtered_correction_cannot_leave_old_plan_in_its_session(blocked):
    rows = [episode('1', '周六修书架。'), episode('2', blocked)]
    result = select_evidence(rows, '周六', SCOPE, mode='session', max_chars=2000)
    assert result['blocked_sessions'] == ['s1'] and not result['source_ids']


def test_whole_packet_budget_counts_serialized_metadata_and_escaping():
    rows = [episode('1', '周六“修书架”。'), episode('2', '改了。\n那天整理照片。')]
    size = len(packet_text(rows))
    assert not select_evidence(rows, '周六', SCOPE, mode='session', max_chars=size-1)['source_ids']
    assert select_evidence(rows, '周六', SCOPE, mode='session', max_chars=size)['source_ids'] == ['1', '2']


def test_ambiguous_source_ids_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        select_evidence([episode('1', '周六修书架。')]*2, '周六', SCOPE, mode='turn', max_chars=2000)


def test_suffix_keeps_cross_session_updates_without_asserting_a_link():
    rows = [episode('1', '周六修书架。'), episode('2', '我今天读完一本书。', session='s2'),
            episode('3', '之前那个安排取消了。', session='s3')]
    result = select_evidence(rows, '周六', SCOPE, mode='suffix', max_chars=2000)
    assert result['source_ids'] == ['1', '2', '3']
    assert result['packet'] == packet_text(rows)
    # Budget pressure must not silently convert the window into the old plan.
    small = select_evidence(rows, '周六', SCOPE, mode='suffix', max_chars=len(packet_text(rows))-1)
    assert small['source_ids'] == [] and small['budget_rejected'] == ['complete_suffix']


def test_suffix_does_not_cross_a_filtered_update():
    rows = [episode('1', '周六修书架。'), episode('2', '不要保存改动。', session='s2')]
    assert not select_evidence(rows, '周六', SCOPE, mode='suffix', max_chars=2000)['source_ids']
