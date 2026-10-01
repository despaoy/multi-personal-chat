"""New positive/negative controls for literal reported booking status."""

import pytest

from evaluation.memory_friend_audit import reported_confirmation_and_validity


@pytest.mark.parametrize(('reply', 'expected'), [
    ('书面确认：已收到北岸榆影工坊周日纸雕课的书面预约确认。记录中的效力：这份预约在转述时仍然有效。', True),
    ('按你的转述，已收到书面确认。预约仍有效。', True),
    ('已收到书面预约确认。尚未参加过课程，没有出发。预约仍然有效。', True),
    ('尚未收到北岸榆影工坊周日纸雕课的书面预约确认。预约仍然有效。', False),
    ('季澄未收到书面预约确认。预约仍然有效。', False),
    ('无法判断他是否收到书面确认，也无法确认预约仍然有效。', False),
    ('收到书面预约确认。不能确认这份预约仍有效。', False),
    ('是否收到书面预约确认、预约仍有效？', False),
    ('已收到书面预约确认。预约不再有效。', False),
    ('已收到' + '甲' * 100 + '确认。预约仍有效。', False),
])
def test_reported_status_clauses(reply, expected):
    assert reported_confirmation_and_validity(reply) is expected
