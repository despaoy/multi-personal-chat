"""Only the new historical-claim audit scope, with explicit denial boundaries."""

import pytest

from evaluation.memory_history_audit import has_never_received_assertion


@pytest.mark.parametrize(('reply', 'asserted'), [
    ('先收到确认，后来撤销；不是从未收到确认。', False),
    ('并非从未收到书面确认。', False),
    ('不能说你从来没有收到书面确认。', False),
    ('不是说你从未收到确认。', False),
    ('你从未收到书面确认。', True),
    ('你一直没有收到确认。', True),
    ('不是从未收到确认；但你从来没有收到确认。', True),
    ('不是说回执无效，你从来没有收到确认。', True),
])
def test_never_received_claim_scope(reply, asserted):
    assert has_never_received_assertion(reply) is asserted
