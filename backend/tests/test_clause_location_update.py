"""Clause-scoped evidence must survive semantic write field resolution."""
import json

import pytest

from character.memory_extractor import extract_memories
from character.memory_llm import parse_llm_proposals


@pytest.mark.parametrize('source,value', [
    ('我已经搬到绵阳了，绍兴是之前住的地方。', '绵阳'),
    ('我刚搬到贵阳了，我的专业是历史学。', '贵阳'),
    ('我叫林晓，我已经搬家到珠海了。', '珠海'),
    ('我搬到银川了，今天去买家具。', '银川'),
])
def test_complete_self_relocation_in_a_clause_updates_residence(source, value):
    items = extract_memories(source)
    residence, = [item for item in items if item.memory_key == 'user_residence']
    assert residence.content == '用户说自己居住在' + value
    assert residence.evidence in source
    raw = dict(kind='location', value=value, content='用户已经搬到' + value + '了', evidence=source,
        confidence=.98, operation='SUPERSEDE', target_memory_id='7', target_memory_key='user_residence')
    proposals = parse_llm_proposals(json.dumps({'memories': [raw]}, ensure_ascii=False), source_message=source,
        existing_memories=({'id': 7, 'memory_key': 'user_residence', 'content': '用户说自己居住在原住处',
                            'status': 'active'},))
    assert len(proposals) == 1
    assert proposals[0].memory.memory_key == 'user_residence'
    assert proposals[0].operation == 'SUPERSEDE'


@pytest.mark.parametrize('source', [
    '我打算搬到贵阳了，我的专业是历史学。',
    '我姐姐已经搬到贵阳了，我的专业是历史学。',
    '假设我已经搬到贵阳了，我要怎么办？',
    '我已经搬到贵阳了，但只是暂时出差。',
    '我已经搬到贵阳了，这句话是假的。',
    '我已经搬到贵阳了吗，我记不清了。',
    '我已经搬到贵阳了，我已经搬到珠海了。',
])
def test_clause_recovery_does_not_promote_nonfacts_or_choose_conflicting_destinations(source):
    assert not [item for item in extract_memories(source)
                if item.memory_key == 'user_residence' and not item.qualifiers]
