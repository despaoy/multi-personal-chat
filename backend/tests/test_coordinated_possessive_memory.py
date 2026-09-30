import pytest

from character.memory_extractor import extract_memories


@pytest.mark.parametrize('message', [
    '我叫林岳，专业是气象学。',
    '我住在成都，专业是物理学。',
    '我喜欢红茶，专业为数学。',
])
def test_possessive_field_inherits_explicit_self_owner(message):
    items = {item.memory_key: item for item in extract_memories(message)}
    assert 'user_major' in items
    assert items['user_major'].evidence == message
    reparsed = {item.memory_key: item.content for item in extract_memories(items['user_major'].evidence)}
    assert reparsed['user_major'] == items['user_major'].content


@pytest.mark.parametrize('message', [
    '林岳，专业是气象学。',
    '我叫林岳，他的专业是气象学。',
    '我叫林岳，朋友来了，专业是气象学。',
    '我叫林岳。专业是气象学。',
    '我叫林岳，专业是气象学吗？',
    '假如我叫林岳，专业是气象学。',
])
def test_owner_inheritance_never_crosses_ambiguity_or_nonassertion(message):
    assert 'user_major' not in {item.memory_key for item in extract_memories(message)}
