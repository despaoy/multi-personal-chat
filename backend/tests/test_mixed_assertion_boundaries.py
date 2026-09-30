"""Mixed discourse must preserve independent facts, not retractable prefixes."""

import pytest

from character.memory_extractor import extract_memories


@pytest.mark.parametrize('separator', ['。', '；', ';', '！', '\n'])
@pytest.mark.parametrize('fact,key', [
    ('我的专业是建筑学', 'user_major'),
    ('我现在住在银川', 'user_residence'),
    ('我喜欢陶艺', 'preference_陶艺'),
])
def test_independent_fact_before_locally_attributed_quote(fact, key, separator):
    text = fact + separator + '刚才那些‘我学音乐’的台词是朋友剧本里的，不是我的经历。'
    items = extract_memories(text)
    assert [item.memory_key for item in items] == [key]
    assert items[0].evidence in text
    assert '音乐' not in items[0].content


@pytest.mark.parametrize('suffix', [
    '前面都是剧本里的台词。',
    '上面那句是小说里的，不是我的经历。',
    '刚才说的都是故事里的设定。',
    '这些是剧本里的台词。',
    '这整段都是小说里的内容。',
    '其实是小说里的台词。',
    '这是朋友剧本里的台词。',
    '朋友剧本里的人学音乐。前面说的都不是真的。',
    '朋友剧本里也有这句，以上都不是我的经历。',
    '“我的专业是建筑学”是朋友剧本里的台词，不是我的经历。',
])
def test_backward_attribution_does_not_save_retracted_prefix(suffix):
    assert not extract_memories('我的专业是建筑学。' + suffix)


@pytest.mark.parametrize('text', [
    '小说台词：我的专业是建筑学。我住在银川。',
    '“我的专业是建筑学。我住在银川。”',
    '‘我的专业是建筑学。我住在银川。’',
    '我的专业是建筑学，但这是朋友剧本里的台词。',
    '假设我的专业是建筑学。朋友剧本里的人学音乐。',
    '我的专业是建筑学。不要保存。朋友剧本里的人学音乐。',
])
def test_fiction_scope_and_write_gate_remain_closed(text):
    assert not extract_memories(text)


def test_no_implicit_return_to_user_after_fiction():
    assert not extract_memories('朋友剧本里的人说：我喜欢陶艺。我住在银川。')


def test_two_independent_facts_are_not_joined_or_rewritten():
    text = '我叫叶青。我的专业是建筑学。朋友在剧本里写了“我住在北京”。'
    items = extract_memories(text)
    assert {item.memory_key for item in items} == {'user_name', 'user_major'}
    assert all(item.evidence in text for item in items)


@pytest.mark.parametrize('switch', ['说回我本人，', '说回我自己：', '回到我本人，'])
@pytest.mark.parametrize('fact,key', [('我来自贵阳。', 'user_origin'),
    ('我的专业是建筑学。', 'user_major'), ('我现在住在银川。', 'user_residence')])
def test_explicit_self_return_after_closed_quote(switch, fact, key):
    text = '朋友的小说里有句台词：“我来自大连。”' + switch + fact
    items = extract_memories(text)
    assert [item.memory_key for item in items] == [key]
    assert all(item.evidence in text and '大连' not in item.content for item in items)


@pytest.mark.parametrize('text', [
    '小说台词：“说回我本人，我来自贵阳。”',
    '小说台词：“我来自大连。说回我本人，我来自贵阳。',
    '小说台词：“我来自大连。”如果说回我本人，我来自贵阳。',
    '小说台词：“我来自大连。”说回我本人，我来自贵阳。以上都是剧本里的内容。',
    '小说台词：“我来自大连。”说回我本人，我来自贵阳。不要保存。',
    '小说台词：“我来自大连。”说回我本人，我可能来自贵阳。',
])
def test_return_marker_does_not_override_quote_condition_or_retraction(text):
    assert not extract_memories(text)


@pytest.mark.parametrize('opening,closing', [('“', '”'), ('‘', '’'), ('「', '」'), ('『', '』'), ('"', '"')])
def test_quote_style_does_not_change_reset_semantics(opening, closing):
    text = '小说台词：' + opening + '我叫小白。' + closing + '说回我自己，我叫林青。'
    items = extract_memories(text)
    assert len(items) == 1 and items[0].content == '用户说自己叫林青'


def test_nested_and_mismatched_quotes_cannot_create_self_return():
    assert not extract_memories('小说台词：“他说‘我来自大连。’说回我本人，我来自贵阳。”')
    assert not extract_memories('小说台词：“我来自大连。』说回我本人，我来自贵阳。')
