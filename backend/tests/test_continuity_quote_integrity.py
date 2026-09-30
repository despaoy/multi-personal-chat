import json

from character.conversation_flow import compile_continuity


def user(text):
    return {'role': 'user', 'content': text}


def packet(text):
    return json.loads(text[text.index('{'):]) if text else {}


def test_long_correction_is_preserved_whole_not_just_affirmative_prefix():
    text = '我的意思是，明天准备去北京。' + '这是故事中的情节。' * 24 + '这些都只是虚构角色经历，不是我的计划。'
    result = packet(compile_continuity([user(text)], '你记住的是什么？'))
    assert text in result['最近用户话题原文']
    for values in result.values():
        assert all(value == text for value in values)


def test_oversized_latest_turn_does_not_leave_partial_or_stale_cues():
    text = '我的意思是，明天去面试。' + '背景说明。' * 400 + '前面提到的计划已经取消。'
    result = compile_continuity([user('明天面试'), user(text)], '然后呢？')
    assert result == ''


def test_escaped_text_budget_never_creates_partial_quotes():
    history = [user('先聊别的。'), user('不要建议。' + '\\"' * 220 + '这只是引用，不是请求。')]
    result = compile_continuity(history, '还有呢？')
    if result:
        assert len(result) < 1800
        assert all(value in [m['content'] for m in history] for values in packet(result).values() for value in values)
