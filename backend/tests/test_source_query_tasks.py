import pytest

from knowledge.query_tasks import requests_only_source_excerpt, requests_source_text


@pytest.mark.parametrize('query', [
    '我不想知道天文学', '不是我想知道天文学',
    '她说我想了解天文学', '请翻译“我想了解天文学”',
    '他说：“今天有空。我想了解天文学。”',
    '不要再告诉我这件事。',
    '她说再告诉我天气。',
    '请翻译“接着告诉我实验步骤”。',
])
def test_shared_information_act_excludes_denial_and_reported_mentions(query):
    from knowledge.query_tasks import requests_explicit_information

    assert not requests_explicit_information(query)


def test_quoted_request_does_not_hide_a_real_later_request():
    from knowledge.query_tasks import requests_explicit_information

    assert requests_explicit_information('他说“我想了解天文学”，我想知道气象学的起源')


@pytest.mark.parametrize('query', [
    '不要分析，但请查找原文。', '不用解释，请给出原文依据。',
    '请展示原句，不需要总结。', '无需评价；请提供相关段落。',
    '请不要分析，请列出原文。', '请查找原文。',
])
def test_negative_interpretation_control_does_not_create_an_extra_task(query):
    assert requests_source_text(query)
    assert requests_only_source_excerpt(query)


@pytest.mark.parametrize('query', [
    '不要查找原文。', '不用给出原文。', '请不要引用原文。',
    '不要分析，但请翻译原文。', '不要分析，但请解释原文背景。',
    '不用建议，先回答我的专业是什么，然后给出原文。',
    '我今天读完了原文，不用分析。', '请给出原文，并比较两个版本。',
    '不要分析这个人的动机，请给出原文。',
])
def test_source_shortcut_never_drops_other_tasks_or_source_negation(query):
    assert not requests_only_source_excerpt(query)


@pytest.mark.parametrize('query', [
    '可以提供相关的原书段落吗？', '依据？', '能找出原句吗？',
    '请列出支持这个结论的证据。', '这些判断出自哪些章节？',
])
def test_source_requests(query):
    assert requests_source_text(query)


@pytest.mark.parametrize('query', [
    '我今天读完了另一本原著。', '我在整理章节目录。',
    '今天谈谈你的看法。', '我喜欢读原作。', '',
])
def test_source_mentions_do_not_become_requests(query):
    assert not requests_source_text(query)
