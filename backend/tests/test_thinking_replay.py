import pytest

from evaluation.thinking_replay import final_view, paired_requests


def response(content, reason='stop'):
    return dict(choices=[dict(finish_reason=reason, message=dict(content=content))])


@pytest.mark.parametrize('text,answer', [
    ('<think>内部过程</think>最终回答', '最终回答'),
    ('<THINK>内部\n过程</THINK> 最终回答 ', '最终回答'),
    ('内部过程</think>最终回答', '最终回答'),
    ('正常回答', '正常回答'),
])
def test_reports_only_final_text(text, answer):
    assert final_view(response(text)) == dict(text=answer, status='complete')


@pytest.mark.parametrize('text,reason', [('<think>未完成', 'stop'),
    ('<think>未完成', 'length'), ('半句回答', 'length')])
def test_partial_reasoning_and_truncated_answers_not_counted_as_complete(text, reason):
    assert not final_view(response(text, reason))['text']


def test_mode_comparison_changes_only_template_switch():
    parameters = dict(temperature=0.7, max_tokens=2048, top_p=0.9,
                      repetition_penalty=1.0, frequency_penalty=0, enable_thinking=False)
    trace = dict(model_calls=[dict(parameters=parameters, messages=[dict(role='user', content='问题')])],
                 generation=dict(plan=dict(generation=parameters)))
    variants = paired_requests(trace, 'm', 1)
    off, on = variants['disabled'][0], variants['enabled'][0]
    assert off.pop('chat_template_kwargs') == {'enable_thinking': False}
    assert on.pop('chat_template_kwargs') == {'enable_thinking': True}
    assert on == off
