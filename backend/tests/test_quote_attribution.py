import pytest

from evaluation.quote_attribution import audit_quote_attribution


def doc(raw, **source):
    return {'id': 'a', 'content': '摘要不得作为依据\n证据：' + raw,
            'source': {'source_path': 'story.txt', 'line_start': 10,
                       'line_end': 10 + len(raw.splitlines()) - 1, **source}}


@pytest.mark.parametrize('reply,status', [
    ('甲说：“今天下雨。”', 'supported_label'),
    ('乙说：“今天下雨。”', 'conflicting_label'),
    ('甲提到：“我们还要继续选择。”', 'narration_only'),
    ('甲说：“明天晴天。”', 'unmatched_literal'),
    ('丙说：“今天下雨。”', 'unknown_speaker'),
])
def test_literal_quote_provenance(reply, status):
    raw = '[甲] 「今天下雨。」\n我们还要继续选择。\n[乙] 「我带了伞。」'
    finding, = audit_quote_attribution(reply, [doc(raw)])
    assert finding['status'] == status
    if finding['evidence']:
        assert finding['evidence'][0]['line'] in (10, 11)


@pytest.mark.parametrize('prefix', ['而人物甲也', '人物甲则', '人物甲曾经', '人物甲也曾'])
def test_explicit_alias_and_grammatical_prefix(prefix):
    finding, = audit_quote_attribution(prefix + '回应：“我带了伞。”',
        [doc('[甲] 「我带了伞。」')], aliases={'人物甲': '甲'})
    assert finding['status'] == 'supported_label'


def test_narration_does_not_override_an_explicit_matching_speaker():
    finding, = audit_quote_attribution('甲说：“好。”', [doc('好。\n[甲] 「好。」\n[乙] 「好。」')])
    assert finding['status'] == 'supported_label'
    assert finding['labeled_speakers'] == ['乙', '甲']
    assert len(finding['evidence']) == 3


def test_mismatched_source_span_never_fabricates_absolute_lines():
    finding, = audit_quote_attribution('甲说：“好。”', [doc('[甲] 「好。」', line_end=99)])
    assert finding['evidence'][0]['line'] is None


def test_summary_and_viewpoint_cannot_certify_direct_speech():
    document = {'content': '甲说：“好。”', 'metadata': {'viewpoint': '甲第一人称'}}
    finding, = audit_quote_attribution('甲说：“好。”', [document])
    assert finding['status'] == 'unmatched_literal'


def test_multiline_or_malformed_dialogue_is_not_labeled_narration():
    finding, = audit_quote_attribution('甲说：“等一等。”', [doc('[甲] 「等一等。\n先别走。」')])
    assert finding['status'] == 'unresolved_structure'
