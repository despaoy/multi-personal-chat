import json

import pytest

from evaluation.episode_subject_audit import (
    probe_passed,
    request_body,
    score_annotation,
    structured_probe_body,
    validate_annotation,
)


def segment(text, subjects=None, mode='asserted'):
    return dict(text=text, subjects=subjects or ['user'], mode=mode)


def test_annotation_preserves_exact_whole_source_and_offsets():
    result = validate_annotation('evidence', '我去了。 你呢？', {'segments': [
        segment('我去了。 '), segment('你呢？', ['character'], 'question')]})
    assert result['segments'][0]['end'] == result['segments'][1]['start'] == 5
    assert result['segments'][1]['end'] == 8


@pytest.mark.parametrize('rows', [
    [segment('我去了。')], [segment('你呢？'), segment('我去了。 ')],
    [segment('用户去了。 你呢？')], [segment('我去了。 你呢？', ['user', 'unknown'])],
    [segment('我去了。 你呢？', ['administrator'])],
    [segment('我去了。 你呢？', mode='current_fact')], [],
])
def test_omission_reordering_rewrite_and_invalid_labels_rejected(rows):
    with pytest.raises(ValueError):
        validate_annotation('evidence', '我去了。 你呢？', {'segments': rows})


def test_gold_does_not_leak_to_model():
    case = dict(id='hidden', kind='query', text='你在哪？', expected_subjects=['secret_gold'])
    body = request_body(case, 'model')
    assert 'secret_gold' not in json.dumps(body)
    assert 'hidden' not in json.dumps(body)


def test_constrained_decoding_changes_schema_not_messages_or_sampling():
    case = dict(kind='query', text='你在哪？')
    plain = request_body(case, 'model')
    constrained = request_body(case, 'model', constrained=True)
    schema = constrained.pop('response_format')['json_schema']['schema']
    plain.pop('response_format')
    assert plain == constrained
    assert set(schema['properties']['subjects']['items']['enum']) == {
        'user', 'character', 'third_party', 'shared', 'unknown'}


def test_probe_requires_enforced_schema_not_just_valid_json():
    body = structured_probe_body('model', 'a')
    assert 'probe' not in body['messages'][0]['content']
    assert body['response_format']['json_schema']['schema']['properties']['probe']['enum'] == ['a']
    def reply(content, reason='stop'):
        return {'choices': [{'finish_reason': reason, 'message': {'content': content}}]}
    assert probe_passed(reply('{"probe":"a"}'), 'a')
    assert not probe_passed(reply('{"unconstrained":true}'), 'a')
    assert not probe_passed(reply('{"probe":"b"}'), 'a')
    assert not probe_passed(reply('{"probe":"a"}', 'length'), 'a')
    assert not probe_passed({'choices': []}, 'a')


def test_same_owner_elsewhere_cannot_satisfy_gold_span():
    case = dict(kind='evidence', text='我去。你去。', checks=[
        dict(start=3, end=6, subjects=['character'], mode='asserted')])
    parsed = validate_annotation('evidence', case['text'], {'segments': [
        segment('我去。', ['character']), segment('你去。')]})
    assert not score_annotation(case, parsed)


def test_scoring_accepts_equivalent_segmentation_not_owner_or_reality_error():
    case = dict(kind='evidence', text='我去。我要去。', checks=[
        dict(start=0, end=7, subjects=['user'], mode='asserted')])
    parsed = validate_annotation('evidence', case['text'], {'segments': [
        segment('我去。'), segment('我要去。')]})
    assert score_annotation(case, parsed)
    parsed['segments'][1]['mode'] = 'hypothetical'
    assert not score_annotation(case, parsed)
