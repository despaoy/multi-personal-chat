"""Ambiguous reviewer fields cannot become a successful interaction state."""
import json

import pytest
from fastapi import HTTPException
from test_semantic_character_context import _service
from test_semantic_state_estimator import _review_payload, _Reviewer, _state

from character.semantic_state_estimator import SemanticStateEstimator
from db.schemas import MessageRequest

QUERY = '我今天约好的朋友临时取消了见面，我有点失落。现在还是按你刚才说的那个办法陪我聊一会儿吧。'
HISTORY = (
    {'role': 'user', 'content': '朋友临时取消了今天的见面，我有点失落，暂时不需要建议。'},
    {'role': 'assistant', 'content': '可以先休息一会儿，我陪你聊，不急着分析或给建议。'},
)


def conflicting_json(kind):
    payload = _review_payload()
    if kind == 'list_entry':
        payload['state']['user_needs'] = [dict(signal_id='validation', score=.8)]
    raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    if kind == 'root':
        return '{"state":{},' + raw[1:]
    key = {'scalar': '"confidence":0.72', 'signal': '"validation":0.8', 'list_entry': '"score":0.8'}[kind]
    assert raw.count(key) == 1
    name = key.split(':')[0]
    return raw.replace(key, name + ':9,' + key)


@pytest.mark.parametrize('kind', ['root', 'scalar', 'signal', 'list_entry'])
@pytest.mark.parametrize('fenced', [False, True])
async def test_duplicate_json_keys_fail_before_applied_diagnostics(kind, fenced):
    raw = conflicting_json(kind)
    reviewer = _Reviewer('```json\n' + raw + '\n```' if fenced else raw)
    estimator = SemanticStateEstimator(reviewer)
    with pytest.raises(ValueError, match='duplicate_json_key'):
        await estimator.refine_with_diagnostics(QUERY, HISTORY, _state(confidence=.2))
    assert len(reviewer.calls) == 1


@pytest.mark.parametrize('field,signal', [('situation_scores', 'emotional'),
                                         ('user_acts', 'seek_support'), ('user_needs', 'validation')])
@pytest.mark.parametrize('scores', [(0.0, .8), (.8, 0.0)])
async def test_duplicate_signal_ids_fail_even_when_one_score_is_zero(field, signal, scores):
    payload = _review_payload(**{field: [dict(signal_id=signal, score=score) for score in scores]})
    reviewer = _Reviewer(payload)
    with pytest.raises(ValueError, match='duplicate signal id'):
        await SemanticStateEstimator(reviewer).refine(QUERY, HISTORY, _state(confidence=.2))
    assert len(reviewer.calls) == 1


@pytest.mark.parametrize('representation', ['mapping', 'flat_mapping', 'json', 'flat_json', 'json_fence', 'bare_fence'])
async def test_unique_list_signals_preserve_supported_response_forms_and_zero_filtering(representation):
    payload = _review_payload(
        situation_scores=[dict(signal_id='emotional', score=.82), dict(signal_id='daily', score=0.0)],
        user_acts=[dict(signal_id='seek_support', score=.76), dict(signal_id='greeting', score=0.0)],
        user_needs=[dict(signal_id='validation', score=.8), dict(signal_id='guidance', score=0.0)],
    )
    if representation in {'flat_mapping', 'flat_json'}:
        payload = payload['state']
    raw = payload if representation.endswith('mapping') else json.dumps(payload, ensure_ascii=False)
    if representation in {'json_fence', 'bare_fence'}:
        raw = ('```json\n' if representation == 'json_fence' else '```\n') + raw + '\n```'
    outcome = await SemanticStateEstimator(_Reviewer(raw)).refine_with_diagnostics(QUERY, HISTORY, _state(confidence=.2))
    assert outcome.status == 'applied' and outcome.review_confidence == .72
    assert [(item.signal_id, item.score) for item in outcome.state.situation_scores] == [('emotional', .82)]
    assert [(item.signal_id, item.score) for item in outcome.state.user_acts] == [('seek_support', .76)]
    assert [(item.signal_id, item.score) for item in outcome.state.user_needs] == [('validation', .8)]


@pytest.mark.parametrize('kind', ['scalar', 'duplicate_signal'])
async def test_invalid_review_stops_api_preparation_then_healthy_retry_releases_guard(kind):
    from api import generate

    raw = (conflicting_json(kind) if kind == 'scalar' else _review_payload(
        user_needs=[dict(signal_id='validation', score=0.0), dict(signal_id='validation', score=.8)]))
    reviewer = _Reviewer(raw)
    service = _service(SemanticStateEstimator(reviewer))
    request = MessageRequest(message=QUERY, history=list(HISTORY), senderId='synthetic-review',
                             conversationId='synthetic-review')
    with pytest.raises(HTTPException) as caught:
        await generate._prepare_character_turn(request, 'tsukiyashiro_kisaki', character_service=service)
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, ValueError)
    assert len(reviewer.calls) == 1
    reviewer.result = json.dumps(_review_payload(), ensure_ascii=False)
    prepared = await generate._prepare_character_turn(request, 'tsukiyashiro_kisaki', character_service=service)
    assert prepared.semantic_review_status == 'applied' and len(reviewer.calls) == 2
