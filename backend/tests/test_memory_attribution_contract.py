"""A user-fact writer must state ownership rather than rely on a default."""
import json

import pytest

from character.memory_llm import MemoryResponseError, parse_llm_proposals


def response(owner='user', *, omit=False):
    candidate = dict(operation='ADD', kind='like', value='红茶', content='用户喜欢红茶',
                     evidence='我喜欢红茶', confidence=.95, qualifiers={})
    if not omit:
        candidate['attributed_to'] = owner
    return json.dumps({'memories':[candidate]}, ensure_ascii=False)

@pytest.mark.parametrize('owner', [None, False, 0, 1, [], {}, ''])
def test_invalid_attribution_raises_instead_of_defaulting_to_user(owner):
    with pytest.raises(ValueError, match='attributed_to'):
        parse_llm_proposals(response(owner), source_message='请记住，我喜欢红茶。')

def test_missing_attribution_raises():
    with pytest.raises(ValueError, match='attributed_to'):
        parse_llm_proposals(response(omit=True), source_message='请记住，我喜欢红茶。')

@pytest.mark.parametrize('owner', ['user', 'self', '用户', '本人'])
def test_explicit_user_attribution_preserves_supported_labels(owner):
    assert len(parse_llm_proposals(response(owner), source_message='请记住，我喜欢红茶。')) == 1

def test_wrong_fact_owner_is_an_explicit_model_protocol_error():
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(response('assistant'), source_message='请记住，我喜欢红茶。')
    assert caught.value.field == 'attributed_to'
