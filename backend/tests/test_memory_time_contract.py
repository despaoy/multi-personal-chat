"""Malformed model validity dates fail explicitly instead of dropping a fact."""
import json

import pytest

from character.memory_llm import parse_llm_proposals


def parse(**fields):
    row=dict(operation='ADD',kind='study_stage',value='大三',evidence='今年刚升大三',
             confidence=.96,attributed_to='user',**fields)
    return parse_llm_proposals(json.dumps({'memories':[row]},ensure_ascii=False),source_message='今年刚升大三')

@pytest.mark.parametrize('field',['valid_from','valid_to','valid_at','invalid_at'])
@pytest.mark.parametrize('bad',['not-a-date','2026-02-30',0,False,[],{}])
def test_invalid_date_or_structure_is_not_a_normal_empty_result(field,bad):
    with pytest.raises(ValueError,match='valid_from|valid_to'):
        parse(**{field:bad})

def test_inverted_validity_interval_is_explicit_error():
    with pytest.raises(ValueError,match='valid_from.*valid_to'):
        parse(valid_from='2026-10-09',valid_to='2026-10-08')

@pytest.mark.parametrize('empty',[None,''])
def test_unspecified_optional_times_are_still_valid(empty):
    proposal=parse(valid_from=empty,valid_to=empty)[0]
    assert proposal.valid_from==proposal.valid_to==''
