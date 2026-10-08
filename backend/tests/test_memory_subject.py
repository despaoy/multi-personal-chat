import json

import pytest

from character.memory_llm import parse_llm_proposals
from character.memory_query import plan_memory_query
from character.memory_subject import explicitly_other_subject


@pytest.mark.parametrize('owner', ['我姐姐', '我的弟弟', '我父亲', '我的老师', '我室友', '她', '你'])
@pytest.mark.parametrize('predicate,value,kind', [('在', '青州', 'location'),
                                               ('喜欢', '陶艺', 'like'),
                                               ('叫', '林澈', 'name'),
                                               ('的专业是', '地质学', 'major'),
                                               ('是', '研二', 'study_stage'),
                                               ('在', '博物馆工作', 'workplace')])
def test_explicit_other_subject_is_not_a_personal_field(owner, predicate, value, kind):
    source = owner + predicate + value + '。'
    output = json.dumps(dict(memories=[dict(operation="ADD", kind=kind, value=value, evidence=source,
        content='用户说自己' + predicate + value, attributed_to='user', confidence=0.99)]))
    assert parse_llm_proposals(output, source_message=source) == []


@pytest.mark.parametrize('owner', ['我妹妹', '我的哥哥', '我室友', '她', '你'])
@pytest.mark.parametrize('verb', ['已经搬到', '刚搬家到', '刚刚搬到'])
def test_relocation_owner_is_checked_before_personal_location_admission(owner, verb):
    source = owner + verb + '衡阳了。'
    output = json.dumps(dict(memories=[dict(operation="ADD", kind='location', value='衡阳', evidence=source,
        content='用户搬到了衡阳', attributed_to='user', confidence=.99)]))
    assert parse_llm_proposals(output, source_message=source) == []
    assert explicitly_other_subject(source=source, evidence='衡阳', value='衡阳')


def test_trimmed_quote_cannot_drop_its_original_subject():
    assert explicitly_other_subject(source='我妹妹现在住在滨州。', evidence='住在滨州', value='滨州')


def test_other_clause_does_not_discard_separate_self_fact():
    source = '我哥哥在青州，我自己在临沂学雕刻。'
    assert not explicitly_other_subject(source=source, evidence='我自己在临沂学雕刻', value='临沂')
    output = json.dumps(dict(memories=[dict(operation="ADD", kind='location', value='临沂',
        evidence='我自己在临沂学雕刻', confidence=0.99)]))
    assert len(parse_llm_proposals(output, source_message=source)) == 1


@pytest.mark.parametrize('source,value', [('我喜欢姐姐送的围巾。', '围巾'),
    ('我在父亲的公司工作。', '公司'), ('我和姐姐都在青州。', '青州'),
    ('我姐姐在青州，我也在青州。', '青州'), ('青州是姐姐推荐我去的。', '青州')])
def test_object_mentions_shared_or_unknown_are_not_certified_as_other(source, value):
    assert not explicitly_other_subject(source=source, evidence=source, value=value)


def test_question_owner_grammar_includes_existing_write_vocabulary():
    assert plan_memory_query('我室友的专业是什么？').excluded_fields == ('major',)
    assert plan_memory_query('我的专业是什么？').fields == ('major',)


def test_shared_event_is_not_reclassified_by_personal_field_guard():
    source = '我姐姐喜欢我送给她的陶杯。'
    output = json.dumps(dict(memories=[dict(operation="ADD", kind='shared_event', value='陶杯',
        evidence=source, confidence=0.99)]))
    assert len(parse_llm_proposals(output, source_message=source)) == 1
