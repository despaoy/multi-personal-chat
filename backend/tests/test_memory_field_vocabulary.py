import pytest

from character.memory_query import MEMORY_FIELD_NAMES, lookup_fields, plan_memory_query
from knowledge.task_plan import plan_turn_tasks


@pytest.mark.parametrize('noun,field', MEMORY_FIELD_NAMES.items())
def test_every_field_noun_has_one_retrieval_and_execution_mapping(noun, field):
    query = f'我的{noun}是什么？'
    assert plan_memory_query(query).fields == (field,)
    assert lookup_fields(query) == (field,)
    tasks = plan_turn_tasks(f'先说我的{noun}，再告诉我林远是谁。')
    assert tasks[0].kind == 'memory' and tasks[0].memory_fields == (field,)


@pytest.mark.parametrize('noun', MEMORY_FIELD_NAMES)
def test_aliases_do_not_swallow_other_owners_or_nonlookup_tasks(noun):
    assert lookup_fields(f'我朋友的{noun}是什么？') == ()
    assert lookup_fields(f'我的{noun}应该怎么改变？') == ()


@pytest.mark.parametrize('query', ['我的来自', '我的住在', '我的住哪里'])
def test_retrieval_predicates_are_not_nouns(query):
    assert lookup_fields(query) == ()
