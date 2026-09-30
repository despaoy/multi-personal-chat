from types import SimpleNamespace

import pytest

from knowledge.entity_scope import explicit_identity_subject, identity_evidence_subject
from knowledge.task_plan import plan_turn_tasks


@pytest.mark.parametrize('name,field', [('林远', '专业'), ('安宁', '工作地点'), ('Alex', '姓名')])
def test_memory_and_identity_have_separate_dependencies(name, field):
    message = f'不要分析，先说我的{field}，再告诉我{name}是谁。'
    tasks = plan_turn_tasks(message)
    assert [task.kind for task in tasks] == ['control', 'memory', 'content']
    assert all(message[task.start:task.end] == task.original for task in tasks)
    analysis = SimpleNamespace(entities=[name], normalized_query=message)
    assert identity_evidence_subject(analysis) == name
    assert explicit_identity_subject(analysis) == ''


@pytest.mark.parametrize('message', [
    '先说我的专业适合什么工作，再告诉我林远是谁。',
    '先说我朋友的专业，再告诉我林远是谁。',
    '先说我的专业，再告诉我林远的哥哥是谁。',
    '先说我的专业，再告诉我林远是谁，再解释他的选择。',
    '请翻译“先说我的专业，再告诉我林远是谁”。',
    '如果我的专业改变了，再告诉我林远是谁。',
])
def test_unknown_dependent_or_quoted_tasks_do_not_narrow_evidence(message):
    analysis = SimpleNamespace(entities=['林远'], normalized_query=message)
    assert not identity_evidence_subject(analysis)
    assert plan_turn_tasks(message)
