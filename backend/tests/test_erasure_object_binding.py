"""Scoped object binding uses stored predicates, not translations of opaque keys."""
import pytest

from character.erasure_authority import partial_erasure_plan
from character.memory_query import lookup_fields, profile_lookup_fields


def records():
    return (dict(id=1, memory_type='user_fact', memory_key='fact_pet', content='用户的猫叫豆沙、糯米'),
            dict(id=2, memory_type='user_fact', memory_key='user_major', content='用户的大学专业是海洋工程'))


def test_opaque_named_object_and_typed_field_bind_without_key_translation():
    plan = partial_erasure_plan('请忘掉我的猫名字和新增猫的记录，保留大学专业。', records())
    assert plan.valid and not plan.unresolved_protection
    assert plan.allowed_ids == ('1',) and plan.protected_ids == ('2',)
    assert plan.protected_keys == ('user_major',)


@pytest.mark.parametrize('noun', ['专业', '学科', '大学专业'])
def test_shared_field_names_work_in_either_delete_or_retain_position(noun):
    plan = partial_erasure_plan('请忘掉我的' + noun + '，保留我的猫的名字。', records())
    assert plan.allowed_ids == ('2',) and plan.protected_ids == ('1',)
    assert not plan.unresolved_protection


@pytest.mark.parametrize('owner', ['朋友的', '我的朋友的', '你的', '李四的'])
def test_other_owner_field_is_not_a_self_field_delete_target(owner):
    plan = partial_erasure_plan('请忘掉' + owner + '大学专业，保留我的猫的名字。', records())
    assert not plan.allowed_ids and plan.protected_ids == ('1',)


@pytest.mark.parametrize('noun', ['猫粮名字', '朋友的猫名字', '李四的猫名字'])
def test_named_object_does_not_match_longer_or_other_owner_noun(noun):
    plan = partial_erasure_plan('请忘掉我的' + noun + '，保留大学专业。', records())
    assert not plan.allowed_ids and plan.protected_ids == ('2',)


@pytest.mark.parametrize('change', [
    dict(content='猫叫豆沙'), dict(content='朋友的猫叫豆沙'),
    dict(content='用户的猫不叫豆沙'), dict(memory_type='shared_event'),
    dict(metadata=dict(content_semantics='quoted_source', speaker_role='user', described_subject='not_resolved')),
])
def test_unproved_named_subject_does_not_authorize_opaque_key(change):
    current = (dict(records()[0], **change), records()[1])
    plan = partial_erasure_plan('请忘掉我的猫名字，保留大学专业。', current)
    assert not plan.allowed_ids and plan.protected_ids == ('2',)


def test_unknown_retention_still_denies_all_deletion():
    plan = partial_erasure_plan('请忘掉我的猫名字，保留朋友的大学专业。', records())
    assert plan.unresolved_protection and not plan.allowed_ids


def test_university_major_has_one_shared_field_definition():
    assert lookup_fields('我的大学专业是什么？') == ('major',)
    assert profile_lookup_fields('请核对我的个人资料：大学专业是什么？') == ('major',)
