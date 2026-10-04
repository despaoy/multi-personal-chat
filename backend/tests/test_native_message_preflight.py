"Independent API-plan contracts, never native sources or paid model evidence."

import pytest
from pydantic import ValidationError

from evaluation.mixed_subject_history_probe import validate_native_messages


def test_invalid_later_remaining_task_is_rejected_before_plan_runs():
    fixture = dict(question="核对当前偏好与旧版本。", history_advancement_tasks=[
        dict(message="第三方资料：甲费用7元，时限2小时；请按表列出。"),
        dict(message="紫" * 8001),
    ])
    with pytest.raises(ValidationError):
        validate_native_messages(fixture, advance_history=True)


def test_inherited_tasks_are_not_revalidated_as_new_requests():
    fixture = dict(question="核对当前偏好与旧版本。", history_advancement_tasks=[
        dict(message="紫" * 8001), dict(message="第三方资料：甲费用7元，时限2小时。"),
    ])
    validate_native_messages(fixture, advance_history=True, completed_history=1)


@pytest.mark.parametrize("message", ["", "紫" * 8001])
def test_final_question_must_satisfy_real_api_contract(message):
    with pytest.raises(ValidationError):
        validate_native_messages(dict(question=message))


def test_author_only_validates_real_authoring_without_sending_future_question():
    fixture = dict(question="紫" * 8001, additional_source_message="请记住：我现在不喜欢紫茶。")
    validate_native_messages(fixture, authoring=True, author_only=True)


def test_complete_api_boundary_message_is_kept_whole():
    validate_native_messages(dict(question="紫" * 8000))
