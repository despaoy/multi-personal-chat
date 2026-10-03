"""One independent other-owner task does not veto a deferred scoped report read."""

from datetime import datetime, timezone

import pytest

from character.memory_service import CharacterMemoryService, _detect_memory_intents
from character.models import UserScope


@pytest.mark.parametrize(
    "query,field",
    [
        ("结合我已告诉你的本人办理偏好判断本轮条件。另回答你最喜欢哪个方案？", "preference"),
        ("你最喜欢哪个方案？结合我已告诉你的本人办理偏好列出记录。", "preference"),
        ("按我告诉你的饮食偏好查桂花茶。你平时最喜欢什么？", "preference"),
        ("根据我告诉你的饮料偏好核对记录。另说她喜欢什么。", "preference"),
        ("我告诉过你的名字是什么？你的名字是什么？", "name"),
        ("你的名字是什么？请回忆我告诉你的名字。", "name"),
        ("结合咱们告知您的名字列出原话。她叫什么名字？", "name"),
        ("并结合我们之前告诉你们的个人出行偏好核对。你喜欢什么？", "preference"),
    ],
)
def test_mixed_deferred_report_and_other_task_remains_retrievable(query, field):
    intents = _detect_memory_intents(query)
    assert not getattr(intents, "suppress_" + field)
    assert not getattr(intents, field), "Report retrieval must not fabricate ownership or positive intent"


@pytest.mark.parametrize(
    "query,field",
    [
        ("她告诉你的饮食偏好是什么？你喜欢什么？", "preference"),
        ("结合我告诉你的朋友的办理偏好。你最喜欢哪项？", "preference"),
        ("结合我告诉你的本人同事办理偏好。你喜欢什么？", "preference"),
        ("假如我告诉你的饮料偏好，你喜欢什么？", "preference"),
        ("不要读取我告诉你的饮料偏好。你喜欢什么？", "preference"),
        ("我告诉你她的名字。你的名字是什么？", "name"),
        ("回忆我告诉你的名字。你喜欢什么？", "preference"),
        ("回忆我告诉你的饮料偏好。你的名字是什么？", "name"),
    ],
)
def test_other_or_excluded_or_different_field_keeps_suppression(query, field):
    assert getattr(_detect_memory_intents(query), "suppress_" + field)


class Repo:
    async def list_memory_records(self, character, scope, **kwargs):
        assert character == "role" and scope.sender_id == "owner" and scope.conversation_id == "owner"
        values = [
            ("窗口受理", "用户喜欢窗口受理", "我喜欢窗口受理"),
            ("代办受理", "用户不喜欢代办受理", "我不喜欢代办受理"),
            (
                "加速受理",
                "用户喜欢加速受理，但只有完整表格已提交且身份核验通过才使用",
                "我喜欢加速受理，但我只有完整表格已提交且身份核验通过才使用加速受理",
            ),
        ]
        return [
            dict(
                id=str(i + 1),
                memory_key="preference_" + name,
                memory_type="user_fact",
                content=content,
                status="active",
                importance=0.8,
                evidence=[evidence],
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            for i, (name, content, evidence) in enumerate(values)
        ]


async def test_three_scoped_candidates_survive_mixed_task_without_history():
    trace = {}
    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    items, _ = await service.load_relevant_memories(
        "role",
        UserScope("web", "test", "owner", "owner", "private"),
        "结合我已告诉你的本人办理偏好，比较窗口受理、代办受理和加速受理条件。另回答你最喜欢哪项？",
        diagnostics=trace,
    )
    assert trace["records_read"] == trace["usable_records"] == 3 and {x.memory_id for x in items} == {"1", "2", "3"}
    assert any("完整表格已提交且身份核验通过" in x.content for x in items)


async def test_pure_character_task_still_has_no_private_candidates():
    trace = {}
    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    items, _ = await service.load_relevant_memories(
        "role",
        UserScope("web", "test", "owner", "owner", "private"),
        "你最喜欢窗口受理还是加速受理？",
        diagnostics=trace,
    )
    assert trace["records_read"] == 3 and trace["usable_records"] == 0 and not items


def test_explicit_self_intent_still_has_positive_priority():
    intent = _detect_memory_intents("我告诉你我的饮料偏好。你最喜欢什么？")
    assert intent.preference and not intent.suppress_preference
