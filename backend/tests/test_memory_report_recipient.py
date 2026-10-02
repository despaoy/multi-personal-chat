"""A report recipient is not ownership proof; explicit other topics stay suppressed."""

from datetime import datetime, timezone

import pytest

from character.memory_service import CharacterMemoryService, _detect_memory_intents
from character.models import UserScope

QUERY = "按我目前明确告诉你的饮料偏好，列出喜欢的两种饮料、不喜欢的一种饮料，并完整说明喝低咖啡因咖啡的必要条件；条件满足也不能推断我一定会喝。"


@pytest.mark.parametrize(
    "query,field",
    [
        (QUERY, "preference"),
        ("我之前明确告诉你的本人饮料偏好是什么？", "preference"),
        ("根据我曾经亲口告知您的饮料偏好列出记录。", "preference"),
        ("请回忆我告诉你们的偏好。", "preference"),
        ("我们刚才说给你的爱好是什么？", "preference"),
        ("咱们已经讲给她的名字是什么？", "name"),
        ("你还记得我告诉过你的名字吗？", "name"),
        ("按我目前明确告诉你 的饮料 偏好列出记录。", "preference"),
    ],
)
def test_recipient_defers_without_claiming_reporter_is_topic_owner(query, field):
    intent = _detect_memory_intents(query)
    assert not getattr(intent, "suppress_" + field)
    assert not getattr(intent, field)


@pytest.mark.parametrize(
    "query,field",
    [
        ("你目前最喜欢什么？", "preference"),
        ("我想知道你喜欢什么。", "preference"),
        ("我告诉你最喜欢什么。", "preference"),
        ("我告诉你喜欢什么。", "preference"),
        ("我告诉你你喜欢什么。", "preference"),
        ("我告诉你她喜欢什么。", "preference"),
        ("我告诉你她的饮料偏好。", "preference"),
        ("她告诉你的饮料偏好是什么？", "preference"),
        ("根据她明确告诉你的饮料偏好列出。", "preference"),
        ("你的名字是什么？", "name"),
        ("我告诉你她的名字。", "name"),
        ("我告诉你你的名字。", "name"),
        ("按我目前告诉你的朋友的饮料偏好列出。", "preference"),
    ],
)
def test_explicit_other_owner_remains_suppressed(query, field):
    assert getattr(_detect_memory_intents(query), "suppress_" + field)


@pytest.mark.parametrize(
    "query,field",
    [
        ("我告诉你我的饮料偏好是什么。", "preference"),
        ("你告诉我喜欢什么。", "preference"),
        ("我告诉你我的名字。", "name"),
    ],
)
def test_explicit_self_after_recipient_still_uses_self_intent(query, field):
    intent = _detect_memory_intents(query)
    assert getattr(intent, field) and not getattr(intent, "suppress_" + field)


class Repo:
    async def list_memory_records(self, *args, **kwargs):
        return [
            dict(
                id="1",
                memory_key="preference_桂花乌龙茶",
                memory_type="user_fact",
                content="用户喜欢桂花乌龙茶",
                importance=0.8,
                status="active",
                evidence=["我喜欢桂花乌龙茶。"],
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        ]


@pytest.mark.asyncio
async def test_recipient_query_keeps_relevant_scoped_candidate():
    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    trace = {}
    items, _ = await service.load_relevant_memories(
        "role",
        UserScope("web", "test", "owner", "owner", "private"),
        "按我目前明确告诉你的饮料偏好，列出桂花乌龙茶的记录。",
        diagnostics=trace,
    )
    assert trace["usable_records"] == 1 and items
    assert items[0].memory_id == "1"
