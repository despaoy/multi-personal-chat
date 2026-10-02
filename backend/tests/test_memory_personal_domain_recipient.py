"""Report recipients in personal-domain modifiers do not suppress scoped recall."""

from datetime import datetime, timezone

import pytest

from character.memory_service import CharacterMemoryService, _detect_memory_intents
from character.models import UserScope


@pytest.mark.parametrize(
    "query",
    [
        "结合我已告诉你的本人受理偏好判断本轮偏好条件。",
        "并结合我之前亲口告诉您的个人交通偏好。",
        "根据我目前告诉你的本人家居偏好。",
        "按我们明确告诉你们的个人出行偏好。",
        "关于咱们已经告知她的本人住宿偏好。",
        "请回忆我之前说给你的个人阅读偏好。",
        "结合我 已 告诉你 的 本人 办理 偏好。",
        "结合我已告诉你的饮料偏好。",
        "结合我已明确告诉你的本人服务偏好。",
    ],
)
def test_personal_domain_recipient_defers_ownership(query):
    intent = _detect_memory_intents(query)
    assert not intent.suppress_preference and not intent.preference


@pytest.mark.parametrize(
    "query",
    [
        "结合她已告诉你的本人受理偏好。",
        "结合我告诉你的朋友的受理偏好。",
        "结合我告诉你的本人朋友受理偏好。",
        "结合我告诉你的个人同事出行偏好。",
        "结合我告诉你的她的本人受理偏好。",
        "结合我告诉你的名字和你的偏好。",
        "结合我告诉你喜欢什么。",
        "结合我告诉你的本人受理的偏好。",
        "结合我已告诉你喜欢什么。",
    ],
)
def test_other_owner_or_unknown_relative_clause_stays_suppressed(query):
    assert _detect_memory_intents(query).suppress_preference


class Repo:
    async def list_memory_records(self, character, scope, **kwargs):
        assert character == "role" and scope.sender_id == "owner" and scope.conversation_id == "owner"
        rows = [
            ("现场受理", "用户喜欢现场受理", "我喜欢现场受理"),
            ("预约受理", "用户不喜欢预约受理", "我不喜欢预约受理"),
            (
                "加急受理",
                "用户喜欢加急受理但只有申请齐全且附件核验通过才使用",
                "我喜欢加急受理，但我只有申请齐全且附件核验通过才使用加急受理",
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
            for i, (name, content, evidence) in enumerate(rows)
        ]


async def test_scoped_three_preferences_and_conjunction_survive_filter():
    service = CharacterMemoryService(Repo(), semantic_enabled=False)
    trace = {}
    items, _ = await service.load_relevant_memories(
        "role",
        UserScope("web", "test", "owner", "owner", "private"),
        "结合我已告诉你的本人受理偏好，比较现场受理、预约受理、加急受理的条件。",
        diagnostics=trace,
    )
    assert trace["usable_records"] == 3 and {x.memory_id for x in items} == {"1", "2", "3"}
    assert any("申请齐全且附件核验通过" in x.content for x in items)
