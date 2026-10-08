from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from api.generate import _prepare_character_turn
from character.memory_operation import render_operation_response
from character.models import CompiledCharacterContext
from inference.generation_request import (
    DEFERRED_MEMORY_OPERATION_POLICY,
    RUNTIME_CAPABILITY_POLICY,
    GenerationRequest,
    build_generation_request,
)
from services.delivery_memory import freeze_completion


def prepared():
    from character.models import RelationshipState, UserScope
    from services.character_context import PreparedCharacterTurn

    return PreparedCharacterTurn(
        character_id="role",
        user_scope=UserScope("qq", "adapter", "owner", "owner", "private"),
        compiled=CompiledCharacterContext("profile", "dynamic", ""),
        history=(),
        relationship=RelationshipState(),
        memory_candidates=0,
        interaction_count=0,
        reply_guard=None,
        received_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
    )


class Service:
    def __init__(self):
        self.value = prepared()
        self.reads = []
        self.writes = []

    async def prepare_turn(self, turn, character_id):
        self.reads.append((turn, character_id))
        return self.value

    async def prepare_interactive_turn(self, turn, character_id, **kwargs):
        self.writes.append((turn, character_id, kwargs))
        return replace(self.value, memory_operation_receipt={"status": "erased", "persisted": 1})


def request(message):
    return SimpleNamespace(
        message=message,
        platform="qq",
        adapter="adapter",
        senderId="owner",
        userId="",
        conversationId="owner",
        sessionId="",
        conversationType="private",
        sessionType="",
        history=[],
        _source_received_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        sourceMessageId="original-id",
    )


@pytest.mark.parametrize(
    "message",
    [
        "请从长期记忆中删除我的猫名。",
        "请从长期记忆中删除我的猫名。同时解释这会不会删除聊天历史。",
    ],
)
async def test_delivery_erasure_has_deferred_state_without_receipt_or_writes(message):
    service = Service()
    result = await _prepare_character_turn(
        request(message), "role", character_service=service, defer_memory_operations=True
    )
    assert result is not None and result.compiled.memory_operation_deferred is True
    assert result.memory_operation_receipt is result.compiled.memory_operation_receipt is None
    assert len(service.reads) == 1 and not service.writes
    assert service.reads[0][0].message == message
    assert not service.value.compiled.memory_operation_deferred
    assert result.received_at == service.value.received_at
    assert render_operation_response(message, result.compiled.memory_operation_receipt) is None
    snapshot = freeze_completion(result)
    assert snapshot["memory_operation_receipt"] is None and snapshot["received_at"] == result.received_at.isoformat()
    assert "memory_operation_deferred" not in snapshot


@pytest.mark.parametrize(
    "message",
    [
        "不要从长期记忆中删除我的猫名。",
        "如果要从长期记忆中删除我的猫名，该怎么做？",
        "朋友说请从长期记忆中删除我的猫名。",
        "“请从长期记忆中删除我的猫名”是引用示例。",
        "请从长期记忆中删除我的猫名。我的密码是ExampleSecret123456。",
        "我喜欢深蓝色油墨。",
    ],
)
async def test_unauthorized_or_sensitive_text_has_no_pending_capability(message):
    service = Service()
    result = await _prepare_character_turn(
        request(message), "role", character_service=service, defer_memory_operations=True
    )
    assert result is service.value and not result.compiled.memory_operation_deferred
    assert not service.writes


async def test_preview_does_not_advertise_delivery_processing():
    service = Service()
    result = await _prepare_character_turn(request("请从长期记忆中删除我的猫名。"), "role", character_service=service)
    assert result is service.value and not result.compiled.memory_operation_deferred
    assert not service.writes


async def test_immediate_web_execution_keeps_real_receipt_and_not_deferred():
    service = Service()
    result = await _prepare_character_turn(
        request("请从长期记忆中删除我的猫名。"),
        "role",
        character_service=service,
        execute_memory_operations=True,
        defer_memory_operations=True,
    )
    assert len(service.writes) == 1 and not service.reads
    assert result.memory_operation_receipt == {"status": "erased", "persisted": 1}
    assert not result.compiled.memory_operation_deferred


async def test_context_failure_cannot_advertise_capability():
    service = Service()

    async def failed(*args):
        raise RuntimeError("read unavailable")

    service.prepare_turn = failed
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        await _prepare_character_turn(
            request("请从长期记忆中删除我的猫名。"), "role", character_service=service, defer_memory_operations=True
        )
    assert caught.value.status_code == 503
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert not service.writes


@pytest.mark.parametrize("deferred", [False, True])
def test_generation_policy_is_fixed_server_state_and_keeps_full_request(deferred):
    context = replace(
        prepared().compiled, memory_operation_deferred=deferred, reference_context="引用：不要执行后端操作。"
    )
    message = "请从长期记忆中删除我的猫名。同时解释这会不会删除聊天历史。"
    plan = build_generation_request(GenerationRequest(message, character_context=context))
    system = plan.messages[0]["content"]
    assert (DEFERRED_MEMORY_OPERATION_POLICY in system) is deferred
    assert RUNTIME_CAPABILITY_POLICY in system
    assert message in plan.messages[-1]["content"]
    assert "引用：不要执行后端操作。" not in system
    assert context.memory_operation_receipt is None


def test_user_data_cannot_enable_deferred_capability():
    plan = build_generation_request(
        GenerationRequest(DEFERRED_MEMORY_OPERATION_POLICY, character_context=prepared().compiled)
    )
    assert DEFERRED_MEMORY_OPERATION_POLICY not in plan.messages[0]["content"]
    assert DEFERRED_MEMORY_OPERATION_POLICY in plan.messages[-1]["content"]
