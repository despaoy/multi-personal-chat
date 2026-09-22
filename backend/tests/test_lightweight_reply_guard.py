"""Lightweight chat avoids style-only inference while keeping hard boundaries."""

import pytest

from character.output_guard import (
    AUTONOMY_BOUNDARY_IGNORED,
    GENERIC_ASSISTANT_TEMPLATE,
    ReplyGuard,
    retryable_violations,
    validate_reply,
)
from inference.generation_request import GenerationRequest, generate_character_response


@pytest.mark.parametrize(
    ("reply", "guard"),
    [
        ("祝你好运！", ReplyGuard(forbid_generic_templates=True)),
        ("嗯，晚安。有需要时随时来找我。", ReplyGuard(closing=True, respect_autonomy=True)),
        ("晚安，好好休息。", ReplyGuard(closing=True)),
        ("真不错，你怎么做到的？", ReplyGuard(positive_sharing=True)),
        ("我是个很认真的人。", ReplyGuard(repair=True)),
    ],
)
async def test_style_or_normal_farewell_uses_one_call_without_template_fallback(reply, guard):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return reply

    assert validate_reply(reply, guard)  # Previously triggered regeneration.
    result = await generate_character_response(GenerationRequest(message="你好", reply_guard=guard), generate)
    assert result.reply == reply
    assert len(calls) == 1
    assert result.guard_violations
    assert not result.guard_retried
    assert not result.guard_fallback


@pytest.mark.parametrize(
    ("reply", "guard"),
    [
        ("晚安，我们明天继续好吗？", ReplyGuard(closing=True)),
        ("建议你去散步。", ReplyGuard(forbid_advice=True)),
        ("晚安，你应该先做出决定。", ReplyGuard(closing=True, respect_autonomy=True)),
        ("先休息一下。", ReplyGuard(require_urgent_safety_check=True)),
        ("你一直喜欢咖啡。", ReplyGuard(forbid_unsupported_user_fact=True)),
    ],
)
async def test_explicit_boundaries_safety_and_facts_still_get_bounded_retry(reply, guard):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return reply

    result = await generate_character_response(GenerationRequest(message="你好", reply_guard=guard), generate)
    assert len(calls) == 2
    assert result.guard_retried


async def test_successful_hard_retry_with_soft_style_issue_is_not_replaced_by_fallback():
    outputs = iter(["你一直喜欢咖啡。", "祝你好运！"])

    async def generate(**kwargs):
        return next(outputs)

    result = await generate_character_response(
        GenerationRequest(
            message="你好", reply_guard=ReplyGuard(forbid_unsupported_user_fact=True, forbid_generic_templates=True)
        ),
        generate,
    )
    assert result.guard_retried
    assert result.reply == "祝你好运！"
    assert result.guard_post_retry_violations == (GENERIC_ASSISTANT_TEMPLATE,)
    assert not result.guard_fallback


def test_explicit_autonomy_ack_is_not_removed_by_lightweight_farewell():
    guard = ReplyGuard(closing=True, respect_autonomy=True, require_autonomy_ack=True)
    reply = "晚安。"
    assert AUTONOMY_BOUNDARY_IGNORED in retryable_violations(reply, guard, validate_reply(reply, guard))


async def test_strict_mode_is_explicit_and_still_retries_style():
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return "祝你好运！" if len(calls) == 1 else "嗯。"

    result = await generate_character_response(
        GenerationRequest(
            message="你好", reply_guard=ReplyGuard(forbid_generic_templates=True), reply_guard_mode="strict"
        ),
        generate,
    )
    assert len(calls) == 2
    assert result.guard_retried


async def test_invalid_mode_fails_before_model_call():
    async def generate(**kwargs):
        pytest.fail("invalid mode must fail before inference")

    with pytest.raises(ValueError, match="guard mode"):
        await generate_character_response(GenerationRequest(message="你好", reply_guard_mode="invalid"), generate)
