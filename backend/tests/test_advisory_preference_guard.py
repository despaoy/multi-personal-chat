"""Weak preference heuristics must not overwrite natural answers in normal chat."""
import pytest

from character.models import CompiledCharacterContext
from character.output_guard import UNSUPPORTED_USER_FACT, ReplyGuard, retryable_violations, validate_reply
from inference.generation_request import GenerationRequest, generate_character_response


@pytest.mark.parametrize('reply', [
    '你提到过喜欢拼模型。音乐类型目前不清楚。',
    '你喜欢羽毛球，但手腕康复前不能安排。',
    '你喜欢篆刻，你室友喜欢滑雪。不能确定你也喜欢滑雪。',
    '你喜欢徒步，至于最喜欢什么我不能确定。',
    # Also document the tradeoff: heuristic matching is NOT proof of support.
    '你一直喜欢咖啡。',
])
@pytest.mark.parametrize('status', ['no_match', 'available', 'retrieval_error'])
async def test_preference_wording_never_alone_causes_retry_or_whole_reply_fallback(reply, status):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return reply

    result = await generate_character_response(GenerationRequest(message='接着聊。',
        character_context=CompiledCharacterContext('', '', '', memory_status=status),
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True)), model)
    assert result.reply == reply and len(calls) == 1
    assert not result.guard_retried and not result.guard_fallback
    assert UNSUPPORTED_USER_FACT in result.guard_violations


async def test_diagnostic_does_not_reactivate_after_independent_boundary_retry():
    outputs = iter(['建议你去散步。你喜欢陶艺。', '你喜欢陶艺。'])

    async def model(**kwargs):
        return next(outputs)

    result = await generate_character_response(GenerationRequest(message='不要给建议。',
        reply_guard=ReplyGuard(forbid_advice=True, forbid_unsupported_user_fact=True)), model)
    assert result.guard_retried and not result.guard_fallback
    assert result.reply == '你喜欢陶艺。'
    assert result.guard_post_retry_violations == (UNSUPPORTED_USER_FACT,)


def test_strict_mode_retains_opt_in_behavior():
    guard = ReplyGuard(forbid_unsupported_user_fact=True)
    reply = '你喜欢陶艺。'
    violations = validate_reply(reply, guard)
    assert retryable_violations(reply, guard, violations) == ()
    assert retryable_violations(reply, guard, violations, strict=True) == (UNSUPPORTED_USER_FACT,)
