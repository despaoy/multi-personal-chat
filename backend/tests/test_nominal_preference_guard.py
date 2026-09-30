"""Mentioning preferences is different from asserting their concrete values."""
import pytest

from character.models import CompiledCharacterContext
from character.output_guard import UNSUPPORTED_USER_FACT, ReplyGuard, validate_reply
from inference.generation_request import GenerationRequest, generate_character_response


@pytest.mark.parametrize('reply', [
    '我尊重你的偏好。', '这取决于你的习惯吧。', '不替你决定你的喜好。',
    '你现在的口味，应该更符合你现在的习惯吧。',
    '先听听你的喜好，再讨论。', '我不了解你的习惯。',
    '你的偏好呢？',
])
def test_bare_possessive_reference_is_not_a_concrete_fact(reply):
    assert UNSUPPORTED_USER_FACT not in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


@pytest.mark.parametrize('reply', [
    '你的偏好是甜食。', '你的习惯是早起。', '你的喜好偏向爵士乐。',
    '你对甜品的喜好可是出了名的。',
    '我尊重你的偏好。你一直喜欢咖啡。',
    '先听听你的喜好，不过你总是熬夜。',
    '你的习惯，你喜欢读历史书。',
    '我记得你以前说过喜欢那种味道。',
])
def test_nominal_reference_does_not_license_a_concrete_value_or_another_claim(reply):
    assert UNSUPPORTED_USER_FACT in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


@pytest.mark.asyncio
async def test_nominal_reference_is_not_replaced_by_no_memory_fallback():
    original = '口味会变化。你现在的口味，应该更符合你现在的习惯吧。'
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return original

    result = await generate_character_response(GenerationRequest(
        message='现在口味变了，我不吃香菜了。',
        character_context=CompiledCharacterContext('', '', '', memory_status='no_match'),
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True)), model)
    assert result.reply == original
    assert len(calls) == 1 and not result.guard_fallback and not result.guard_violations
