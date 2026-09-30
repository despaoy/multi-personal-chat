import pytest

from character.models import CompiledCharacterContext
from character.output_guard import UNSUPPORTED_USER_FACT, ReplyGuard, validate_reply
from inference.generation_request import GenerationRequest, generate_character_response


@pytest.mark.parametrize('reply', [
    '你是不是喜欢画画？', '你是否习惯早起？', '你会不会更喜欢散步？',
    '你有没有一直练习写字？', '你最近是不是经常看电影？',
    '不过，你是不是觉得看电影总是要等到事情做完才来得及？',
    '我想知道你是否擅长写作？',
    '你朋友是不是也喜欢用这种简单的话来介绍自己？',
    '你的同事是否习惯早起？',
    '你母亲最近会不会更喜欢散步？',
    '请问你老师是否经常读书？',
    '你的室友有没有一直练习写字？',
])
def test_polar_questions_do_not_assert_their_embedded_predicate(reply):
    assert UNSUPPORTED_USER_FACT not in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


@pytest.mark.parametrize('reply', [
    '你是不是累了？你喜欢咖啡。', '你是否喜欢散步？不过你经常熬夜。',
    '你喜欢咖啡，你是不是累了？', '你是不是累了但你喜欢咖啡？',
    '你是不是累了，不过你一直喜欢咖啡。', '既然你一直喜欢咖啡，你是否想休息？',
    '你为什么一直喜欢咖啡？', '你是不是喜欢咖啡我很清楚。',
    '你是不是因为喜欢咖啡才熬夜？', '你是否因为擅长绘画才转行？',
    '你朋友一直喜欢咖啡。', '你的同事经常熬夜。',
    '你朋友为什么一直喜欢咖啡？',
    '你朋友是不是因为喜欢咖啡才熬夜？',
    '你朋友是否喜欢散步？不过你经常熬夜。',
    '你朋友是不是累了但你喜欢咖啡？',
])
def test_question_operator_cannot_license_an_independent_assertion(reply):
    assert UNSUPPORTED_USER_FACT in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


@pytest.mark.asyncio
@pytest.mark.parametrize('reply', [
    '嗯，工作之后看电影也不错。不过，你是不是觉得看电影总是要等到事情做完才来得及？',
    '你朋友是不是也喜欢用这种简单的话来介绍自己？',
    '请问你的室友最近是否经常阅读？',
])
async def test_traced_style_of_question_survives_shared_pipeline(reply):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return reply

    result = await generate_character_response(GenerationRequest(
        message='我只有完成工作才看电影。',
        character_context=CompiledCharacterContext('', '', '', memory_status='no_match'),
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True)), model)
    assert result.reply == reply and len(calls) == 1
    assert not result.guard_fallback and not result.guard_retried
