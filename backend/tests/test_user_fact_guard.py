"""User-fact guard: linguistic boundaries and advisory/strict pipeline behavior."""
import pytest

from character.models import CompiledCharacterContext
from character.output_guard import UNSUPPORTED_USER_FACT, ReplyGuard, retryable_violations, validate_reply
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


@pytest.mark.parametrize('reply', [
    '你最喜欢读什么类型的书？', '你喜欢听哪种音乐？', '你喜欢研究什么课题？',
    '你喜欢收藏哪些东西？', '你擅长写哪一类文章？', '你经常练什么？',
    '你喜欢咖啡还是茶？', '你喜欢咖啡吗？', '如果你喜欢画画，可以聊聊。',
    '你喜欢咖啡，还是茶？', '你喜欢跑步，还是游泳？',
    '我不知道你喜欢什么。你喜欢学哪种乐器？',
])
def test_open_questions_and_conditionals_do_not_assert_a_value(reply):
    assert UNSUPPORTED_USER_FACT not in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


@pytest.mark.parametrize('reply', [
    '你喜欢读历史书。', '你擅长写诗。你喜欢什么？',
    '你喜欢咖啡，今晚看电影还是散步？', '你喜欢咖啡，今天喝了吗？',
    '如果有时间，另外你喜欢咖啡。', '我不知道你喜欢什么，但你肯定喜欢咖啡。',
    '你喜欢咖啡到什么程度？', '你为什么喜欢咖啡？',
    '你喜欢看书；你喜欢听什么音乐？', '你经常练琴。你喜欢什么？',
])
def test_a_later_question_or_remote_condition_cannot_license_a_claim(reply):
    assert UNSUPPORTED_USER_FACT in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


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


@pytest.mark.parametrize("message,reply", [
    pytest.param(
        '现在口味变了，我不吃香菜了。',
        '口味会变化。你现在的口味，应该更符合你现在的习惯吧。',
        id='nominal-reference',
    ),
    pytest.param(
        '今天整理了书架。',
        '书架整理好了？你最喜欢读什么类型的书？',
        id='open-question',
    ),
    pytest.param(
        '我只有完成工作才看电影。',
        '嗯，工作之后看电影也不错。不过，你是不是觉得看电影总是要等到事情做完才来得及？',
        id='polar-question',
    ),
    pytest.param(
        '我只有完成工作才看电影。',
        '你朋友是不是也喜欢用这种简单的话来介绍自己？',
        id='third-party-question',
    ),
    pytest.param(
        '我只有完成工作才看电影。',
        '请问你的室友最近是否经常阅读？',
        id='possessive-question',
    ),
])
async def test_non_assertive_reply_survives_shared_pipeline(message, reply):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return reply

    result = await generate_character_response(GenerationRequest(
        message=message,
        character_context=CompiledCharacterContext('', '', '', memory_status='no_match'),
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True)), model)
    assert result.reply == reply and len(calls) == 1
    assert not result.guard_violations
    assert not result.guard_retried


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
    assert not result.guard_retried
    assert UNSUPPORTED_USER_FACT in result.guard_violations


async def test_diagnostic_does_not_reactivate_after_independent_boundary_retry():
    outputs = iter(['建议你去散步。你喜欢陶艺。', '你喜欢陶艺。'])

    async def model(**kwargs):
        return next(outputs)

    result = await generate_character_response(GenerationRequest(message='不要给建议。',
        reply_guard=ReplyGuard(forbid_advice=True, forbid_unsupported_user_fact=True)), model)
    assert result.guard_retried
    assert result.reply == '你喜欢陶艺。'
    assert result.guard_post_retry_violations == (UNSUPPORTED_USER_FACT,)


def test_strict_mode_retains_opt_in_behavior():
    guard = ReplyGuard(forbid_unsupported_user_fact=True)
    reply = '你喜欢陶艺。'
    violations = validate_reply(reply, guard)
    assert retryable_violations(reply, guard, violations) == ()
    assert retryable_violations(reply, guard, violations, strict=True) == (UNSUPPORTED_USER_FACT,)
