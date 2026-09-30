"""Questions must not suppress real assertions in neighboring clauses."""
import pytest

from character.models import CompiledCharacterContext
from character.output_guard import UNSUPPORTED_USER_FACT, ReplyGuard, validate_reply
from evaluation.replay_reply_guard import replay_trace
from inference.generation_request import GenerationRequest, generate_character_response


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


@pytest.mark.asyncio
async def test_open_question_is_not_replaced_with_no_memory_fallback():
    calls = []
    original = '书架整理好了？你最喜欢读什么类型的书？'

    async def model(**kwargs):
        calls.append(kwargs)
        return original

    result = await generate_character_response(GenerationRequest(message='今天整理了书架。',
        character_context=CompiledCharacterContext('', '', '', memory_status='no_match'),
        reply_guard=ReplyGuard(forbid_unsupported_user_fact=True)), model)
    assert result.reply == original and len(calls) == 1
    assert result.guard_fallback == '' and not result.guard_violations


def test_replay_uses_first_raw_reply_not_retry_or_final_fallback():
    row = dict(model_calls=[{'reply': '你喜欢听什么音乐？'}, {'reply': '重试后的内容'}],
               prepared={'reply_guard': {'forbid_unsupported_user_fact': True}},
               generation={'guard_violations': [UNSUPPORTED_USER_FACT], 'reply': '记不清。'})
    replay = replay_trace(row)
    assert replay['changed'] and replay['after'] == ()
    assert replay['reply'] == '你喜欢听什么音乐？'
    assert replay_trace(dict(model_calls=[]))['status'] == 'not_replayable'


def test_replay_uses_effective_retrieval_grounded_guard():
    row = dict(model_calls=[{'reply': '甲与乙是朋友。'}],
               prepared={'reply_guard': {'forbidden_terms': ['甲']}},
               generation={'guard_violations': [],
                           'plan': {'retrieval': {'status': 'ok', 'evidence': '甲与乙'}}})
    assert replay_trace(row)['after'] == ()
    row['generation']['plan']['retrieval']['status'] = 'character_abstention'
    assert replay_trace(row)['after'] == ('unprompted_canonical_identity',)


def test_replay_checks_raw_retry_before_final_sanitization():
    row = dict(model_calls=[{'reply': '你喜欢咖啡。'}, {'reply': '你喜欢读什么书？'}],
               prepared={'reply_guard': {'forbid_unsupported_user_fact': True}},
               generation={'guard_violations': [UNSUPPORTED_USER_FACT], 'guard_retried': True,
                           'guard_post_retry_violations': [UNSUPPORTED_USER_FACT], 'reply': '记不清。'})
    replay = replay_trace(row)
    assert replay['after'] == (UNSUPPORTED_USER_FACT,)
    assert replay['retry_after'] == () and replay['changed']
