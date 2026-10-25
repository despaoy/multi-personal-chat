"""Replay diagnostics use raw attempts and the effective retrieval guard."""
from character.output_guard import UNSUPPORTED_USER_FACT
from evaluation.replay_reply_guard import replay_trace


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
