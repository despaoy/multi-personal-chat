from character.models import CharacterProfile, DecisionPlan, InteractionState
from character.output_guard import UNSUPPORTED_USER_FACT, build_reply_guard, validate_reply
from evaluation.preference_gate_audit import audit_trace


def trace(reply='你喜欢咖啡。', disabled=True):
    return dict(message='我住在临海。', model_calls=[dict(reply=reply)],
        prepared=dict(reply_guard=dict(forbid_unsupported_user_fact=not disabled),
            compiled=dict(used_memory_ids=['1'], memory_packets=[dict(memory_key='user_location')]),
            history=[dict(role='assistant', content='你喜欢咖啡。')]),
        generation=dict(guard_violations=[], reply='最终回退回复'))


def test_candidate_is_not_automatically_labeled_a_hallucination():
    result = audit_trace(trace())
    assert result['review_candidate'] and result['detector_matches']
    assert result['semantic_verdict'] == 'not_evaluated' and not result['reason_verified']
    assert result['possible_exemption_reasons'] == ['any_admitted_memory']
    assert result['prepared_user_texts'] == ['我住在临海。']
    assert result['first_reply'] == '你喜欢咖啡。'


def test_supported_claim_is_still_only_a_review_candidate():
    row = trace()
    row['message'] = '我喜欢咖啡。'
    result = audit_trace(row)
    assert result['review_candidate'] and result['extracted_user_preferences']
    assert result['semantic_verdict'] == 'not_evaluated'


def test_question_and_enabled_gate_are_not_exemption_candidates():
    assert not audit_trace(trace('你喜欢咖啡吗？'))['review_candidate']
    assert not audit_trace(trace(disabled=False))['review_candidate']


def test_post_generation_writes_cannot_license_the_first_reply():
    row = trace()
    row['claims'] = [dict(content='用户喜欢咖啡')]
    result = audit_trace(row)
    assert not result['extracted_user_preferences']
    assert result['memory_packets'] == [dict(memory_key='user_location')]


def test_missing_or_future_trace_schema_is_reported_not_silently_scored():
    assert audit_trace({})['status'] == 'not_replayable'
    row = trace()
    row['prepared']['reply_guard']['future_field'] = True
    assert audit_trace(row)['status'] == 'incompatible_guard_schema'


def test_prepared_history_is_not_mislabeled_as_final_model_input():
    row = trace()
    row['prepared']['history'].append(dict(role='user', content='旧消息被预算排除'))
    row['model_calls'][0]['request'] = dict(messages=[dict(role='user', content='实际入模正文')])
    result = audit_trace(row)
    assert '旧消息被预算排除' in result['prepared_user_texts']
    assert result['model_user_messages'] == ['实际入模正文']
    row['prepared']['reply_guard'].clear()
    assert audit_trace(row)['status'] == 'not_replayable'


def test_current_global_exemption_reproduction_not_a_desired_contract():
    profile = CharacterProfile(character_id='test', display_name='测试', identity='测试')
    args = (profile, '今天整理了桌面。', (), InteractionState(), DecisionPlan())
    empty = build_reply_guard(*args)
    admitted = build_reply_guard(*args, has_relevant_memory=True)
    assert UNSUPPORTED_USER_FACT in validate_reply('你喜欢咖啡。', empty)
    assert UNSUPPORTED_USER_FACT not in validate_reply('你喜欢咖啡。', admitted)
    # One explicit preference disables checking a different preference too.
    stated = build_reply_guard(profile, '我喜欢画画。', (), InteractionState(), DecisionPlan())
    assert UNSUPPORTED_USER_FACT not in validate_reply('你喜欢咖啡。', stated)
