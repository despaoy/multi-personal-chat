from evaluation.deepseek_frozen_replay import inspect_guard, payload


def test_cloud_payload_keeps_input_and_removes_local_only_settings():
    messages = [dict(role='system', content='人物设定'), dict(role='user', content='原话与问题')]
    request = dict(messages=messages, max_tokens=256, temperature=.2, top_p=.9,
        frequency_penalty=0, repetition_penalty=1.0, lora_name='local', enable_thinking=False)
    result = payload(request, 'test-model')
    assert result['messages'] == messages and result['max_tokens'] == 256
    assert not {'lora_name', 'repetition_penalty', 'chat_template_kwargs'} & result.keys()
    assert result['thinking'] == {'type': 'disabled'}


def test_preference_heuristic_is_diagnostic_in_default_cloud_comparison():
    row = dict(prepared=dict(reply_guard=dict(forbid_unsupported_user_fact=True),
        compiled=dict(memory_status='no_match')), generation=dict(plan=dict(retrieval={})))
    result = inspect_guard(row, '你喜欢咖啡。')
    assert result['action'] == 'pass' and result['violations'] == ('unsupported_user_fact',)
    row['prepared']['compiled']['memory_status'] = 'available'
    result = inspect_guard(row, '你喜欢咖啡。')
    assert result['action'] == 'pass' and result['final_if_no_retry'] == '你喜欢咖啡。'
    result = inspect_guard(row, '你喜欢咖啡吗？')
    assert result['action'] == 'pass'
