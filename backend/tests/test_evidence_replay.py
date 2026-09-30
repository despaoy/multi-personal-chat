import pytest

from evaluation.replay_evidence_ablation import (
    annotated_evidence_messages,
    card_evidence_messages,
    replay_parameters,
    speaker_evidence_messages,
    variants,
)


def test_speaker_ablation_keeps_history_raw_text_and_non_card_context():
    from html import escape

    raw = '[甲] 「你好。」\n我没有回答。\n[乙] 「未完成'
    card = '【卡片关联证据】问候\n索引标注：{}\n' + raw
    background = '【父场景】保持原样'
    evidence = card + '\n\n' + background
    retrieval = dict(evidence=evidence, documents=[{'id': 'a', 'content': '摘要\n证据：' + raw}],
        evidence_packets=[{'text': card, 'document_ids': ['a']}, {'text': background, 'document_ids': []}])
    wrapped = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    messages = [{'role': 'system', 'content': '人物'}, {'role': 'assistant', 'content': '之前'},
        {'role': 'user', 'content': wrapped + escape(evidence) + '\n</retrieved_evidence>\n问题'}]
    changed = speaker_evidence_messages(messages, retrieval)
    assert changed[:-1] == messages[:-1]
    assert '【台词】[甲] 「你好。」' in changed[-1]['content']
    assert '【叙述】我没有回答。' in changed[-1]['content']
    assert '【未分类】[乙] 「未完成' in changed[-1]['content']
    restored = changed[-1]['content']
    for marker in ('【台词】', '【叙述】', '【未分类】'):
        restored = restored.replace(marker, '')
    assert restored == messages[-1]['content']
    with pytest.raises(ValueError, match='suffix'):
        speaker_evidence_messages(messages, {**retrieval, 'documents': [{'id': 'a', 'content': '摘要\n证据：不同'}]})


def test_replay_keeps_actual_source_current_query_system_and_user_topic_unchanged():
    original = [{'role': 'system', 'content': '角色规则'},
                {'role': 'user', 'content': '询问关系'},
                {'role': 'assistant', 'content': '待核实说法'},
                {'role': 'user', 'content': '原文证据和当前问题'}]
    alternatives = variants(original)
    assert alternatives['full_history'] == original
    reduced = alternatives['without_assistant_history']
    assert reduced == [original[0], original[1], original[3]]
    assert len(original) == 4


@pytest.mark.parametrize('messages', [[], [{'role': 'assistant', 'content': '答复'}],
                                      [{'role': 'user', 'content': '无历史'}]])
def test_replay_rejects_non_ablatable_inputs(messages):
    with pytest.raises(ValueError):
        variants(messages)


def test_actual_generation_parameters_override_plan_and_missing_are_disclosed():
    plan = dict(temperature=0.5, max_tokens=2048, top_p=0.9,
                repetition_penalty=1.0, frequency_penalty=0.0, enable_thinking=True)
    values, assumed = replay_parameters({'parameters': {'temperature': 0.7, 'enable_thinking': False}}, plan)
    assert values['temperature'] == 0.7 and values['enable_thinking'] is False
    assert set(assumed) == {'max_tokens', 'top_p', 'repetition_penalty', 'frequency_penalty'}
    assert plan['temperature'] == 0.5


def test_card_view_preserves_prompt_history_query_and_all_evidence():
    messages = [{'role': 'system', 'content': '角色规则'},
                {'role': 'assistant', 'content': '历史'},
                {'role': 'user', 'content': '<retrieved_evidence trust="untrusted">\n摘要\n</retrieved_evidence>\n问题'}]
    docs = [{'title': '关系', 'content': '摘要\n证据：原文一\n但有限定'}]
    changed = card_evidence_messages(messages, docs)
    assert changed[:2] == messages[:2]
    assert changed[-1]['content'].endswith('</retrieved_evidence>\n问题')
    assert '原文一\n但有限定' in changed[-1]['content']
    assert '摘要' in messages[-1]['content']
    with pytest.raises(ValueError, match='budget'):
        card_evidence_messages(messages, docs, max_chars=2)
    with pytest.raises(ValueError, match='own evidence'):
        card_evidence_messages(messages, [{'content': '摘要'}])


def test_annotation_ablation_changes_only_evidence_metadata():
    text = '【卡片关联证据】事实\n原文。\n但尚未发生。'
    wrapped = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    messages = [{'role': 'system', 'content': '固定人物设定'},
                {'role': 'assistant', 'content': '原回复'},
                {'role': 'user', 'content': wrapped + text + '\n</retrieved_evidence>\n当前问题'}]
    retrieval = {'evidence': text, 'evidence_packets': [{'text': text, 'document_ids': ['a']}],
                 'documents': [{'id': 'a', 'title': '事实', 'metadata': {'subject': '人物甲'}}]}
    changed = annotated_evidence_messages(messages, retrieval)
    assert changed[:2] == messages[:2]
    assert changed[-1]['content'].endswith('原文。\n但尚未发生。\n</retrieved_evidence>\n当前问题')
    assert '人物甲' in changed[-1]['content'] and '人物甲' not in messages[-1]['content']
    with pytest.raises(ValueError, match='reconstruct'):
        annotated_evidence_messages(messages, {**retrieval, 'evidence': 'other'})


def test_background_ablation_keeps_complete_cards_and_all_other_messages():
    from html import escape

    from evaluation.replay_evidence_ablation import without_background_messages

    packets = [{'kind': 'evidence', 'text': '卡片原文 <甲>\n不过尚未发生'},
               {'kind': 'background', 'text': '父场景开头…'},
               {'kind': 'evidence', 'text': '第二条完整原文'}]
    evidence = '\n\n'.join(p['text'] for p in packets)
    prefix = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    messages = [{'role': 'system', 'content': '固定设定'}, {'role': 'user', 'content': '历史'},
                {'role': 'user', 'content': prefix + escape(evidence, quote=False) + '\n</retrieved_evidence>\n当前问题'}]
    retrieval = {'evidence': evidence, 'evidence_packets': packets}
    changed = without_background_messages(messages, retrieval)
    assert changed[:-1] == messages[:-1]
    assert changed[-1]['content'] == prefix + escape(packets[0]['text'] + '\n\n' + packets[2]['text'], quote=False) + '\n</retrieved_evidence>\n当前问题'
    assert '父场景开头' in messages[-1]['content']
    with pytest.raises(ValueError, match='reconstruct'):
        without_background_messages(messages, {**retrieval, 'evidence': 'wrong'})
    with pytest.raises(ValueError, match='both'):
        without_background_messages(messages, {'evidence': 'only', 'evidence_packets': [{'kind': 'evidence', 'text': 'only'}]})


def test_task_ablation_changes_only_bound_terminal_user_query():
    from evaluation.replay_evidence_ablation import full_task_messages

    messages = [{'role': 'system', 'content': '固定设定'}, {'role': 'user', 'content': '历史'},
                {'role': 'user', 'content': '不变的记忆与证据\n<user_query>\n林远是谁\n</user_query>'}]
    trace = {'message': '我的专业是什么，林远是谁？', 'generation': {'response_mode': 'task_composite',
             'plan': {'retrieval': {'identity_task': {'query': '林远是谁'}}}}}
    changed = full_task_messages(messages, trace)
    assert changed[:-1] == messages[:-1]
    assert changed[-1]['content'] == '不变的记忆与证据\n<user_query>\n我的专业是什么，林远是谁？\n</user_query>'
    assert messages[-1]['content'].endswith('林远是谁\n</user_query>')
    with pytest.raises(ValueError, match='residual'):
        full_task_messages(messages[:-1], trace)


def test_selection_comparison_is_same_query_complete_evidence_only():
    from copy import deepcopy

    from evaluation.replay_evidence_ablation import selection_evidence_messages

    prefix = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    messages = [{'role': 'system', 'content': '设定'}, {'role': 'user', 'content': '历史'},
                {'role': 'user', 'content': '记忆\n' + prefix + '旧资料\n</retrieved_evidence>\n当前问题'}]
    trace = {'message': '当前问题', 'generation': {'plan': {'retrieval': {
        'evidence': '旧资料', 'evidence_packets': [{'text': '旧资料'}]}}}}
    audit = {'query': '当前问题', 'result': {'context_text': '新资料<甲>',
             'evidence_packets': [{'text': '新资料<甲>', 'document_ids': ['a']}], 'citations': [{'id': 'a'}]}}
    changed = selection_evidence_messages(messages, trace, audit)
    assert changed[:-1] == messages[:-1]
    assert changed[-1]['content'] == messages[-1]['content'].replace('旧资料', '新资料&lt;甲&gt;')
    for variant, error in [('query', 'query'), ('packet', 'reconstruct'), ('citation', 'citations')]:
        invalid = deepcopy(audit)
        if variant == 'query':
            invalid['query'] = '另一个问题'
        elif variant == 'packet':
            invalid['result']['context_text'] = '被裁剪'
        else:
            invalid['result']['citations'] = []
        with pytest.raises(ValueError, match=error):
            selection_evidence_messages(messages, trace, invalid)


def test_probe_selection_cannot_silently_replay_ordinary_turn():
    from evaluation.replay_evidence_ablation import select_trace

    ordinary = {'case_id': 'a', 'turn': 1}
    probe = {'case_id': 'a', 'mode': 'memory_only'}
    rows = [ordinary, probe]
    assert select_trace(rows, 'a', turn=1) is ordinary
    assert select_trace(rows, 'a', memory_only=True) is probe
    for kwargs in ({}, {'turn': 1, 'memory_only': True}, {'turn': 0}):
        with pytest.raises(ValueError):
            select_trace(rows, 'a', **kwargs)
    with pytest.raises(ValueError, match='exactly one'):
        select_trace(rows + [probe], 'a', memory_only=True)


@pytest.mark.parametrize('query,expected', [
    ('先说我的专业，再告诉我林远是谁。', '先说当前对话用户的专业，再告诉我林远是谁。'),
    ('我叫阿青。我的名字是什么？', '我叫阿青。当前对话用户的名字是什么？'),
    ('你还记得我的工作地点吗？', '你还记得当前对话用户的工作地点吗？'),
    ('我来自哪里？', '当前对话用户来自哪里？'),
    ('我现在住哪里？', '当前对话用户现在住哪里？'),
    ('我的名字和专业分别是什么？', '当前对话用户的名字和专业分别是什么？'),
    ('先说我的专业；再说我的名字。', '先说当前对话用户的专业；再说当前对话用户的名字。'),
])
def test_owner_binding_only_changes_closed_lookup_spans(query, expected):
    from evaluation.replay_evidence_ablation import owner_bound_query

    assert owner_bound_query(query) == expected


@pytest.mark.parametrize('query', ['我的朋友的专业是什么？', '请翻译“我的专业是什么？”',
                                    '我的专业是数学。', '假如我的专业是数学呢？',
                                    '你叫什么？', '我的专业适合什么工作？'])
def test_owner_binding_leaves_data_other_people_and_open_requests_untouched(query):
    from evaluation.replay_evidence_ablation import owner_bound_query

    assert owner_bound_query(query) == query


def test_owner_ablation_keeps_all_evidence_history_and_task_order():
    from evaluation.replay_evidence_ablation import owner_bound_messages

    query = '先说我的专业，再告诉我林远是谁。'
    messages = [{'role': 'system', 'content': '设定'}, {'role': 'user', 'content': '历史'},
                {'role': 'user', 'content': '记忆及证据\n<user_query>\n' + query + '\n</user_query>'}]
    changed = owner_bound_messages(messages, {'message': query})
    assert changed[:-1] == messages[:-1]
    assert changed[-1]['content'].startswith('记忆及证据\n')
    assert changed[-1]['content'].endswith('再告诉我林远是谁。\n</user_query>')
    assert query in messages[-1]['content']


def test_native_message_boundary_preserves_raw_query_and_untrusted_evidence():
    from html import escape

    from evaluation.replay_evidence_ablation import separated_query_messages

    query = '先说我的专业，再比较 <甲> 与 <乙>。'
    reference = '<retrieved_evidence trust="untrusted">\n引文\n</retrieved_evidence>\n'
    messages = [{'role': 'system', 'content': '设定'}, {'role': 'assistant', 'content': '历史'},
                {'role': 'user', 'content': reference + '<user_query>\n' + escape(query) + '\n</user_query>'}]
    changed = separated_query_messages(messages, {'message': query})
    assert changed[:-2] == messages[:-1]
    assert changed[-2] == {'role': 'user', 'content': reference}
    assert changed[-1] == {'role': 'user', 'content': query}
    with pytest.raises(ValueError, match='exact terminal'):
        separated_query_messages(messages, {'message': '另外的问题'})


def test_persona_ablation_keeps_runtime_boundaries_and_all_data():
    from evaluation.replay_evidence_ablation import persona_variants

    directive = '- 始终以第一人称和人物甲身份自然交流。\n'
    system = '【人物身份】\n姓名：人物甲\n【人物行为边界】\n' + directive + '保留其他画像。\n【当前关系】\n保留安全与记忆规则。'
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': '历史'},
                {'role': 'user', 'content': '数据与问题'}]
    views = persona_variants(messages)
    assert views['without_first_person_directive'][0]['content'] == system.replace(directive, '')
    assert views['without_profile'][0]['content'] == '【当前关系】\n保留安全与记忆规则。'
    for view in views.values():
        assert view[1:] == messages[1:]
    assert messages[0]['content'] == system
    with pytest.raises(ValueError, match='complete structured persona'):
        persona_variants([{'role': 'system', 'content': '自定义画像'}])
