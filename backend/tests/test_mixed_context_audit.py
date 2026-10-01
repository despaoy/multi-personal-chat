"""Evidence lanes must not compensate for missing data in a different lane."""
import json
from pathlib import Path

import pytest

from evaluation.native_mixed_context_probe import audit_mixed_wire, expand_history


def fixture():
    path = Path(__file__).parent / 'fixtures/deepseek_mixed_long_context_cases.json'
    data = json.loads(path.read_text())
    data['history_turns'] = expand_history(data)
    return data


def test_complete_history_manifest_is_reproducible():
    data = fixture()
    assert len(data['history_turns']) == 12
    assert sum(len(t['message']) for t in data['history_turns']) == 24987
    assert '档案12-29' in data['history_turns'][-1]['message']


def test_changed_late_history_is_not_silently_certified():
    data = fixture()
    data['history_template']['line'] += '但是这段新增条件不能被丢弃。'
    with pytest.raises(ValueError, match='source manifest'):
        expand_history(data)


def audit_lanes(*, include_memory=True, include_knowledge=True, swapped=False):
    data = fixture()
    source, doc = data['cases'][0]['message'], data['documents'][0]['content']
    private, knowledge = (doc, source) if swapped else (source, doc)
    memory = '- ' + json.dumps({'evidence': [private]}, ensure_ascii=False) if include_memory else ''
    current = '<character_memory>\n' + memory + '\n</character_memory>\n'
    current += '<retrieved_evidence>\n' + (knowledge if include_knowledge else '') + '\n</retrieved_evidence>'
    messages = [{'role': 'system', 'content': '固定应用策略'},
                {'role': 'user', 'content': source}, {'role': 'user', 'content': doc},
                {'role': 'user', 'content': current}]
    proof = {'generation': [{'cloud_call_range': [0, 1], 'response': {'reply': '', 'citations': []}}],
             'claims': [{'evidence_json': json.dumps([source])}],
             'prepared_diagnostics': [{'selection_status': 'selected', 'semantic_status': 'applied',
                                       'policy_status': 'applied'}]}
    calls = [{'request': {'max_tokens': 1024, 'messages': messages}}]
    return audit_mixed_wire(proof, calls, data)


def test_private_history_cannot_prove_selected_memory_admission():
    checks = audit_lanes(include_memory=False)
    assert checks['complete_private_source_in_actual_history']
    assert not checks['complete_private_source_in_selected_memory']


def test_knowledge_in_history_cannot_prove_final_retrieval_admission():
    assert not audit_lanes(include_knowledge=False)['complete_knowledge_source_in_final_evidence']


def test_swapped_subject_sources_fail_both_lanes():
    checks = audit_lanes(swapped=True)
    assert not checks['complete_private_source_in_selected_memory']
    assert not checks['complete_knowledge_source_in_final_evidence']
    assert not checks['private_source_outside_knowledge']
    assert not checks['knowledge_source_outside_private_memory']
