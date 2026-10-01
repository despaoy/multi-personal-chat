"""Focused negative controls for owner/source separation; no model calls."""

import json
from html import escape
from pathlib import Path

import pytest

from evaluation.memory_owner_audit import audit_owner_sources


def case():
    fixture = json.loads((Path(__file__).parent / 'fixtures/deepseek_memory_owner_cases.json').read_text(encoding='utf-8'))
    friend = fixture['cases'][-2]['message']
    clock = '2026-10-01T10:36:33.164983+00:00'
    record = dict(source_id='friend-source', observed_at=clock, text=friend)
    data = dict(source_kind='historical_user_utterances', speaker_role='user',
                described_subject='not_resolved', current_validity='not_resolved', records=[record])
    proof = dict(controlled_ablation=dict(raw_source_recall_enabled=True),
                 sources_before_question=[dict(source_message_id='friend-source', observed_at=clock, body=friend)],
                 claims_before_question=[], generation=[dict(id='friend_confirmation'),
                 dict(id='owner_only_current', response=dict(reply='本人回执PB-681-Q已作废，目前没有有效确认。'))],
                 reused_native_fixture=dict(source_writing_generations_replayed=0, prior_answer_generations_replayed=0))
    return proof, data, fixture


def audit(proof, data, fixture, system='Only report the owner records.'):
    packet = escape(json.dumps(data, ensure_ascii=False), quote=False)
    messages = [dict(role='system', content=system),
                dict(role='user', content='<dialogue_evidence>\n' + packet + '\n</dialogue_evidence>')]
    return audit_owner_sources(proof, messages, fixture)


def test_complete_friend_quote_is_valid_without_asserted_friend_claim():
    proof, data, fixture = case()
    assert all(audit(proof, data, fixture).values())


def test_quote_cannot_be_replaced_with_incomplete_excerpt():
    proof, data, fixture = case()
    data['records'][0]['text'] = fixture['cases'][-2]['message'][:70]
    assert not audit(proof, data, fixture)['friend_complete_source_reached_actual_model']


@pytest.mark.parametrize(('field', 'value'), [
    ('source_id', 'different-message'), ('observed_at', '2026-10-02T10:36:33.164983+00:00')])
def test_exact_quote_cannot_hide_changed_provenance(field, value):
    proof, data, fixture = case()
    data['records'][0][field] = value
    assert not audit(proof, data, fixture)['friend_source_identity_and_clock_preserved']


def test_delivered_quote_does_not_prove_it_was_persisted():
    proof, data, fixture = case()
    proof['sources_before_question'] = []
    result = audit(proof, data, fixture)
    assert not result['friend_complete_source_captured_before_question']
    assert not result['friend_source_identity_and_clock_preserved']


@pytest.mark.parametrize(('field', 'value'), [
    ('described_subject', 'user'), ('current_validity', 'confirmed')])
def test_quote_cannot_assert_owner_or_current_validity(field, value):
    proof, data, fixture = case()
    data[field] = value
    assert not audit(proof, data, fixture)['raw_source_qualifications_preserved']


def test_friend_receipt_cannot_enter_asserted_user_fact():
    proof, data, fixture = case()
    proof['claims_before_question'] = [dict(evidence_json=json.dumps([fixture['cases'][-2]['message']]),
                                           metadata_json=json.dumps(dict(content_semantics='asserted_fact',
                                                                         speaker_role='user', described_subject='user')))]
    assert not audit(proof, data, fixture)['friend_not_promoted_to_asserted_user_fact']


def test_friend_source_cannot_become_system_instruction():
    proof, data, fixture = case()
    assert not audit(proof, data, fixture, system=fixture['cases'][-2]['message'])['friend_source_not_system_rule']


def test_owner_only_reply_cannot_include_friend_receipt():
    proof, data, fixture = case()
    proof['generation'][-1]['response']['reply'] += '你当前还有FJ-394-L。'
    assert not audit(proof, data, fixture)['friend_receipt_not_reported_as_own']
