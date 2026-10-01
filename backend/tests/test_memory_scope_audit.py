"""Every native model request must respect the authenticated private scope."""

import pytest

from evaluation.memory_scope_audit import forbidden_private_payloads


def call(*messages):
    return {'request': {'messages': list(messages)}}


def test_scoped_public_evidence_is_not_confused_with_private_receipts():
    assert forbidden_private_payloads([call({'role': 'user', 'content': '公开编号YY-573-R。当前账户无私人原话。'})], ['FJ-394-L', 'PB-681-Q'])


@pytest.mark.parametrize('role', ['system', 'user', 'assistant'])
def test_private_receipt_leak_is_rejected_in_every_message_role(role):
    assert not forbidden_private_payloads([call({'role': role, 'content': '朋友私人回执FJ-394-L。'})], ['FJ-394-L'])


def test_secondary_writer_or_selector_call_cannot_hide_leak_from_answer_only_audit():
    calls = [call({'role': 'user', 'content': '当前问题'}), call({'role': 'user', 'content': '{"candidates":["本人PB-681-Q"]}'})]
    assert not forbidden_private_payloads(calls, ['PB-681-Q'])


def test_html_escaped_model_input_is_checked_after_unescaping():
    assert not forbidden_private_payloads([call({'role': 'user', 'content': 'FJ&#45;394&#45;L'})], ['FJ-394-L'])


def test_complete_other_owner_quote_is_rejected_without_a_receipt():
    source = '朋友预约有效，尚未参加，未出发。'
    assert not forbidden_private_payloads([call({'role': 'user', 'content': source})], [source])
