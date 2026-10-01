"""Owner-only answers under complete, qualified third-party source recall."""

import json
import re
from html import unescape

from evaluation.memory_correction_audit import audit_memory_correction_wire


def audit_owner_sources(proof, messages, fixture):
    wire = unescape(messages[-1]['content'])
    source = re.search(r'<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>', wire, re.S)
    data = json.loads(source[1]) if source else {}
    friend = fixture['cases'][-2]['message']
    receipt = fixture['friend_receipt']
    captured = [row for row in proof.get('sources_before_question', []) if row['body'] == friend]
    delivered = [row for row in data.get('records', []) if row['text'] == friend]
    friend_claims = [row for row in proof['claims_before_question'] if receipt in str(row.get('evidence_json', ''))]
    reply = proof['generation'][-1]['response'].get('reply', '')

    def qualified_quote(row):
        metadata = row.get('metadata') or json.loads(row.get('metadata_json') or '{}')
        return (metadata.get('content_semantics') == 'quoted_source' and metadata.get('speaker_role') == 'user'
                and metadata.get('described_subject') == 'not_resolved')

    return dict(
        raw_source_recall_explicitly_enabled=proof['controlled_ablation'].get('raw_source_recall_enabled') is True,
        friend_complete_source_captured_before_question=bool(captured),
        friend_complete_source_reached_actual_model=bool(delivered),
        friend_source_identity_and_clock_preserved=bool(delivered) and all(
            any(row['source_id'] == original['source_message_id'] and row['observed_at'] == original['observed_at']
                for original in captured) for row in delivered),
        raw_source_qualifications_preserved=data.get('source_kind') == 'historical_user_utterances'
        and data.get('speaker_role') == 'user' and data.get('described_subject') == 'not_resolved'
        and data.get('current_validity') == 'not_resolved',
        friend_not_promoted_to_asserted_user_fact=all(qualified_quote(row) for row in friend_claims),
        friend_source_not_system_rule=all(friend not in m['content'] for m in messages if m['role'] == 'system'),
        own_receipt_recovered=fixture['own_receipt'] in reply,
        friend_receipt_not_reported_as_own=receipt not in reply,
        exactly_two_new_generations=[g['id'] for g in proof['generation']]
        == ['friend_confirmation', 'owner_only_current'],
        no_prior_writing_or_answers_replayed=proof['reused_native_fixture']['source_writing_generations_replayed'] == 0
        and proof['reused_native_fixture']['prior_answer_generations_replayed'] == 0,
    )


def audit_memory_owner_wire(proof, calls, fixture):
    projected = {**fixture, 'cases': fixture['cases'][:2] + fixture['cases'][-1:]}
    checks = audit_memory_correction_wire(proof, calls, projected)
    final = proof['generation'][-1]
    answers = [c for c in calls[slice(*final['cloud_call_range'])] if c['request'].get('max_tokens') == 1024]
    if len(answers) != 1:
        return {'one_actual_owner_answer': False}
    checks.update(audit_owner_sources(proof, answers[0]['request']['messages'], fixture))
    return checks
