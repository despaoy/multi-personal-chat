"""Only new audit-tool cases; these are not actual model test results."""

import copy
import json
import unittest

from evaluation.memory_correction_audit import audit_memory_correction_wire


def complete_saved_example():
    first, correction, question = '合成先前预约来源。', '合成后来撤销来源。', '合成当前状态问题。'
    document = '合成完整公开课程规则。'
    metadata = dict(content_semantics='quoted_source', speaker_role='user', described_subject='not_resolved')
    observation_time = '2026-10-01T12:00:00+00:00'
    packet = json.dumps(dict(id='correction-claim', evidence=[correction], source_message_ids=['correction-message'],
                             content_semantics='quoted_source', speaker_role='user', subject_scope='not_resolved',
                             temporal_mode='observation', observed_at=observation_time, valid_from=''), ensure_ascii=False)
    wire = ('<character_memory>\n- ' + packet + '\n</character_memory>\n'
            '<retrieved_evidence>\n' + document + '\n</retrieved_evidence>\n' + question)
    reply = ('当前没有有效的书面预约确认。私人回执PB-681-Q已作废。尚未参加，也未出发。'
             '公开编号YY-573-R，普通雨天照常举办。')
    fixture = {'cases': [{'message': s} for s in (first, correction, question)], 'documents': [{'content': document}]}
    proof = dict(
        generation=[dict(cloud_call_range=[0, 1], response=dict(reply=reply, citations=[{'source_id': 'doc_1_chunk_0'}]))],
        claims_before_question=[{'id': claim_id, 'evidence_json': json.dumps([source]), 'metadata_json': json.dumps(metadata),
                                 'source_message_ids_json': json.dumps(['correction-message']),
                                 'temporal_mode': 'observation', 'temporal_observed_at': observation_time}
                                for claim_id, source in [('initial-claim', first), ('correction-claim', correction)]],
        cold_history_diagnostics=[dict(query=question, returned_history=[], original_history=[
            {'role': 'user', 'content': source} for source in (first, correction)])],
        candidate_diagnostics=[dict(query=question, items=[{'memory_id': 'correction-claim', 'evidence': [correction]}])],
        controlled_ablation={'history_ablation_enabled': True},
        prepared_diagnostics=[dict(history=[], used_memory_ids=['correction-claim'], selection_status='selected', semantic_status='applied', policy_status='applied')],
    )
    calls = [dict(request=dict(max_tokens=1024, messages=[{'role': 'system', 'content': '应用规则'}, {'role': 'user', 'content': wire}]))]
    return proof, calls, fixture


class CorrectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.proof, self.calls, self.fixture = copy.deepcopy(complete_saved_example())

    def checks(self):
        return audit_memory_correction_wire(self.proof, self.calls, self.fixture)

    def test_complete_declared_example_is_accepted(self):
        self.assertTrue(all(self.checks().values()))

    def test_ordinary_history_cannot_rescue_missing_memory(self):
        wire = self.calls[0]['request']['messages'][-1]['content']
        self.calls[0]['request']['messages'][-1]['content'] = wire.replace('合成后来撤销来源。', '')
        self.assertFalse(self.checks()['complete_correction_admitted_as_private_memory'])
        self.assertTrue(self.checks()['stored_history_not_erased'])

    def test_previous_source_does_not_substitute_for_correction(self):
        self.calls[0]['request']['messages'][-1]['content'] = self.calls[0]['request']['messages'][-1]['content'].replace('合成后来撤销来源。', '合成先前预约来源。')
        self.assertFalse(self.checks()['complete_correction_admitted_as_private_memory'])

    def test_missing_candidate_is_not_claimed_as_recall_success(self):
        self.proof['candidate_diagnostics'][0]['items'] = []
        self.assertFalse(self.checks()['complete_correction_recalled_as_candidate'])

    def test_ablation_must_be_explicit(self):
        del self.proof['controlled_ablation']
        self.assertFalse(self.checks()['ordinary_history_ablation_declared'])

    def test_stored_sources_must_exist_before_question(self):
        self.proof['claims_before_question'] = []
        self.assertFalse(self.checks()['complete_correction_persisted_before_question'])

    def test_stale_confirmed_answer_is_rejected(self):
        self.proof['generation'][0]['response']['reply'] = '当前预约仍然有效，已经出发并参加课程。'
        checks = self.checks()
        self.assertFalse(checks['current_confirmation_negated'])
        self.assertFalse(checks['attendance_still_absent'])
        self.assertFalse(checks['departure_still_absent'])

    def test_public_rule_in_system_is_rejected(self):
        self.calls[0]['request']['messages'][0]['content'] += self.fixture['documents'][0]['content']
        self.assertFalse(self.checks()['sources_not_system_rules'])

    def alter_packet(self, **changes):
        wire = self.calls[0]['request']['messages'][-1]['content']
        line = next(line for line in wire.splitlines() if line.startswith('- {'))
        packet = json.loads(line[2:])
        packet.update(changes)
        self.calls[0]['request']['messages'][-1]['content'] = wire.replace(line, '- ' + json.dumps(packet, ensure_ascii=False))

    def test_different_id_cannot_borrow_complete_source_text(self):
        self.alter_packet(id='different-claim')
        self.assertFalse(self.checks()['admitted_correction_links_to_persisted_and_recalled_id'])

    def test_missing_selected_id_is_rejected(self):
        self.proof['prepared_diagnostics'][0]['used_memory_ids'] = []
        self.assertFalse(self.checks()['admitted_correction_links_to_persisted_and_recalled_id'])

    def test_missing_source_message_id_is_rejected(self):
        self.alter_packet(source_message_ids=[])
        self.assertFalse(self.checks()['admitted_correction_retains_original_source_ids'])

    def test_user_quote_cannot_be_promoted_to_verified_fact(self):
        self.alter_packet(content_semantics='asserted_fact', subject_scope='verified_user')
        self.assertFalse(self.checks()['quoted_observation_and_clock_preserved'])

    def test_observation_clock_cannot_be_manufactured(self):
        self.alter_packet(temporal_mode='fact', valid_from='2099-01-01')
        self.assertFalse(self.checks()['quoted_observation_and_clock_preserved'])

    def test_missing_update_is_not_absence_of_valid_confirmation(self):
        self.proof['generation'][0]['response']['reply'] = '我没有找到你之后更新的记录，无法核对现在预约是否仍然有效。'
        self.assertFalse(self.checks()['current_confirmation_negated'])

    def test_historical_absence_does_not_determine_current_confirmation(self):
        self.proof['generation'][0]['response']['reply'] = '以前没有有效的书面预约确认，目前已经确认成功。'
        self.assertFalse(self.checks()['current_confirmation_negated'])

    def test_native_not_needed_review_is_legitimate(self):
        self.proof['prepared_diagnostics'][0]['semantic_status'] = 'not_needed'
        self.assertTrue(self.checks()['semantic_review_status_valid'])

    def test_review_error_is_not_native_success(self):
        self.proof['prepared_diagnostics'][0]['semantic_status'] = 'timeout'
        self.assertFalse(self.checks()['semantic_review_status_valid'])

    def test_raw_storage_observation_clock_is_retained(self):
        row = self.proof['claims_before_question'][1]
        row['observed_at'] = row.pop('temporal_observed_at')
        del row['temporal_mode']
        self.assertTrue(self.checks()['quoted_observation_and_clock_preserved'])
        self.alter_packet(observed_at='2099-01-01T00:00:00+00:00')
        self.assertFalse(self.checks()['quoted_observation_and_clock_preserved'])

    def test_no_real_answer_record_is_rejected(self):
        self.assertFalse(all(audit_memory_correction_wire(self.proof, [], self.fixture).values()))


if __name__ == '__main__':
    unittest.main()
