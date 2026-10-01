"""Saved-wire audit of a complete correction under declared cold history."""

import json
import re
from html import unescape


def confirmation_negated(reply):
    """Require an explicit confirmation status, not inability to find records."""
    negative = re.compile(r'(?:没有|无|不再(?:有|持有)?|未持有)(?:任何|一份)?(?:有效的?)?(?:书面)?(?:预约)?确认')
    for clause in re.split(r'[，,。！？；;\n]', reply):
        match = negative.search(clause)
        if match is None:
            continue
        prefix = clause[:match.start()]
        if re.search(r'以前|当时|先前|曾经', prefix) and not re.search(r'目前|现在|当前|现已', prefix):
            continue
        return True
    return False


def audit_memory_correction_wire(proof, calls, fixture):
    first, correction, question = (c['message'] for c in fixture['cases'])
    final = proof['generation'][-1]
    answers = [c for c in calls[slice(*final['cloud_call_range'])]
               if c['request'].get('max_tokens') == 1024]
    if len(answers) != 1:
        return {'one_actual_cold_correction_answer': False}
    messages = answers[0]['request']['messages']
    wire = unescape(messages[-1]['content'])
    memory = re.search(r'<character_memory[^>]*>\n(.*?)\n</character_memory>', wire, re.S)
    knowledge = re.search(r'<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>', wire, re.S)
    packets = [json.loads(line[2:]) for line in memory[1].splitlines() if line.startswith('- {')] if memory else []
    corrected_packets = [p for p in packets if correction in p.get('evidence', [])]
    before = proof.get('claims_before_question', [])
    cold = proof.get('cold_history_diagnostics', [])
    recalled = [d for d in proof.get('candidate_diagnostics', []) if d['query'] == question]
    latest_recalled = bool(recalled) and any(correction in item.get('evidence', []) for item in recalled[-1]['items'])
    persisted_corrections = {
        str(row.get('id', '')): row for row in before
        if correction in json.loads(row.get('evidence_json') or '[]') and row.get('id')
    }
    candidate_ids = {str(item.get('memory_id', '')) for d in recalled for item in d['items']
                     if correction in item.get('evidence', [])}

    def source_ids(row):
        values = row.get('source_message_ids') or json.loads(row.get('source_message_ids_json') or '[]')
        return {str(value) for value in values if value}

    def retains_observation(packet):
        row = persisted_corrections.get(str(packet.get('id', '')))
        if row is None:
            return False
        metadata = row.get('metadata') or json.loads(row.get('metadata_json') or '{}')
        if (metadata.get('content_semantics') == 'quoted_source'
                and not (packet.get('content_semantics') == 'quoted_source'
                         and packet.get('speaker_role') == metadata.get('speaker_role') == 'user'
                         and packet.get('subject_scope') == metadata.get('described_subject') == 'not_resolved')):
            return False
        if row.get('temporal_mode') == 'observation' or metadata.get('content_semantics') == 'quoted_source':
            clock = row.get('temporal_observed_at') or row.get('observed_at') or metadata.get('temporal_provenance', {}).get('observed_at')
            return (packet.get('temporal_mode') == 'observation' and not packet.get('valid_from')
                    and bool(clock) and packet.get('observed_at') == clock)
        return True

    diag = proof['prepared_diagnostics'][-1]
    reply = final['response'].get('reply', '')
    doc = fixture['documents'][0]

    def persisted(source):
        return any(source in json.loads(row.get('evidence_json') or '[]') for row in before)

    return dict(
        one_actual_cold_correction_answer=True,
        complete_initial_source_persisted_before_question=persisted(first),
        complete_correction_persisted_before_question=persisted(correction),
        ordinary_history_ablation_declared=proof.get('controlled_ablation', {}).get('history_ablation_enabled', False)
        and len(cold) == 1 and cold[0]['query'] == question and cold[0]['returned_history'] == [],
        stored_history_not_erased=len(cold) == 1 and all(any(m['role'] == 'user' and m['content'] == source
                                                        for m in cold[0]['original_history']) for source in (first, correction)),
        no_ordinary_history_delivered=diag['history'] == [] and len(messages) == 2
        and [m['role'] for m in messages] == ['system', 'user'],
        complete_correction_recalled_as_candidate=latest_recalled,
        complete_correction_admitted_as_private_memory=bool(corrected_packets),
        admitted_correction_links_to_persisted_and_recalled_id=bool(corrected_packets) and all(
            str(p.get('id', '')) in persisted_corrections and str(p.get('id', '')) in candidate_ids
            and str(p.get('id', '')) in diag['used_memory_ids'] for p in corrected_packets),
        admitted_correction_retains_original_source_ids=bool(corrected_packets) and all(
            bool(source_ids(persisted_corrections.get(str(p.get('id', '')), {})))
            and source_ids(persisted_corrections[str(p.get('id', ''))]).issubset(set(p.get('source_message_ids', [])))
            for p in corrected_packets),
        quoted_observation_and_clock_preserved=bool(corrected_packets) and all(retains_observation(p) for p in corrected_packets),
        complete_current_query_delivered=question in wire,
        complete_public_rule_delivered=bool(knowledge) and doc['content'] in knowledge[1],
        private_correction_outside_public_knowledge=bool(knowledge) and correction not in knowledge[1],
        public_rule_outside_private_memory=bool(memory) and doc['content'] not in memory[1],
        sources_not_system_rules=all(first not in m['content'] and correction not in m['content']
                                     and doc['content'] not in m['content'] for m in messages if m['role'] == 'system'),
        memory_selection_succeeded=diag['selection_status'] == 'selected',
        semantic_review_status_valid=diag['semantic_status'] in {'applied', 'not_needed'},
        contextual_policy_succeeded=diag['policy_status'] == 'applied',
        current_confirmation_negated=confirmation_negated(reply),
        private_receipt_voided='PB-681-Q' in reply and bool(re.search(r'作废|失效|无效|撤销', reply)),
        attendance_still_absent=bool(re.search(r'(?:未|没有|尚未|还没|未曾|不曾).{0,10}(?:参加|上课)', reply)),
        departure_still_absent=bool(re.search(r'(?:未|没有|尚未|还没).{0,10}出发', reply)),
        public_course_number_correct='YY-573-R' in reply,
        ordinary_rain_course_still_runs=bool(re.search(r'(?:普通雨天|普通下雨|雨天).{0,24}(?:照常|正常|不取消|不停课)', reply)),
        only_authoritative_course_cited=[c.get('source_id') for c in final['response'].get('citations', [])] == ['doc_1_chunk_0'],
    )
