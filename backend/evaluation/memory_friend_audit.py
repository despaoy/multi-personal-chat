"""Actual friend-only cold-history retrieval, separate from owner facts."""

import json
import re
from html import unescape


def reported_confirmation_and_validity(reply):
    """Recognize bounded declarative clauses, not a question or missing data."""
    received = False
    valid = False
    uncertain = re.compile(r'没有|尚未|未曾|未能|没|不能|无法|不清楚|不确定|是否|并未|不是|不曾|未$')
    for clause in re.split(r'[，,。！？；;\n]', reply):
        confirmation = re.search(r'收到.{0,64}(?:书面|预约)?确认', clause)
        if confirmation and not uncertain.search(clause[:confirmation.start()]):
            received = True
        validity = re.search(r'仍(?:然)?有效|预约(?:现在|目前)?(?:仍然|仍)?有效|回执(?:现在|目前)?(?:仍然|仍)?有效', clause)
        if validity and not uncertain.search(clause[:validity.start()]):
            valid = True
    return received and valid


def audit_memory_friend_wire(proof, calls, fixture):
    final = proof['generation'][-1]
    answers = [c for c in calls[slice(*final['cloud_call_range'])] if c['request'].get('max_tokens') == 1024]
    if len(answers) != 1:
        return {'one_actual_friend_answer': False}
    messages = answers[0]['request']['messages']
    wire = unescape(messages[-1]['content'])
    match = re.search(r'<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>', wire, re.S)
    data = json.loads(match[1]) if match else {}
    friend = fixture['cases'][-2]['message']
    query = fixture['cases'][-1]['message']
    captured = [row for row in proof.get('sources_before_question', []) if row['body'] == friend]
    delivered = [row for row in data.get('records', []) if row['text'] == friend]
    diag = proof['prepared_diagnostics'][-1]
    cold = proof['cold_history_diagnostics']
    reply = final['response'].get('reply', '')
    knowledge = re.search(r'<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>', wire, re.S)
    return dict(
        one_actual_friend_answer=True,
        complete_friend_source_persisted_before_question=bool(captured),
        complete_friend_source_reached_actual_model=bool(delivered),
        friend_source_identity_and_clock_preserved=bool(delivered) and all(
            any(row['source_id'] == source['source_message_id'] and row['observed_at'] == source['observed_at']
                for source in captured) for row in delivered),
        quoted_source_subject_and_validity_unresolved=data.get('source_kind') == 'historical_user_utterances'
        and data.get('speaker_role') == 'user' and data.get('described_subject') == 'not_resolved'
        and data.get('current_validity') == 'not_resolved',
        friend_source_not_system_rule=all(friend not in m['content'] for m in messages if m['role'] == 'system'),
        source_recall_available=diag['raw_source_status'] == 'available',
        ordinary_history_ablation_declared=proof['controlled_ablation']['history_ablation_enabled']
        and len(cold) == 1 and cold[0]['query'] == query and cold[0]['returned_history'] == [],
        stored_friend_history_not_erased=len(cold) == 1 and any(m['role'] == 'user' and m['content'] == friend
                                                              for m in cold[0]['original_history']),
        no_ordinary_history_delivered=diag['history'] == [] and [m['role'] for m in messages] == ['system', 'user'],
        complete_current_query_delivered=query in wire,
        private_friend_record_outside_public_knowledge=bool(knowledge) and friend not in knowledge[1],
        complete_public_rule_delivered=bool(knowledge) and fixture['documents'][0]['content'] in knowledge[1],
        friend_receipt_recovered=fixture['friend_receipt'] in reply,
        own_receipt_not_reported_as_friend=fixture['own_receipt'] not in reply,
        friend_report_attributed_to_user=bool(re.search(r'转述|按你|根据你|你(?:此前|之前|提供|说过|的记录)', reply)),
        friend_confirmation_and_reported_validity=reported_confirmation_and_validity(reply),
        friend_attendance_absent=bool(re.search(r'(?:未|没有|尚未|还没).{0,10}(?:参加|上课)', reply)),
        friend_departure_absent=bool(re.search(r'(?:未|没有|尚未|还没).{0,10}出发', reply)),
        public_course_number_correct='YY-573-R' in reply,
        ordinary_rain_course_still_runs=bool(re.search(r'(?:普通雨天|普通下雨|雨天).{0,24}(?:照常|正常|不取消|不停课)', reply)),
        only_authoritative_course_cited=[c.get('source_id') for c in final['response'].get('citations', [])] == ['doc_1_chunk_0'],
        exactly_one_new_generation=[g['id'] for g in proof['generation']] == ['friend_only_report'],
        no_prior_writing_or_answers_replayed=proof['reused_native_fixture']['source_writing_generations_replayed'] == 0
        and proof['reused_native_fixture']['prior_answer_generations_replayed'] == 0,
    )
