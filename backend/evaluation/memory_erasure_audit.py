"""Committed native source-only erasure and cold post-erasure read."""

import json
import re
from html import unescape


def _payload(text):
    if not isinstance(text, str):
        return {}
    try:
        value = json.loads(text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def audit_erasure_archive(sql, original_history, receipt):
    return dict(
        chat_archive_quote_physically_preserved=sql.get('archive_quote_row_count') == 1,
        revoked_receipt_filtered_from_regular_history=all(receipt not in m['content'] for m in original_history),
    )


def audit_memory_erasure_wire(proof, calls, fixture):
    friend = fixture['cases'][2]['message']
    erase, query = (c['message'] for c in fixture['cases'][-2:])
    original = [row for row in proof.get('sources_before_erasure', []) if row['body'] == friend]
    expected_ids = {row['source_message_id'] for row in original}
    generations = proof['generation']
    erase_calls = calls[slice(*generations[0]['cloud_call_range'])]
    writers = []
    for call in erase_calls:
        for message in call['request']['messages']:
            payload = _payload(message['content'])
            if isinstance(payload, dict) and 'source_erasure_candidates' in payload:
                writers.append((call, payload))
    writer, request = writers[0] if len(writers) == 1 else ({}, {})
    response = _payload(writer.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content', ''))
    candidates = request.get('source_erasure_candidates', [])
    targeted = set(response.get('erase_source_ids', []))
    receipts = proof.get('operation_diagnostics', [])
    receipt = receipts[0].get('receipt', {}) if len(receipts) == 1 else {}
    after = proof.get('sources_before_question', [])
    sql = proof.get('source_erasure_sql', {})
    final_calls = calls[slice(*generations[-1]['cloud_call_range'])]
    answers = [c for c in final_calls if c['request'].get('max_tokens') == 1024]
    if len(answers) != 1:
        return {'one_actual_post_erasure_answer': False}
    messages = answers[0]['request']['messages']
    wire = unescape(messages[-1]['content'])
    knowledge = re.search(r'<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>', wire, re.S)
    reply = generations[-1]['response'].get('reply', '')
    cold = proof['cold_history_diagnostics']
    diag = proof['prepared_diagnostics'][-1]
    return dict(
        one_actual_post_erasure_answer=True,
        complete_friend_source_present_before_erase=len(original) == 1,
        erase_request_does_not_reintroduce_receipt=fixture['friend_receipt'] not in erase,
        post_erase_query_does_not_reintroduce_receipt=fixture['friend_receipt'] not in query,
        exactly_one_actual_source_erase_writer=len(writers) == 1,
        full_friend_candidate_reached_actual_writer=any(row['source_id'] in expected_ids and row['text'] == friend for row in candidates),
        actual_writer_targets_only_friend_source=bool(expected_ids) and targeted == expected_ids,
        actual_writer_received_complete_erase_request=request.get('current_user_message') == erase,
        committed_receipt_confirms_one_source_erased=receipt.get('status') == 'erased'
        and receipt.get('source_erased') == 1 and receipt.get('persisted') == 1,
        source_removed_from_current_read=not any(row['source_message_id'] in expected_ids for row in after),
        sql_source_retained_only_as_revoked_anchor=sql.get('state') == 'revoked' and sql.get('body_is_null') is True
        and sql.get('observed_at_is_null') is True,
        erased_source_index_empty=sql.get('term_count') == 0,
        erased_source_has_no_claim_links=sql.get('link_count') == 0,
        unrelated_sources_preserved=all(row in after for row in proof.get('sources_before_erasure', []) if row['source_message_id'] not in expected_ids),
        own_claims_unchanged_after_erase=proof['claims_before_question'] == proof['claims_before_erasure'],
        post_query_did_not_change_own_claims=proof['claims'] == proof['claims_before_erasure'],
        ordinary_cold_read_declared=proof['controlled_ablation']['history_ablation_enabled']
        and len(cold) == 1 and cold[0]['query'] == query and cold[0]['returned_history'] == [],
        **audit_erasure_archive(sql, cold[0]['original_history'] if len(cold) == 1 else [], fixture['friend_receipt']),
        no_ordinary_history_delivered=diag['history'] == [] and [m['role'] for m in messages] == ['system', 'user'],
        source_read_executed_after_erase=diag['raw_source_status'] in {'available', 'no_match'},
        erased_receipt_not_on_actual_answer_wire=all(fixture['friend_receipt'] not in m['content'] for m in messages),
        erased_receipt_not_in_reply=fixture['friend_receipt'] not in reply,
        own_receipt_not_substituted_for_friend=fixture['own_receipt'] not in reply,
        missing_private_evidence_acknowledged=bool(re.search(r'无法|不能|不可|未找到|没有.{0,12}(?:信息|记录|原话)|找不到|已删除', reply)),
        complete_current_query_delivered=query in wire,
        public_rule_preserved_on_actual_wire=bool(knowledge) and fixture['documents'][0]['content'] in knowledge[1],
        public_course_number_correct='YY-573-R' in reply,
        ordinary_rain_course_still_runs=bool(re.search(r'(?:普通雨天|普通下雨|雨天).{0,24}(?:照常|正常|不取消|不停课)', reply)),
        only_authoritative_course_cited=[c.get('source_id') for c in generations[-1]['response'].get('citations', [])] == ['doc_1_chunk_0'],
        raw_source_read_explicitly_enabled=proof['controlled_ablation']['raw_source_recall_enabled'] is True,
        only_two_new_requests=[g['id'] for g in generations] == ['erase_friend_source', 'post_erase_friend_read'],
        no_prior_writing_or_answers_replayed=proof['reused_native_fixture']['source_writing_generations_replayed'] == 0
        and proof['reused_native_fixture']['prior_answer_generations_replayed'] == 0,
    )
