"""Authenticated account boundaries on actual source-memory model requests."""

import json
import re
from html import unescape


def forbidden_private_payloads(calls, forbidden):
    """Inspect every message of every actual model request, including writer."""
    return all(all(token not in unescape(json.dumps(message, ensure_ascii=False))
                   for token in forbidden)
               for call in calls for message in call['request']['messages'])


def audit_memory_scope_wire(proof, calls, fixture):
    owner = proof['scope_owner_identity']
    identity = proof['scope_current_identity']
    diag = proof['prepared_diagnostics'][-1]
    case = fixture['cases'][-1]
    generation = proof['generation'][-1]
    answers = [call for call in calls[slice(*generation['cloud_call_range'])]
               if call['request'].get('max_tokens') == 1024]
    reply = generation['response'].get('reply', '')
    before = proof['scope_owner_sources_before']
    friend = [row for row in before if row['body'] == fixture['cases'][2]['message']]
    scope = diag['user_scope']
    sql = proof['scope_sql']
    forged = proof['scope_forged_request']
    forbidden = [fixture['friend_receipt'], fixture['own_receipt'], *[case['message'] for case in fixture['cases'][:3]]]
    answer_messages = answers[0]['request']['messages'] if len(answers) == 1 else []
    return dict(
        source_scope_uses_distinct_authenticated_account=identity != owner and proof['scope_secondary_role'] == 'user',
        native_owner_login_and_me_succeeded=proof['auth_statuses'] == [200, 200],
        native_secondary_register_login_me_succeeded=proof['scope_secondary_auth_statuses'] == [200, 200, 200],
        source_recall_matches_live_enabled_configuration=proof['controlled_ablation']['raw_source_recall_enabled'] is True,
        complete_owner_friend_quote_preexisted=len(friend) == 1,
        complete_owner_claims_preexisted=len(proof['scope_owner_claims_before']) == 2,
        forged_sender_and_user_belong_to_other_owner=forged['senderId'] == forged['userId'] == owner,
        forged_conversation_and_session_belong_to_other_owner=forged['conversationId'] == forged['sessionId'] == owner,
        forged_source_id_matches_preexisting_other_owner=bool(friend) and forged['sourceMessageId'] == friend[0]['source_message_id'],
        query_does_not_contain_private_receipts=fixture['friend_receipt'] not in case['message'] and fixture['own_receipt'] not in case['message'],
        real_preparation_normalized_sender=scope['sender_id'] == identity,
        real_preparation_kept_expected_web_scope=scope['platform'] == 'web' and scope['adapter'] == 'web-character' and scope['conversation_type'] == 'private',
        memory_scope_uses_current_authenticated_owner=diag['memory_scope_key'] == ['web', 'web-character', 'private', identity],
        no_injected_or_cold_ablated_history=not proof['controlled_ablation']['history_ablation_enabled'] and not proof['cold_history_diagnostics'],
        actual_secondary_history_is_empty=diag['history'] == [],
        actual_secondary_memory_candidates_are_empty=bool(proof['candidate_diagnostics']) and all(not row['items'] and row['candidate_count'] == 0 for row in proof['candidate_diagnostics']),
        no_old_owner_selected_claims=diag['used_memory_ids'] == [],
        source_lookup_really_executed_without_cross_user_match=diag['raw_source_status'] == 'no_match',
        raw_quote_not_placed_in_prepared_context=all(token not in diag['episodic_context'] for token in forbidden),
        all_actual_model_inputs_exclude_other_owner_private_material=bool(calls) and forbidden_private_payloads(calls, forbidden),
        one_actual_pro_answer=len(answers) == 1 and answers[0]['response'].get('model') == 'deepseek-v4-pro',
        complete_current_question_reaches_actual_model=bool(answer_messages) and case['message'] in unescape(answer_messages[-1]['content']),
        answer_excludes_other_owner_receipts=fixture['friend_receipt'] not in reply and fixture['own_receipt'] not in reply,
        answer_acknowledges_missing_current_owner_evidence=bool(re.search(r'无法|不能|未找到|没有.{0,20}(?:信息|记录|原话)|找不到|不清楚', reply)),
        original_owner_sources_unchanged=proof['scope_owner_sources_after'] == before,
        original_owner_claims_unchanged=proof['scope_owner_claims_after'] == proof['scope_owner_claims_before'],
        original_owner_chat_archive_unchanged=sql['owner_messages_sha256_before'] == sql['owner_messages_sha256_after'],
        cloned_source_database_unchanged=sql['source_database_snapshot_before'] == sql['source_database_snapshot_after'],
        original_owner_quote_still_recorded=sql['owner_friend_state'] == 'recorded' and sql['owner_friend_body_preserved'] is True,
        same_source_id_stored_in_separate_owner_scopes=sql['owner_friend_archive_count'] == sql['current_request_archive_count'] == 1 and sql['source_keys_distinct'] is True,
        only_current_owner_question_captured=any(row['body'] == case['message'] and row['source_message_id'] == forged['sourceMessageId'] for row in proof['scope_current_sources_after']),
        private_scope_must_not_follow_forged_session=diag['memory_scope_key'][-1] != forged['sessionId'],
        one_new_request_only=len(proof['generation']) == 1 and generation['id'] == case['id'],
        no_prior_writing_answers_or_imports_replayed=proof['reused_native_fixture']['source_writing_generations_replayed'] == proof['reused_native_fixture']['prior_answer_generations_replayed'] == proof['document_imports_replayed'] == 0,
    )


def collision_receipt_is_safe(receipt, source_id):
    return receipt.get('source_message_id') == source_id and receipt.get('source_capture') == 'conflict' and receipt.get('status') == 'skipped' and receipt.get('reason') == 'source_conflict' and receipt.get('accepted') == receipt.get('persisted') == 0


def audit_memory_collision_wire(proof, calls, fixture):
    source_id = proof['scope_forged_request']['sourceMessageId']
    query = fixture['cases'][-1]['message']
    original = [row for row in proof['scope_owner_sources_before'] if row['body'] == fixture['cases'][2]['message']]
    current = [row for row in proof['scope_owner_sources_after'] if row['source_message_id'] == source_id]
    receipt = proof['memory_status']['recent_results']
    generation = proof['generation'][-1]
    actual = calls[slice(*generation['cloud_call_range'])]
    answers = [call for call in actual if call['request'].get('max_tokens') == 1024]
    writers = []
    for call in actual:
        for message in call['request']['messages']:
            try:
                payload = json.loads(message['content'])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and 'current_user_message' in payload and ('existing_memories' in payload or 'source_erasure_candidates' in payload):
                writers.append(call)
    diag = proof['prepared_diagnostics'][-1]
    sql = proof['scope_sql']
    return dict(
        same_authenticated_owner_used=proof['scope_current_identity'] == proof['scope_owner_identity'],
        normal_native_login_and_me_succeeded=proof['auth_statuses'] == [200, 200],
        complete_old_source_and_id_preexist=len(original) == 1 and original[0]['source_message_id'] == source_id,
        different_complete_current_statement=query != fixture['cases'][2]['message'] and len(query) > 100,
        current_statement_is_not_erasure=not proof['collision_query_is_erasure'],
        real_private_scope_uses_original_owner=diag['memory_scope_key'] == ['web', 'web-character', 'private', proof['scope_owner_identity']],
        actual_source_read_enabled=proof['controlled_ablation']['raw_source_recall_enabled'] is True and diag['raw_source_status'] == 'available',
        ordinary_history_not_injected_or_ablated=not proof['controlled_ablation']['history_ablation_enabled'] and proof['cold_history_diagnostics'] == [],
        exact_one_committed_capture_conflict=len(receipt) == 1 and collision_receipt_is_safe(receipt[0], source_id),
        no_actual_semantic_writer_for_conflicting_source=not writers,
        no_other_completion_after_answer=bool(answers) and actual[-1] == answers[0],
        complete_current_statement_reaches_actual_pro_answer=len(answers) == 1 and query in unescape(answers[0]['request']['messages'][-1]['content']),
        one_actual_large_answer=len(answers) == 1 and answers[0]['response'].get('model') == 'deepseek-v4-pro',
        complete_immutable_source_and_clock_preserved=bool(original) and current == original,
        all_original_sources_preserved=proof['scope_owner_sources_after'] == proof['scope_owner_sources_before'],
        no_new_claim_or_source_association=proof['scope_owner_claims_after'] == proof['scope_owner_claims_before'],
        new_conflicting_text_not_captured_under_old_id=not any(row['body'] == query for row in proof['scope_current_sources_after']),
        original_chat_archive_rows_preserved=sql['owner_messages_sha256_before'] == sql['owner_original_messages_sha256_after'],
        one_new_chat_row_with_complete_statement=sql['current_request_archive_count'] == 1 and sql['owner_message_count_after'] == sql['owner_message_count_before'] + 1,
        old_chat_quote_remains_exact_once=sql['owner_friend_archive_count'] == 1,
        same_owner_source_identity_did_not_fork=sql['source_keys_distinct'] is False,
        original_source_still_recorded=sql['owner_friend_state'] == 'recorded' and sql['owner_friend_body_preserved'] is True,
        cloned_source_fixture_database_unchanged=sql['source_database_snapshot_before'] == sql['source_database_snapshot_after'],
        one_new_request_only=len(proof['generation']) == 1 and generation['id'] == 'same_owner_source_collision',
        no_prior_writing_answers_or_document_imports=proof['reused_native_fixture']['source_writing_generations_replayed'] == proof['reused_native_fixture']['prior_answer_generations_replayed'] == proof['document_imports_replayed'] == 0,
    )


def no_writer_results(results):
    return isinstance(results, (tuple, list)) and not results


def audit_memory_collision_rejected(proof, calls, fixture):
    generation = proof['generation'][-1]
    source_id = proof['scope_forged_request']['sourceMessageId']
    original = [row for row in proof['scope_owner_sources_before'] if row['body'] == fixture['cases'][2]['message']]
    sql = proof['scope_sql']
    return dict(
        complete_original_statement_and_identity_verified=len(original) == 1 and original[0]['source_message_id'] == source_id,
        exact_complete_failed_statement_resubmitted=proof['collision_submitted_message'] == fixture['cases'][-1]['message'] and len(proof['collision_submitted_message']) > 100,
        same_authenticated_owner=proof['scope_owner_identity'] == proof['scope_current_identity'],
        normal_native_login=proof['auth_statuses'] == [200, 200],
        conflict_rejected_explicitly_before_answer=generation['http_status'] == 409 and generation['response'].get('detail', {}).get('code') == 'source_identity_conflict',
        no_model_calls_on_rejected_request=calls == [],
        no_preparation_or_generation_after_rejection=proof['prepared_diagnostics'] == [] and proof['generation_diagnostics'] == [],
        no_memory_writer_job_after_rejection=no_writer_results(proof['memory_status']['recent_results']),
        old_source_body_and_clock_unchanged=proof['scope_owner_sources_before'] == proof['scope_owner_sources_after'],
        no_claim_or_link_mutation=proof['scope_owner_claims_before'] == proof['scope_owner_claims_after'],
        chat_archive_unchanged=sql['owner_messages_sha256_before'] == sql['owner_messages_sha256_after'],
        no_new_chat_row_for_rejected_statement=sql['current_request_archive_count'] == 0 and sql['owner_message_count_before'] == sql['owner_message_count_after'],
        old_friend_quote_remains_recorded=sql['owner_friend_state'] == 'recorded' and sql['owner_friend_body_preserved'] is True and sql['owner_friend_archive_count'] == 1,
        source_fixture_database_unchanged=sql['source_database_snapshot_before'] == sql['source_database_snapshot_after'],
        no_source_identity_fork=sql['source_keys_distinct'] is False,
        no_ordinary_history_ablation=proof['controlled_ablation']['history_ablation_enabled'] is False and proof['cold_history_diagnostics'] == [],
        source_flag_matches_production=proof['controlled_ablation']['raw_source_recall_enabled'] is True,
        configured_large_model_unchanged=proof['configured_model'] == 'deepseek-v4-pro' and proof['configured_context_window'] == 65536,
        no_old_writing_answers_documents_or_search_replayed=proof['reused_native_fixture']['source_writing_generations_replayed'] == proof['reused_native_fixture']['prior_answer_generations_replayed'] == proof['document_imports_replayed'] == 0 and proof['searches'] == [],
        only_one_failed_request_revalidated=len(proof['generation']) == 1 and generation['id'] == 'same_owner_source_collision',
    )
