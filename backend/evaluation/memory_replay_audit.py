"""Revoked source replay must not be admitted as a remembered new message."""

from evaluation.memory_scope_audit import no_writer_results


def audit_memory_replay_wire(proof, calls, fixture):
    source_id = proof['scope_forged_request']['sourceMessageId']
    sql = proof['scope_sql']
    receipt = proof['memory_status']['recent_results']
    diag = proof['prepared_diagnostics'][-1]
    query = fixture['cases'][-1]['message']
    generation = proof['generation'][-1]
    answers = [call for call in calls[slice(*generation['cloud_call_range'])] if call['request'].get('max_tokens') == 1024]
    return dict(
        revoked_fixture_gate_explicit=proof['reused_native_fixture']['verified_erasure_state'] is True,
        complete_original_quote_replayed=query == fixture['cases'][2]['message'],
        replay_of_exact_revoked_identity=source_id == proof['reused_erased_source']['source_message_id'],
        same_owner_normally_authenticated=proof['auth_statuses'] == [200, 200] and proof['scope_owner_identity'] == proof['scope_current_identity'],
        original_anchor_already_revoked=sql['replay_anchor_before'] == {'state':'revoked','body_is_null':True,'observed_at_is_null':True,'terms':0,'links':0},
        visible_read_does_not_restore_erased_quote=source_id not in diag['raw_source_diagnostics']['selected_ids'] and fixture['friend_receipt'] not in diag['episodic_context'],
        real_source_read_enabled=proof['controlled_ablation']['raw_source_recall_enabled'] is True and diag['raw_source_status'] in {'available','no_match'},
        current_complete_quote_reaches_actual_answer=len(answers) == 1 and query in answers[0]['request']['messages'][-1]['content'],
        actual_large_model_used=bool(calls) and all(call['response'].get('model') == 'deepseek-v4-pro' for call in calls),
        committed_writer_rejects_revoked_source=len(receipt) == 1 and receipt[0].get('source_message_id') == source_id and receipt[0].get('source_capture') == 'revoked' and receipt[0].get('reason') == 'source_revoked' and receipt[0].get('accepted') == receipt[0].get('persisted') == 0,
        no_completion_after_answer=bool(answers) and calls[slice(*generation['cloud_call_range'])][-1] == answers[0],
        revoked_anchor_never_resurrected=sql['replay_anchor_after'] == sql['replay_anchor_before'],
        original_sources_unchanged=proof['scope_owner_sources_before'] == proof['scope_owner_sources_after'],
        unrelated_claims_unchanged=proof['scope_owner_claims_before'] == proof['scope_owner_claims_after'],
        original_archive_rows_unchanged=sql['owner_messages_sha256_before'] == sql['owner_original_messages_sha256_after'],
        original_quote_archive_retained=sql['owner_friend_archive_count'] == 1,
        replay_chat_only_added_one_row=sql['current_request_archive_count'] == 1 and sql['owner_message_count_after'] == sql['owner_message_count_before'] + 1,
        verified_source_database_unchanged=sql['source_database_snapshot_before'] == sql['source_database_snapshot_after'],
        no_erasure_or_old_write_answer_search_replay=proof['reused_native_fixture']['source_writing_generations_replayed'] == proof['reused_native_fixture']['prior_answer_generations_replayed'] == proof['document_imports_replayed'] == 0 and proof['searches'] == [],
        one_new_replay_request=len(proof['generation']) == 1 and generation['id'] == 'revoked_source_replay',
    )


def audit_memory_replay_rejected(proof, calls, fixture):
    sql = proof['scope_sql']
    generation = proof['generation'][-1]
    return dict(
        verified_native_erasure_fixture_reused=proof['reused_native_fixture']['verified_erasure_state'] is True,
        complete_original_failed_quote_resubmitted=proof['collision_submitted_message'] == fixture['cases'][2]['message'] == fixture['cases'][-1]['message'],
        exact_revoked_source_id_used=proof['scope_forged_request']['sourceMessageId'] == proof['reused_erased_source']['source_message_id'],
        same_owner_normally_authenticated=proof['auth_statuses'] == [200, 200] and proof['scope_owner_identity'] == proof['scope_current_identity'],
        source_revocation_explicitly_rejected=generation['http_status'] == 409 and generation['response'].get('detail',{}).get('code') == 'source_identity_revoked',
        no_model_call_after_rejection=calls == [],
        no_preparation_or_generation=proof['prepared_diagnostics'] == proof['generation_diagnostics'] == [],
        no_writer_job=no_writer_results(proof['memory_status']['recent_results']),
        revoked_anchor_before_and_after_identical=sql['replay_anchor_before'] == sql['replay_anchor_after'] == {'state':'revoked','body_is_null':True,'observed_at_is_null':True,'terms':0,'links':0},
        no_source_or_claim_mutation=proof['scope_owner_sources_before'] == proof['scope_owner_sources_after'] and proof['scope_owner_claims_before'] == proof['scope_owner_claims_after'],
        no_chat_row_added=sql['current_request_archive_count'] == 0 and sql['owner_message_count_after'] == sql['owner_message_count_before'],
        original_archive_preserved=sql['owner_messages_sha256_before'] == sql['owner_messages_sha256_after'] and sql['owner_friend_archive_count'] == 1,
        source_fixture_database_preserved=sql['source_database_snapshot_before'] == sql['source_database_snapshot_after'],
        no_cold_history_ablation=proof['cold_history_diagnostics'] == [] and proof['controlled_ablation']['history_ablation_enabled'] is False,
        current_source_flag_and_large_model_retained=proof['controlled_ablation']['raw_source_recall_enabled'] is True and proof['configured_model'] == 'deepseek-v4-pro' and proof['configured_context_window'] == 65536,
        no_old_writing_erasure_answer_or_search_replayed=proof['reused_native_fixture']['source_writing_generations_replayed'] == proof['reused_native_fixture']['prior_answer_generations_replayed'] == proof['document_imports_replayed'] == 0 and proof['searches'] == [],
        only_failed_native_request_revalidated=len(proof['generation']) == 1 and generation['id'] == 'revoked_source_replay',
    )
