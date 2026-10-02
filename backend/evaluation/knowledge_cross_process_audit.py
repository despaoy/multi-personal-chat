"""Audit measured cross-process authority without inventing source state."""

from evaluation.knowledge_metadata_scope_audit import audit_metadata_scope


def audit_cross_process(proof, fixture, calls):
    checks = audit_metadata_scope(proof, fixture, calls)
    for name in [
        "actual_authenticated_admin",
        "actual_three_document_imports",
        "metadata_change_marks_dirty",
        "original_history_empty",
    ]:
        checks.pop(name)
    # The old title legitimately remains in the unchanged source body and
    # inherited conversation. Validate the current retrieved title separately.
    from html import unescape

    generation = proof["generation"][0]
    answers = [c for c in calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    wire = unescape(answers[0]["request"]["messages"][-1]["content"]) if len(answers) == 1 else ""
    source_id = f"doc_{proof['actual_target_document_id']}_chunk_0"
    retrieved = [
        row
        for diagnostic in proof["retrieval_diagnostics"][1:]
        for row in diagnostic["bundle"].get("results", [])
        if row.get("id") == source_id
    ]
    checks["actual_source_title_updated_on_wire"] = (
        len(proof["retrieval_diagnostics"]) == 2
        and fixture["metadata_update"]["title"] in wire
        and bool(retrieved)
        and all(row.get("title") == fixture["metadata_update"]["title"] for row in retrieved)
    )
    before = proof["before_update"]
    updated = proof["immediately_after_update"]
    writer = proof["writer"]
    checks.update(
        inherited_history_has_no_artificial_ablation=not proof["controlled_ablation"]["history_ablation_enabled"]
        and isinstance(proof["generation_diagnostics"][0]["history"], list),
        actual_authenticated_reader_admin=proof["auth_statuses"] == [200, 200]
        and proof["actual_account_role"] == "admin",
        real_distinct_reader_writer_processes=writer["pid"] == proof["writer_handle"]["pid"]
        and writer["pid"] != proof["reader_pid"],
        same_actual_disposable_database=writer["database_name"]
        == proof["reader_database_name"]
        == proof["reused_native_fixture"]["cloned_database"],
        independent_measured_index_paths=writer["vector_path"] == proof["writer_vector_path"]
        and writer["vector_path"] != proof["reader_vector_path"],
        source_fixture_reused_without_replay=proof["document_imports_replayed"] == 0
        and proof["reused_native_fixture"]["prior_answer_generations_replayed"] == 0
        and proof["parent_source_unchanged"]
        and proof["parent_complete_fixture_verified"],
        actual_writer_revision_dirty_then_complete=writer["before_revision"] == before["revision"]
        and writer["after_update"]["revision"] == before["revision"] + 1
        and writer["after_update"]["status"][0] == "dirty"
        and not writer["after_update"]["index_built"]
        and writer["after_rebuild"]["status"][0] == "complete"
        and writer["after_rebuild"]["status"][3] == writer["after_rebuild"]["revision"] == updated["revision"],
        global_complete_does_not_refresh_local_memory_by_itself=updated["index_built"]
        and updated["vector_metadata"] == before["vector_metadata"]
        and updated["cache_generation"] == before["cache_generation"]
        and updated["status"][0] == "complete",
        writer_new_scope_is_current=any(
            r["documentId"] == f"doc_{proof['actual_target_document_id']}_chunk_0"
            and r["documentTitle"] == fixture["metadata_update"]["title"]
            for r in writer["search_response"]["results"]
        ),
        writer_actual_handle_terminal=proof["writer_handle"]["returncode"] == 0
        and proof["writer_handle"]["terminal"]
        and writer["sync_pending_final"] == 0,
        new_reader_question_not_old_parent_replay=proof["new_query_distinct_from_parent"]
        and len(proof["generation"]) == 1,
    )
    return checks
