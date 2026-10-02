"""Real revision authority, current model evidence and terminal handles."""

import re
from html import unescape


def audit_concurrency(proof, fixture, calls):
    generation = proof["generation"][0]
    answers = [c for c in calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    messages = answers[0]["request"]["messages"] if len(answers) == 1 else []
    wire = unescape(messages[-1]["content"]) if messages else ""
    reply = generation["response"].get("reply", "")
    compact = re.sub(r"\s+", "", reply).replace("：", ":")
    title = fixture["metadata_updates"]["C"]["title"]
    source_id = f"doc_{proof['actual_target_document_id']}_chunk_0"
    metadata = [m for m in proof["after_generation"]["vector_metadata"] if m.get("id") == source_id]
    retrieved = [
        r
        for d in proof["retrieval_diagnostics"][1:]
        for r in d["bundle"].get("results", [])
        if r.get("id") == source_id
    ]
    citations = generation["response"].get("citations") or []
    initial = proof["initial_state"]
    warmed = proof["reader_warmed_after_A"]
    after_C = proof["after_C"]
    final = proof["after_all_writers"]
    after = proof["after_generation"]
    writers = proof["writers"]
    handles = proof["writer_handles"]
    pids = [w["pid"] for w in writers.values()]
    return dict(
        actual_reader_authenticated_admin=proof["auth_statuses"] == [200, 200]
        and proof["actual_account_role"] == "admin",
        real_three_distinct_writer_processes=len(pids) == len(set(pids)) == 3 and proof["reader_pid"] not in pids,
        actual_writer_accounts_and_handles=all(
            w["auth_statuses"] == [200, 200] and w["role"] == "admin" and w["pid"] == handles[name]["pid"]
            for name, w in writers.items()
        ),
        same_real_disposable_database=all(
            w["database_name"] == proof["reader_database_name"] == proof["reused_native_fixture"]["cloned_database"]
            for w in writers.values()
        ),
        full_verified_parent_reused=proof["parent_verified_checks"] == 32
        and proof["document_imports_replayed"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == 0
        and proof["parent_source_unchanged"],
        all_original_documents_and_chunks_preserved=all(
            a["content"] == b["content"] == f["content"]
            for a, b, f in zip(proof["initial_documents"], proof["final_documents"], fixture["documents"])
        )
        and proof["initial_chunks"] == proof["final_chunks"],
        three_successful_metadata_only_commits=all(w["update_response"]["http_status"] == 200 for w in writers.values())
        and all("content" not in p for p in fixture["metadata_updates"].values()),
        final_database_has_latest_submitted_title=final["document"]["title"]
        == proof["final_documents"][0]["title"]
        == title,
        delayed_revision_read_was_actual=proof["B_observed"]["pid"] == writers["B"]["pid"]
        and proof["B_observed"]["revision"] == initial["revision"],
        concurrent_authority_is_database_serialized=proof["atomic_invalidation_available"]
        and proof["B_observed"]["location"] == "locked_database_transaction"
        and bool(proof["actual_database_lock_wait"])
        and all(r["wait_event_type"] == "Lock" and r["blocking_pids"] for r in proof["actual_database_lock_wait"]),
        all_three_changes_increment_revision=final["revision"] == initial["revision"] + 3,
        later_commit_revision_cannot_regress=final["revision"] >= after_C["revision"],
        warm_reader_was_intermediate_authority=warmed["index_built"]
        and warmed["status"][0] == "complete"
        and warmed["document"]["title"] == fixture["metadata_updates"]["A"]["title"],
        committed_revision_changes_warm_reader=final["revision"] != warmed["revision"],
        ordinary_generation_refreshes_local_cache=after["cache_generation"] > final["cache_generation"],
        completed_index_tracks_actual_revision=after["index_built"]
        and after["status"][0] == "complete"
        and after["status"][3] == after["revision"] == final["revision"],
        actual_loaded_evidence_has_final_title=len(metadata) == 1
        and metadata[0]["title"] == title
        and fixture["documents"][0]["content"] in metadata[0]["content"],
        generation_retrieval_has_final_title=len(proof["retrieval_diagnostics"]) == 2
        and bool(retrieved)
        and all(r.get("title") == title for r in retrieved),
        model_wire_has_current_title_and_full_source=title in wire and fixture["documents"][0]["content"] in wire,
        new_complete_question_reached_model=fixture["cases"][0]["message"] in wire
        and proof["new_query_distinct_from_parent"],
        real_one_current_pro_answer=len(proof["generation"]) == len(answers) == 1
        and generation["http_status"] == 200
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in calls),
        reply_uses_latest_title=title in reply and fixture["metadata_updates"]["A"]["title"] not in reply,
        current_citation_title=any(
            c.get("source_id") == source_id and c.get("source_title") == title for c in citations
        ),
        no_intermediate_citation=all(
            c.get("source_title") != fixture["metadata_updates"]["A"]["title"] for c in citations
        ),
        current_course_fields=all(v in reply for v in ["澜岚", "LL-936-T", "舒棠"])
        and bool(re.search(r"15[:：]20|十五点二十", reply))
        and bool(re.search(r"周日|星期日|星期天", reply)),
        complete_remaining_course_conditions=all(
            v in compact
            for v in ["2026", "12月13", "17:00", "二楼", "藤月教室", "护目镜", "耐割手套", "12月11", "18:00"]
        ),
        rain_exception_and_written_confirmation=bool(re.search(r"普通.{0,8}雨|一般.{0,8}雨", reply))
        and bool(re.search(r"照常|正常", reply))
        and "红色" in reply
        and bool(re.search(r"书面.{0,10}确认|确认.{0,10}书面", reply)),
        public_rules_not_personal_confirmation=bool(
            re.search(r"(?:不|不能|不代表).{0,18}(?:证明|个人|任何人).{0,18}(?:预约|确认)", reply)
        ),
        source_outside_system_rules=all(
            fixture["documents"][0]["content"] not in m["content"] for m in messages if m.get("role") == "system"
        ),
        persisted_complete_source_index=proof["persisted_vector_stats"]["total_documents"]
        == proof["persisted_vector_stats"]["index_size"]
        == proof["persisted_vector_stats"]["bm25_corpus_size"]
        == proof["persisted_valid_chunk_count"]
        == 3,
        actual_writer_handles_terminal=all(
            h["returncode"] == 0 and h["terminal"] and writers[name]["sync_pending_final"] == 0
            for name, h in handles.items()
        ),
        reader_actual_work_terminal=proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == proof["queue_stats"]["active"]
        == proof["queue_stats"]["queue_size"]
        == 0,
    )
