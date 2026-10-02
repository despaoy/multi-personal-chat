"""Actual database/index/citation consistency after a metadata-only move."""

import re
from html import unescape


def audit_metadata_scope(proof, fixture, calls):
    generation = proof["generation"][0]
    answers = [c for c in calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    messages = answers[0]["request"]["messages"] if len(answers) == 1 else []
    wire = unescape(messages[-1]["content"]) if messages else ""
    citations = generation["response"].get("citations") or []
    before = proof["before_update"]
    updated = proof["immediately_after_update"]
    after = proof["after_generation"]
    doc_id = proof["actual_target_document_id"]
    source_id = f"doc_{doc_id}_chunk_0"
    metadata = [r for r in after["vector_metadata"] if r.get("id") == source_id]

    def search(label):
        return next(s for s in proof["searches"] if s["label"] == label)["response"]["results"]

    reply = generation["response"].get("reply", "")
    compact = re.sub(r"\s+", "", reply).replace("：", ":")
    old_title = fixture["documents"][0]["title"]
    new_title = fixture["metadata_update"]["title"]
    return dict(
        actual_authenticated_admin=proof["auth_statuses"] == [200, 200, 200]
        and proof["actual_account_role"] == "admin",
        actual_three_document_imports=len(proof["documents"]) == 3
        and all(d["http_status"] == 200 for d in proof["documents"]),
        actual_metadata_only_update=proof["metadata_update_response"]["http_status"] == 200
        and "content" not in proof["actual_update_payload"],
        true_database_move_and_title=updated["document"]["knowledge_base_id"] == proof["actual_base_ids"][1]
        and updated["document"]["title"] == new_title,
        complete_source_and_chunks_unchanged=before["document"]["content"]
        == updated["document"]["content"]
        == proof["final_document"]["content"]
        == fixture["documents"][0]["content"]
        and before["chunks"] == updated["chunks"] == proof["final_chunks"],
        actual_index_was_warm=before["index_built"] and before["status"][0] == "complete",
        metadata_change_marks_dirty=not updated["index_built"]
        and updated["status"][0] == "dirty"
        and updated["revision"] == before["revision"] + 1,
        generation_rebuilds_before_read=after["cache_generation"] > updated["cache_generation"]
        and after["status"][0] == "complete"
        and after["status"][3] == after["revision"],
        current_vector_scope_title_category=len(metadata) == 1
        and metadata[0].get("title") == new_title
        and metadata[0].get("knowledge_base_id") == proof["actual_base_ids"][1]
        and metadata[0].get("category") == fixture["metadata_update"]["category"],
        old_scope_no_longer_recalls_moved_source=not any(
            r.get("documentId") == source_id for r in search("old_scope_after_update")
        ),
        new_scope_recalls_moved_source=any(
            r.get("documentId") == source_id and r.get("documentTitle") == new_title
            for r in search("new_scope_after_update")
        ),
        actual_one_current_pro_answer=len(answers) == 1
        and generation["http_status"] == 200
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in calls),
        complete_question_in_actual_wire=fixture["cases"][0]["message"] in wire,
        full_public_source_in_actual_wire=fixture["documents"][0]["content"] in wire,
        actual_source_title_updated_on_wire=new_title in wire and old_title not in wire,
        source_material_outside_system_rules=all(
            fixture["documents"][0]["content"] not in m["content"] for m in messages if m.get("role") == "system"
        ),
        original_history_empty=proof["generation_diagnostics"][0]["history"] == [],
        correct_current_course_fields=all(v in reply for v in ["澜岚", "LL-936-T", "舒棠", "12", "13"])
        and bool(re.search(r"15[:：]20|十五点二十|三点二十", reply))
        and bool(re.search(r"周日|星期日|星期天", reply)),
        complete_remaining_course_fields=all(
            v in compact for v in ["2026", "12月13", "17:00", "藤月教室", "护目镜", "耐割手套", "12月11", "18:00"]
        ),
        public_rules_not_personal_confirmation=bool(
            re.search(r"(?:不|不能|不代表).{0,18}(?:证明|个人|任何人).{0,18}(?:预约|确认)", reply)
        ),
        course_negations_and_exception_understood=bool(re.search(r"普通.{0,8}雨|一般.{0,8}雨", reply))
        and bool(re.search(r"照常|正常", reply))
        and "红色" in reply
        and bool(re.search(r"书面.{0,10}确认|确认.{0,10}书面", reply)),
        returned_citation_has_current_authoritative_title=any(
            c.get("source_id") == source_id and c.get("source_title") == new_title for c in citations
        ),
        no_stale_title_in_api_citations=all(c.get("source_title") != old_title for c in citations),
        persisted_index_reopens_with_all_sources=proof["persisted_vector_stats"]["total_documents"]
        == proof["persisted_vector_stats"]["index_size"]
        == proof["persisted_vector_stats"]["bm25_corpus_size"]
        == proof["persisted_valid_chunk_count"]
        == 3,
        actual_handles_terminal=proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == proof["queue_stats"]["active"]
        == proof["queue_stats"]["queue_size"]
        == 0,
    )
