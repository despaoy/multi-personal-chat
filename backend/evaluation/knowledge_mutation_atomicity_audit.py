"""Audit actual rollback, independent update, refreshed evidence and one answer."""

import re
from html import unescape


def audit_mutation(p, f, calls):
    from inference.answer_citations import citation_keys

    initial = p["initial_state"]
    failed = p["after_failure"]
    after = p["after_writer"]
    final = p["after_generation"]
    writers = p["writers"]
    retry = writers.get("retry", {})
    update = f["update"]
    case = f["cases"][0]
    generation = p["generation"][0]
    answers = [c for c in calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    a = answers[0] if len(answers) == 1 else {}
    messages = a.get("request", {}).get("messages", [])
    wire = unescape(messages[-1]["content"]) if messages else ""
    raw = a.get("response", {}).get("choices", [{"message": {"content": ""}}])[0]["message"].get("content", "")
    reply = generation["response"].get("reply", "")
    citations = generation["response"].get("citations") or []
    namespace = p["generation_diagnostics"][0]["citation_namespace"]
    keys = citation_keys(raw, namespace)
    doc = p["final_documents"][0]
    compact = re.sub(r"\s+", "", reply).replace("：", ":")
    return dict(
        actual_authenticated_independent_writers=all(
            w["auth_statuses"] == [200, 200] and w["pid"] != p["reader_pid"] and w["database"] == p["reader_database"]
            for w in writers.values()
        )
        and p["auth_statuses"] == [200, 200]
        and p["actual_account_role"] == "admin",
        actual_complete_parent_reused=p["parent_verified_checks"] == 32
        and p["document_imports_replayed"] == p["reused_native_fixture"]["prior_answer_generations_replayed"] == 0
        and p["parent_source_unchanged"],
        real_dirty_failure_installed_and_removed=p["actual_database_failure_installed"]
        and p["actual_failure_trigger_removed"],
        failed_notification_is_not_success=writers["failure"]["http_status"] == 500,
        failed_transaction_preserves_document_and_chunks=failed["document"] == initial["document"]
        and failed["chunks"] == initial["chunks"],
        failed_transaction_preserves_revision_authority=failed["revision"] == initial["revision"]
        and failed["status"] == initial["status"],
        actual_retry_commits_full_document_and_chunks=retry.get("http_status") == 200
        and after["document"]["title"] == update["title"]
        and after["document"]["content"] == update["content"]
        and "".join(c["content"] for c in after["chunks"]) == update["content"]
        and after["document"]["chunkCount"] == len(after["chunks"]),
        committed_change_advances_revision_once=after["revision"] == initial["revision"] + 1,
        committed_change_is_dirty_before_reader=after["status"][0] == "dirty",
        other_documents_and_chunks_unchanged=p["initial_documents"][1:] == p["final_documents"][1:]
        and all(
            p["initial_chunks"][str(d["id"])] == p["final_chunks"][str(d["id"])] for d in p["initial_documents"][1:]
        ),
        ordinary_generation_refreshes_warm_reader=final["revision"] == after["revision"]
        and final["cache_generation"] > after["cache_generation"]
        and final["status"][0] == "complete"
        and final["status"][3] == after["revision"],
        actual_loaded_index_has_new_full_evidence=any(
            m["title"] == update["title"] and update["content"] in m["content"] for m in final["vector_metadata"]
        ),
        actual_model_question_and_complete_updated_source=len(answers) == 1
        and case["message"] in wire
        and update["content"] in wire
        and update["title"] in wire,
        source_data_stays_outside_system=all(
            update["content"] not in m["content"] and update["title"] not in m["content"]
            for m in messages
            if m["role"] == "system"
        ),
        current_actual_pro_calls=bool(calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in calls),
        complete_current_facts_and_confirmation=all(
            x in compact
            for x in [
                "2026",
                "12月20",
                "三楼",
                "竹影教室",
                "16:10",
                "17:50",
                "舒棠",
                "LL-842-Q",
                "护目镜",
                "耐割手套",
                "红色",
                "12月18",
                "19:00",
            ]
        )
        and bool(re.search(r"周日|星期日|星期天", reply))
        and bool(re.search(r"周五|星期五", reply))
        and bool(re.search(r"普通雨天.{0,16}(?:照常|正常)", reply))
        and bool(re.search(r"(?:唯一|只有).{0,12}(?:红色|停课)", reply))
        and bool(re.search(r"书面.{0,10}确认|确认.{0,10}书面", reply))
        and bool(re.search(r"(?:不能|不代表|无法).{0,18}(?:证明|个人|任何人).{0,18}(?:预约|确认)", reply)),
        actual_used_current_source_cited=bool(keys)
        and list(dict.fromkeys(keys)) == [c["key"] for c in citations]
        and any(
            c["source_id"] == f"doc_{doc['id']}_chunk_0" and c["source_title"] == update["title"] for c in citations
        ),
        current_owned_markers_stripped=not re.search(r"\[\[cite:" + re.escape(namespace) + r":", reply),
        persisted_actual_index_sources=p["persisted_vector_stats"]["total_documents"]
        == p["persisted_vector_stats"]["index_size"]
        == p["persisted_vector_stats"]["bm25_corpus_size"]
        == p["persisted_valid_chunk_count"]
        == 3,
        all_actual_writer_handles_terminal=bool(p["writer_handles"])
        and all(
            h["terminal"] and h["returncode"] == 0 and writers[name]["sync_pending_final"] == 0
            for name, h in p["writer_handles"].items()
        ),
        reader_all_actual_work_terminal=p["jobs_terminal"]
        and p["sync_pending_final"]
        == p["completion_runtime_terminal"]["active"]
        == p["completion_runtime_terminal"]["reserved"]
        == p["queue_stats"]["active"]
        == p["queue_stats"]["queue_size"]
        == 0,
        exactly_one_new_answer=len(p["generation"]) == len(p["generation_diagnostics"]) == 1
        and generation["http_status"] == 200,
    )
