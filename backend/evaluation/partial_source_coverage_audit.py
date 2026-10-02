"""Saved proof for partial indexed scope and exact final packet admission."""

import json
import re
from html import unescape

from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from inference.evidence_coverage import SOURCE_COVERAGE_POLICY


def audit_partial_source(proof, fixture, calls, storage):
    primary = [c for c in calls if c["request"].get("max_tokens") == fixture["expected_answer_tokens"]]
    messages = primary[0]["request"]["messages"] if len(primary) == 1 else []
    raw_wire = messages[-1]["content"] if messages else ""
    public = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", raw_wire, re.S)
    block = re.search(r"<retrieval_coverage[^>]*>\n(.*?)\n</retrieval_coverage>", raw_wire, re.S)
    try:
        wire_scope = json.loads(unescape(block[1])) if block else {}
    except (ValueError, TypeError):
        wire_scope = {}
    generation = proof.get("generation", [{}])[-1]
    retrieval = generation.get("retrieval", {})
    packets = retrieval.get("evidence_packets", [])
    admitted_ids = {i for p in packets for i in p.get("document_ids", [])}
    bundles = proof.get("retrieval") or []
    results = bundles[0].get("results", []) if len(bundles) == 1 else []
    by_id = {d["id"]: d for d in results}
    source_id = f"doc_{proof.get('document_id')}"
    source = next((s for s in retrieval.get("source_coverage", []) if s.get("source_id") == source_id), {})
    rendered = next((s for s in wire_scope.get("sources", []) if s.get("source_id") == source_id), {})
    indexed_ids = {f"{source_id}_chunk_{i}" for i in range(fixture["expected_chunk_count"])}
    target_retrieved = indexed_ids.intersection(by_id)
    target_admitted = indexed_ids.intersection(admitted_ids)
    response = proof.get("response", {})
    reply = response.get("reply", "")
    compact = re.sub(r"\s+", "", reply).lower()
    prepared = proof.get("prepared", [{}])[-1]
    imported = proof.get("imports", {})
    chain = {
        "complete_synthetic_original": fixture.get("synthetic") is True and proof.get("synthetic_only") is True,
        "actual_authenticated_postgresql": proof.get("database_mode") == "PostgreSQL"
        and proof.get("transport") == "authenticated_ASGI",
        "admin_import_and_new_owner_authenticated": proof.get("auth_statuses") == [200, 200]
        and proof.get("chat_auth_statuses") == [200, 200],
        "authorized_complete_zip_import": imported.get("method") == "authenticated_zip"
        and imported.get("base_status") == imported.get("document_status") == imported.get("listing_status") == 200
        and imported.get("zip_errors") == []
        and imported.get("chunks") == fixture["expected_chunk_count"],
        "original_complete_chunk_set_verified": proof.get("complete_stored_chunk_set") is True,
        "actual_http200": proof.get("http_status") == 200,
        "one_actual_primary": len(primary) == 1,
        "all_actual_request_response_pro_200_stop": bool(calls)
        and all(
            c["request"].get("model") == "deepseek-v4-pro"
            and c["http_status"] == 200
            and c["response"].get("model") == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        "real_confident_generic_retrieval": len(bundles) == 1
        and not bundles[0]["abstained"]
        and bundles[0].get("retrieval_strategy") != "multi_scale_character",
        "source_scope_exact_imported_index_ids": bool(source)
        and set(source.get("indexed_document_ids", [])) == indexed_ids,
        "actual_producer_omission": 0 < len(target_retrieved) < len(indexed_ids),
        "actual_final_budget_omission": 0 < len(target_admitted) < len(target_retrieved),
        "indexed_count_exact": rendered.get("indexed_chunk_count") == len(indexed_ids),
        "retrieved_count_exact": rendered.get("retrieved_chunk_count") == len(target_retrieved),
        "admitted_count_exact": rendered.get("admitted_chunk_count") == len(target_admitted),
        "partial_source_status_exact": rendered.get("status") == source.get("status") == "partial",
        "both_budget_omission_reasons_retained": rendered.get("omission_reasons")
        == ["source_context_budget", "request_budget"],
        "actual_packet_counts_exact": wire_scope.get("packets", {}).get("candidate_packet_count") == len(results)
        and wire_scope.get("packets", {}).get("admitted_packet_count") == len(packets)
        and wire_scope.get("packets", {}).get("status") == "partial",
        "public_body_only_whole_admitted_chunks": bool(public)
        and bool(admitted_ids)
        and all(i in by_id and by_id[i]["content"] in unescape(public[1]) for i in admitted_ids),
        "coverage_has_no_unadmitted_source_text": bool(block)
        and all(d["content"] not in unescape(block[1]) for i, d in by_id.items() if i not in admitted_ids),
        "coverage_counts_separate_from_system_source_data": bool(block)
        and all(fixture["document"]["title"] not in m["content"] for m in messages if m["role"] == "system"),
        "coverage_policy_in_actual_system": any(
            SOURCE_COVERAGE_POLICY in m["content"] for m in messages if m["role"] == "system"
        ),
        "known_device_data_preserved": bool(public) and fixture["device_clause"] in unescape(public[1]),
        "current_question_whole": fixture["question"] in unescape(raw_wire),
        "actual_working_budget_fits": bool(messages)
        and sum(estimated_tokens(m["content"]) + 4 for m in messages)
        + fixture["expected_answer_tokens"]
        + CONTEXT_SAFETY_MARGIN_TOKENS
        <= 65536,
        "response_exposes_partial_warning": "partial_source_context" in (response.get("warnings") or []),
        "uncited_grounded_independent_facts": proof.get("citations_enabled") is False
        and not response.get("citations")
        and response.get("answerMode") == "grounded_answer"
        and response.get("abstained") is False
        and "[[cite:" not in reply,
        "new_owner_history_empty": prepared.get("history") == [],
        "dynamic_policy_applied": prepared.get("policy_status") == "applied",
        "original_document_and_chunks_unchanged": proof.get("document_unchanged") is True
        and proof.get("chunks_unchanged") is True,
        "write_queue_finished": proof.get("sync_pending_final") == 0,
        "exact_isolated_readonly_storage": storage.get("read_only_transaction") is True
        and storage.get("exact_isolated_database") is True,
        "public_facts_not_promoted_to_private_claims": storage.get("typed_claim_count") == 0,
        "exact_owner_question_stored_once": storage.get("exact_owner_scope") is True
        and storage.get("question_source_count") == 1,
        "raw_original_readback_unchanged": storage.get("complete_public_document_unchanged") is True,
    }
    answer = {
        "partial_scope_explained": bool(
            re.search(r"部分(?:资料|片段|章节)|只覆盖|不(?:是|代表)[^。\n]{0,12}全部章节", reply)
        ),
        "missing_scope_remains_unconfirmed": bool(
            re.search(
                r"(?:不能|无法|未能)确认[^。\n]{0,55}(?:未|其他|完整|全部)|(?:未取得|未检索到|缺失)[^。\n]{0,70}(?:不能|无法|未能)确认",
                reply,
            )
        ),
        "no_affirmative_full_review": not re.search(
            r"(?:^|[\n。！？])\s*(?:本次|当前)?(?:已经|已)(?:核对全部|覆盖全部|读完全文|审阅全文)", reply
        ),
        "known_device_vram_correct": "3060" in compact and "12gb" in compact,
        "known_data_split_correct": all(
            str(fixture["expected"][k]) in compact for k in ["total", "train", "validation"]
        )
        and bool(re.search(r"三分类|三类", reply)),
        "no_unsupported_network_training_grant": not re.search(
            r"(?:^|[\n。！？])\s*(?:本次|当前)?(?:已批准|允许|可以|获准)[^。\n]{0,12}(?:联网|训练)", reply
        ),
        "internal_ids_counts_not_displayed": source_id not in reply
        and "partial_source_context" not in reply
        and "indexed_chunk_count" not in reply,
    }
    return {"chain": chain, "answer": answer}
