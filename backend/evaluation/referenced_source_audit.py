"""Audit actual explicit source references, packet ownership and full answer."""

import json
import re
from html import unescape

from inference.citation_recovery import ANNOTATION_POLICY
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens


def audit_referenced_source(proof, fixture, calls, storage):
    primary = [c for c in calls if c["request"].get("max_tokens") == fixture["expected_answer_tokens"]]
    messages = primary[0]["request"]["messages"] if len(primary) == 1 else []
    wire = unescape(messages[-1]["content"]) if messages else ""
    public = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", wire, re.S)
    bundles = proof.get("retrieval") or []
    selected = bundles[0].get("results", []) if len(bundles) == 1 else []
    generation = proof.get("generation", [{}])[-1]
    retrieval = generation.get("retrieval", {})
    clauses = {
        key: fixture[key]
        for key in ["device_clause", "reference_clause", "permission_clause", "report_clause", "current_clause"]
    }
    anchors = [d for d in selected if d.get("retrieval_role") != "source_context"]
    related = [d for d in selected if d.get("retrieval_role") == "source_context"]
    target_prefix = "doc_" + str(proof.get("document_id")) + "_chunk_"
    response = proof.get("response", {})
    reply = response.get("reply", "")
    compact = re.sub(r"\s+", "", reply).lower()
    prepared = proof.get("prepared", [{}])[-1]

    def denied(term):
        return bool(re.search(r"(?:禁止|不得|不能|不允许)[^\n。！？；;]{0,100}" + term, reply))

    chain = {
        "complete_synthetic_input": fixture.get("synthetic") is True and proof.get("synthetic_only") is True,
        "all_required_clauses_in_complete_original": all(
            any(text in document["content"] for document in fixture["documents"]) for text in clauses.values()
        ),
        "authenticated_postgresql": proof.get("database_mode") == "PostgreSQL"
        and proof.get("transport") == "authenticated_ASGI",
        "administrator_import_and_fresh_user_auth": proof.get("auth_statuses") == [200, 200]
        and proof.get("chat_auth_statuses") == [200, 200],
        "actual_import_complete_three_document_set": proof.get("imports")
        == {
            "base_status": 200,
            "document_statuses": [200, 200, 200],
            "chunks": [3, 1, 1],
            "method": "authenticated_json",
        }
        and proof.get("complete_stored_chunk_set") is True,
        "actual_generate_http200": proof.get("http_status") == 200,
        "one_actual_primary": len(primary) == 1,
        "all_actual_calls_pro_http200_stop": bool(calls)
        and all(
            c["request"].get("model") == "deepseek-v4-pro"
            and c["http_status"] == 200
            and c["response"].get("model") == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        "actual_confident_generic_retrieval": len(bundles) == 1
        and not bundles[0]["abstained"]
        and bundles[0].get("retrieval_strategy") != "multi_scale_character"
        and len(anchors) == 3
        and len(related) > 0,
        "actual_imported_source_expanded": sum(d.get("id", "").startswith(target_prefix) for d in selected) == 3,
        "all_required_clauses_actually_selected": all(
            any(text in d.get("content", "") for d in selected) for text in clauses.values()
        ),
        "all_required_clauses_in_actual_public_request": bool(public)
        and all(text in public[1] for text in clauses.values()),
        "cross_document_reference_keeps_exact_authorized_referrer": any(
            d.get("document_id") == proof["document_ids"][1]
            and d.get("retrieval_role") == "source_context"
            and any(
                a["id"] in d.get("supporting_document_ids", [])
                and a.get("knowledge_base_id") == d.get("knowledge_base_id")
                and "《" + d["title"] + "》" in a["content"]
                for a in selected
            )
            for d in related
        ),
        "unreferenced_historical_document_not_expanded": not any(
            d.get("document_id") == proof["document_ids"][2] for d in related
        ),
        "related_context_does_not_add_confidence_votes": bool(related)
        and all(d.get("score") == 0 and d.get("normalized_score") == 0 for d in related),
        "related_packet_budget_dependency_kept": bool(related)
        and all(
            any(
                p.get("document_ids") == [d["id"]]
                and p.get("kind") == "background"
                and p.get("supporting_document_ids") == d.get("supporting_document_ids")
                for p in retrieval.get("evidence_packets", [])
            )
            for d in related
        ),
        "all_selected_chunks_preserved_whole": bool(public)
        and bool(selected)
        and all(d.get("content", "") in public[1] for d in selected),
        "all_selected_packet_identities_and_bodies_preserved": bool(selected)
        and all(
            any(
                p.get("document_ids") == [d["id"]] and d["content"] in p.get("text", "")
                for p in retrieval.get("evidence_packets", [])
            )
            for d in selected
        ),
        "current_question_whole": fixture["question"] in wire,
        "public_source_not_system_instruction": bool(messages)
        and all(all(text not in m["content"] for text in clauses.values()) for m in messages if m["role"] == "system"),
        "working_budget_fits": bool(messages)
        and sum(estimated_tokens(m["content"]) + 4 for m in messages)
        + fixture["expected_answer_tokens"]
        + CONTEXT_SAFETY_MARGIN_TOKENS
        <= 65536,
        "citation_display_remains_disabled": proof.get("citations_enabled") is False
        and not response.get("citations")
        and "[[cite:" not in reply,
        "no_citation_namespace_or_output_policy": not retrieval.get("answer_citations_bound")
        and not retrieval.get("citation_namespace")
        and bool(messages)
        and all("本轮来源标记示例：" not in m["content"] for m in messages),
        "no_annotation_model_call": not any(
            m.get("content") == ANNOTATION_POLICY for c in calls for m in c["request"].get("messages", [])
        ),
        "grounded_answer_mode": response.get("answerMode") == "grounded_answer" and response.get("abstained") is False,
        "fresh_owner_history_empty": prepared.get("history") == [],
        "dynamic_policy_applied": prepared.get("policy_status") == "applied",
        "empty_candidate_review_not_needed": prepared.get("semantic_status") == "not_needed",
        "original_source_document_and_chunks_unchanged": proof.get("document_unchanged") is True
        and proof.get("chunks_unchanged") is True,
        "all_sync_writes_finished": proof.get("sync_pending_final") == 0,
        "exact_isolated_readonly_storage": storage.get("read_only_transaction") is True
        and storage.get("exact_isolated_database") is True,
        "public_rules_not_promoted_to_personal_claims": storage.get("typed_claim_count") == 0
        and storage.get("new_user_no_public_fact_claims") is True,
        "exact_owner_scope_question_stored_once": storage.get("question_source_count") == 1
        and storage.get("exact_owner_scope") is True,
        "original_full_document_readback_exact": storage.get("complete_public_document_unchanged") is True,
    }
    current_id = "doc_" + str(proof["document_ids"][1]) + "_chunk_0"
    scope_match = re.search(r"<retrieval_coverage[^>]*>\n(.*?)\n</retrieval_coverage>", wire, re.S)
    scope = json.loads(scope_match[1]) if scope_match else {}
    current_scope = next(
        (row for row in scope.get("sources", []) if row.get("source_id") == "doc_" + str(proof["document_ids"][1])), {}
    )
    admitted_ids = {
        str(identity) for packet in retrieval.get("evidence_packets", []) for identity in packet.get("document_ids", [])
    }
    chain.update(
        {
            "current_revision_full_scope_in_actual_wire": current_scope.get("indexed_chunk_count")
            == current_scope.get("retrieved_chunk_count")
            == current_scope.get("admitted_chunk_count")
            == 1
            and current_scope.get("status") == "all_indexed_chunks_admitted",
            "current_revision_actual_packet_admitted": current_id in admitted_ids,
            "final_packet_support_ids_admitted": all(
                set(packet.get("supporting_document_ids", [])).intersection(admitted_ids)
                for packet in retrieval.get("evidence_packets", [])
                if packet.get("kind") == "background"
            ),
            "confidence_uses_ranked_anchors_only": bool(anchors)
            and bundles[0].get("confidence")
            == round(
                0.7 * max(0.0, min(1.0, float(anchors[0]["score"])))
                + 0.3 * sum(max(0.0, min(1.0, float(row["score"]))) for row in anchors) / len(anchors),
                4,
            ),
        }
    )
    answer = {
        "procedure_or_current_revision_identified": fixture["expected"]["main"] in reply
        or fixture["expected"]["revision"] in reply,
        "device_and_vram": "3060" in compact and "12gb" in compact,
        "complete_cached_model_resources": "mini-0.6b" in compact
        and all(t in reply for t in ["权重", "分词器", "依赖", "缓存"]),
        "complete_data_split": all(str(fixture["expected"][key]) in compact for key in ["total", "train", "validation"])
        and bool(re.search(r"三分类|三类", reply)),
        "network_download_cloud_denied": denied("联网") and denied("下载") and denied(r"云\s*API"),
        "training_and_finetuning_denied": denied("训练") and denied("微调"),
        "validation_tuning_denied": denied(r"验证集调参"),
        "all_report_requirements": all(
            t in reply for t in ["宏平均", "F1", "显存峰值", "推理延迟", "数据划分", "复现日志"]
        ),
        "fixed_execution_parameters": bool(re.search(r"批(?:大小|量)[^\n。！？]{0,12}10", reply))
        and bool(re.search(r"(?:随机)?种子[^\n。！？]{0,12}97", reply))
        and "fp16" in compact,
        "explicit_offline_inference_only": bool(
            re.search(r"(?:只能|只允许|仅(?:能|限)?)[^\n。！？；;]{0,20}离线[^\n。！？；;]{0,8}推理", reply)
        ),
    }
    return {"chain": chain, "answer": answer}
