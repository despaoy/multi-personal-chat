"""Saved authenticated RAG proof: complete uncited evidence remains grounded."""

import re
from html import unescape

from inference.citation_recovery import ANNOTATION_POLICY
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens


def audit_uncited_evidence(proof, fixture, calls, storage):
    primary = [c for c in calls if c["request"].get("max_tokens") == 1024]
    messages = primary[0]["request"]["messages"] if len(primary) == 1 else []
    wire = unescape(messages[-1]["content"]) if messages else ""
    public = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", wire, re.S)
    generation = proof["generation"][-1] if proof.get("generation") else {}
    retrieval = generation.get("retrieval", {})
    body = fixture["document"]["content"]
    target = "doc_" + str(proof.get("document_id")) + "_chunk_0"
    bundles = proof.get("retrieval") or []
    response = proof.get("response", {})
    reply = response.get("reply", "")
    compact = re.sub(r"\s+", "", reply).lower()

    def denied(term):
        return bool(re.search(r"(?:禁止|不得|不能|不允许)[^\n。！？；;]{0,70}" + term, reply))

    return {
        "synthetic_complete_inputs": proof.get("synthetic_only") is True and fixture.get("synthetic") is True,
        "actual_postgresql_authenticated_transport": proof.get("database_mode") == "PostgreSQL"
        and proof.get("transport") == "authenticated_ASGI",
        "administrator_import_and_fresh_user_auth": proof.get("auth_statuses") == [200, 200]
        and proof.get("chat_auth_statuses") == [200, 200],
        "actual_document_import_one_complete_chunk": proof.get("imports")
        == {"base_status": 200, "document_status": 200, "chunks": 1}
        and proof.get("single_stored_complete_chunk") is True,
        "actual_generate_http200": proof.get("http_status") == 200,
        "one_actual_primary": len(primary) == 1,
        "all_real_calls_pro_http200_stop": bool(calls)
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
        and any(d.get("id") == target and body in d.get("content", "") for d in bundles[0]["results"]),
        "display_disabled_but_bundle_has_source_metadata": proof.get("citations_enabled") is False
        and len(bundles) == 1
        and any(c.get("source_id") == target for c in bundles[0]["citations"]),
        "actual_current_question_whole": fixture["question"] in wire,
        "actual_full_public_body": bool(public) and body in public[1],
        "late_conditions_present": bool(public)
        and "最终有效条件：" in public[1]
        and body.index("最终有效条件：") > 2000,
        "all_selected_chunks_enter_whole": bool(public)
        and len(bundles) == 1
        and all(
            d.get("content", "") in public[1]
            for d in bundles[0]["results"]
            if isinstance(d.get("content"), str) and d["content"].strip()
        ),
        "whole_target_packet_identity": any(
            p.get("document_ids") == [target] and body in p.get("text", "")
            for p in retrieval.get("evidence_packets", [])
        ),
        "public_material_not_system_instruction": bool(messages)
        and all(body not in m["content"] for m in messages if m["role"] == "system"),
        "actual_request_fits_working_budget": bool(messages)
        and sum(estimated_tokens(m["content"]) + 4 for m in messages) + 1024 + CONTEXT_SAFETY_MARGIN_TOKENS <= 65536,
        "no_citation_namespace_or_policy": not retrieval.get("answer_citations_bound")
        and not retrieval.get("citation_namespace")
        and bool(messages)
        and all("本轮来源标记示例：" not in m["content"] for m in messages),
        "no_citation_repair_cloud_call": not any(
            m.get("content") == ANNOTATION_POLICY
            for c in calls
            for m in c["request"].get("messages", [])
            if m.get("role") == "system"
        ),
        "visible_citations_disabled": not response.get("citations") and "[[cite:" not in reply,
        "actual_grounded_answer_mode": response.get("answerMode") == "grounded_answer"
        and response.get("abstained") is False,
        "fresh_owner_history_empty": bool(proof.get("prepared")) and proof["prepared"][-1]["history"] == [],
        "dynamic_policy_applied": bool(proof.get("prepared")) and proof["prepared"][-1]["policy_status"] == "applied",
        "empty_candidate_semantic_review_not_needed": bool(proof.get("prepared"))
        and proof["prepared"][-1]["semantic_status"] == "not_needed",
        "original_document_and_chunk_unchanged": proof.get("document_unchanged") is True
        and proof.get("chunks_unchanged") is True,
        "sync_work_finished": proof.get("sync_pending_final") == 0,
        "readonly_no_public_facts_in_user_memory": storage.get("read_only_transaction") is True
        and storage.get("new_user_no_public_fact_claims") is True
        and storage.get("typed_claim_count") == 0,
        "question_stored_once": storage.get("current_question_source_stored_once") is True
        and storage.get("question_source_count") == 1,
        "answer_public_code": fixture["expected"]["code"] in reply,
        "answer_device_and_vram": "3060" in compact and "12gb" in compact,
        "answer_ready_model_dependencies": "mini-0.6b" in compact
        and all(t in reply for t in ["权重", "分词器", "依赖", "缓存"]),
        "answer_complete_data_split": all(t in compact for t in ["1800", "1440", "360"])
        and bool(re.search(r"三分类|三类", reply)),
        "answer_offline_network_download_cloud_denials": "离线" in reply
        and denied("联网")
        and denied("下载")
        and denied(r"云\s*API"),
        "answer_training_and_finetuning_denials": denied("训练") and denied("微调"),
        "answer_validation_tuning_denial": denied(r"验证集调参"),
        "answer_report_requirements": all(t in reply for t in ["固定划分", "显存峰值", "准确率", "复现日志"]),
        "public_rules_do_not_authorize_private_actions": bool(
            re.search(r"不证明个人获准|不代表个人获准|不构成个人许可", reply)
        ),
    }
