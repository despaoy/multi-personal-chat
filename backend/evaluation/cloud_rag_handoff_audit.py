"""Audit saved mixed-source wire and answer, with no model calls."""

import json
import re
from html import unescape

from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens


def audit_handoff(proof, fixture, calls):
    primary = [c for c in calls if c["request"].get("max_tokens") == 1024]
    messages = primary[0]["request"]["messages"] if len(primary) == 1 else []
    wire = unescape(messages[-1]["content"]) if messages else ""
    match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    try:
        private = json.loads(match[1]) if match else {}
    except (ValueError, TypeError):
        private = {}
    public = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", wire, re.S)
    actual = proof["generation"][-1] if proof.get("generation") else {}
    context = actual.get("context", {})
    retrieval = actual.get("retrieval", {})
    source = fixture["private"]
    document = fixture["public"]
    reply = proof.get("reply", "")
    compact = re.sub(r"\s+", "", reply).lower()
    citations = proof.get("meta", {}).get("citations") or []
    bundles = proof.get("retrieval") or []
    raw = primary[0]["response"]["choices"][0]["message"]["content"] if len(primary) == 1 else ""
    namespace = retrieval.get("citation_namespace", "")
    private_lines = [
        line for line in raw.splitlines() if any(term in line for term in ["3060", "Mini-0.6B", "1440", "360"])
    ]

    def denied(term):
        return bool(re.search(r"(?:禁止|不得|不能|不允许)[^\n。！？；;]{0,55}" + term, reply))

    return {
        "synthetic_complete_inputs": proof.get("synthetic_only") is True and fixture.get("synthetic") is True,
        "one_actual_primary": len(primary) == 1,
        "all_actual_deepseek_pro_http200_stop": bool(calls)
        and all(
            c["http_status"] == 200
            and c["response"].get("model") == "deepseek-v4-pro"
            and c["response"]["choices"][0]["finish_reason"] == "stop"
            for c in calls
        ),
        "native_shared_api_budget": proof.get("requested_budget") == {"window": 65536, "evidence_max_chars": 0},
        "provider_factory_defers_public_chars": len(bundles) == 1 and bundles[0]["context_budget"]["max_chars"] is None,
        "actual_public_retrieval_confident": len(bundles) == 1
        and not bundles[0]["abstained"]
        and document["body"] in bundles[0]["context_text"],
        "actual_question_complete": fixture["question"] in wire,
        "private_whole_source_identity_and_body": any(
            row.get("source_id") == source["id"] and row.get("text") == source["body"]
            for row in private.get("records", [])
        ),
        "private_speech_unresolved": private.get("speaker_role") == "user"
        and private.get("current_validity") == "not_resolved"
        and private.get("described_subject") == "not_resolved",
        "public_whole_source_in_actual_wire": bool(public) and document["body"] in public[1],
        "private_not_public_source": bool(public) and source["body"] not in public[1] and source["id"] not in public[1],
        "public_not_private_source": all(document["body"] != row.get("text") for row in private.get("records", [])),
        "materials_outside_system": bool(messages)
        and all(
            source["body"] not in m["content"] and document["body"] not in m["content"]
            for m in messages
            if m["role"] == "system"
        ),
        "actual_packet_admission_kept_public_id": len(retrieval.get("evidence_packets", [])) == 1
        and retrieval["evidence_packets"][0]["document_ids"] == [document["id"]],
        "final_private_budget_settled": context.get("memory_source_status") == "available"
        and context.get("source_candidate_context") == "",
        "actual_request_fits_working_window": bool(messages)
        and sum(estimated_tokens(m["content"]) + 4 for m in messages) + 1024 + CONTEXT_SAFETY_MARGIN_TOKENS <= 65536,
        "no_original_source_rewrite": proof.get("original_sources_unchanged") is True,
        "no_speech_promoted_to_claim": proof.get("claims_remain_empty") is True,
        "native_dynamic_policy_applied": proof.get("prepared", {}).get("policy_status") == "applied",
        "empty_candidate_semantic_review_not_needed": proof.get("prepared", {}).get("semantic_status") == "not_needed",
        "answer_device_and_vram": "3060" in compact and "12gb" in compact,
        "answer_ready_model_and_dependencies": "mini-0.6b" in compact
        and all(x in reply for x in ["权重", "分词器", "依赖"]),
        "answer_complete_fixed_split": all(x in compact for x in ["1800", "1440", "360"])
        and bool(re.search(r"三分类|三类", reply)),
        "answer_offline_network_and_cloud_denied": "离线" in reply and denied("联网") and denied(r"云\s*API"),
        "answer_training_and_finetuning_denied": denied("训练") and denied("微调"),
        "answer_validation_tuning_denied": denied(r"验证集调参"),
        "answer_public_number": fixture["expected"]["public_code"] in reply,
        "answer_report_requirements": all(x in reply for x in ["显存峰值", "准确率", "固定数据划分"]),
        "answer_no_enrollment_or_attendance": bool(
            re.search(r"尚未报名或参加|(?:尚未|未|没有|还没)[^。\n]{0,12}报名", reply)
        )
        and bool(re.search(r"尚未报名或参加|(?:尚未|未|没有|还没)[^。\n]{0,12}参加", reply)),
        "answer_public_course_not_private_authority": "正常开课" in reply
        and bool(re.search(r"不证明|没有替你确认|不代表", reply)),
        "only_owned_public_source_cited": len(citations) == 1
        and citations[0].get("source_id") == document["id"]
        and citations[0].get("source_title") == document["title"]
        and citations[0].get("source_path") == document["source_path"],
        "actual_namespace_marker_binds_citation": bool(re.fullmatch(r"[0-9a-f]{12}", namespace))
        and any("[[cite:" + namespace + ":" + c.get("key", "") + "]]" in raw for c in citations),
        "private_claims_not_public_citations": bool(private_lines)
        and all("[[cite:" not in line for line in private_lines),
        "visible_answer_transport_stripped": "[[cite:" not in reply,
    }
