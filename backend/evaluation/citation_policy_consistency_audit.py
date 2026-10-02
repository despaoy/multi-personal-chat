"""Only actually returned current application markers authorize citations."""

import re
from html import unescape


def audit_citation_policy(proof, fixture, calls):
    from inference.answer_citations import citation_keys

    documents = proof["initial_documents"]
    checks = dict(
        authenticated_admin=proof["auth_statuses"] == [200, 200] and proof["actual_account_role"] == "admin",
        parent_known_failure_kept_honest=proof["parent_checks_passed"] == 31
        and proof["parent_checks_total"] == 32
        and proof["parent_known_unresolved"] == ["current_citation_title"],
        verified_fixture_reused_without_old_runs=proof["document_imports_replayed"]
        == proof["document_mutations"]
        == proof["reused_native_fixture"]["prior_answer_generations_replayed"]
        == 0
        and proof["new_queries_distinct_from_parent"],
        full_documents_chunks_and_revision_unchanged=proof["initial_documents"] == proof["final_documents"]
        and proof["initial_chunks"] == proof["final_chunks"]
        and proof["initial_revision"] == proof["final_revision"],
        parent_source_unchanged=proof["parent_source_unchanged"],
        actual_current_pro_calls=bool(calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in calls),
        persisted_index_keeps_all_sources=proof["persisted_vector_stats"]["total_documents"]
        == proof["persisted_vector_stats"]["index_size"]
        == proof["persisted_vector_stats"]["bm25_corpus_size"]
        == proof["persisted_valid_chunk_count"]
        == 3,
        all_actual_work_terminal=proof["jobs_terminal"]
        and proof["sync_pending_final"]
        == proof["completion_runtime_terminal"]["active"]
        == proof["completion_runtime_terminal"]["reserved"]
        == proof["queue_stats"]["active"]
        == proof["queue_stats"]["queue_size"]
        == 0,
    )
    for case, generation, diagnostic in zip(fixture["cases"], proof["generation"], proof["generation_diagnostics"]):
        answers = [c for c in calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        answer = answers[0] if len(answers) == 1 else {}
        messages = answer.get("request", {}).get("messages", [])
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        wire = unescape(messages[-1]["content"]) if messages else ""
        raw = answer.get("response", {}).get("choices", [{"message": {"content": ""}}])[0]["message"].get("content", "")
        namespace = diagnostic.get("citation_namespace", "")
        keys = citation_keys(raw, namespace)
        citations = generation["response"].get("citations") or []
        reply = generation["response"].get("reply", "")
        actual_ids = [c.get("source_id") for c in citations]
        expected_ids = {f"doc_{documents[i]['id']}_chunk_0" for i in case["required_document_positions"]}
        current_titles = {f"doc_{d['id']}_chunk_0": d["title"] for d in documents}
        prefix = case["id"] + "_"
        local = dict(
            real_one_answer=len(answers) == 1 and generation["http_status"] == 200,
            no_conflicting_marker_instruction="正文无需输出引用标记" not in system
            and "使用某份资料支持外部事实时，在对应事实后写出该来源标记" in system,
            actual_complete_question_and_sources=case["message"] in wire
            and all(d["content"] in wire for d in documents),
            source_data_stays_outside_system=all(
                d["content"] not in system and d["title"] not in system for d in documents
            ),
            current_application_markers_in_actual_raw=bool(keys) and bool(re.fullmatch(r"[0-9a-f]{12}", namespace)),
            raw_keys_bind_exact_api_citations=list(dict.fromkeys(keys)) == [c.get("key") for c in citations],
            required_answer_used_sources_returned=expected_ids.issubset(set(actual_ids)),
            returned_sources_have_actual_current_identity=bool(citations)
            and all(
                c.get("source_id") in current_titles and c.get("source_title") == current_titles[c["source_id"]]
                for c in citations
            ),
            only_application_markers_removed_from_public_reply=not re.search(
                r"\[\[cite:" + re.escape(namespace) + r":", reply
            ),
            public_number_and_confirmation_conditions="LL-936-T" in reply
            and "18" in reply
            and bool(re.search(r"书面.{0,10}确认|确认.{0,10}书面", reply))
            and bool(re.search(r"(?:不|不能|不代表).{0,18}(?:证明|个人|任何人).{0,18}(?:预约|确认)", reply)),
        )
        if case.get("literal_tokens"):
            local["user_literal_markers_preserved"] = all(
                token in reply and token in raw for token in case["literal_tokens"]
            )
            local["literal_does_not_bind_an_unrelated_source"] = set(actual_ids) == expected_ids and "S99" not in [
                c.get("key") for c in citations
            ]
        else:
            compact = re.sub(r"\s+", "", reply).replace("：", ":")
            local["complete_course_fields_and_exceptions"] = (
                all(
                    v in compact
                    for v in [
                        "2026",
                        "12月13",
                        "17:00",
                        "二楼",
                        "藤月教室",
                        "舒棠",
                        "护目镜",
                        "耐割手套",
                        "12月11",
                        "18:00",
                        "红色",
                    ]
                )
                and bool(re.search(r"15[:：]20|十五点二十", reply))
                and bool(re.search(r"周日|星期日|星期天", reply))
                and bool(re.search(r"照常|正常", reply))
            )
        checks.update({prefix + key: value for key, value in local.items()})
    checks["exact_new_case_count"] = (
        len(fixture["cases"]) == len(proof["generation"]) == len(proof["generation_diagnostics"]) == 2
    )
    return checks
