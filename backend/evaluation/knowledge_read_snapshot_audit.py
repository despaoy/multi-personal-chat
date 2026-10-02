"""Detect impossible cross-page combinations using actual ordered writer states."""

import re
from html import unescape


def audit_read_snapshot(p, f, calls):
    ids = p["target_ids"]
    states = p["writer"]["states"]
    rows = p["reader_rows"]
    by_id = {r["documentId"]: r for r in rows}
    result = p["fallback_response"]["response"]
    returned = {r["documentId"]: r for r in result.get("results", [])}
    possible = [tuple(d["content"] for d in state["documents"]) for state in states]
    actual = tuple(by_id[i]["content"] for i in ids)
    initial = [f["initial_arrangement"], f["initial_confirmation"]]
    final = [f["final_arrangement"], f["final_confirmation"]]
    checks = dict(
        native_login=p["auth_statuses"] == [200, 200],
        actual_verified_parent=p["parent_verified_checks"] == 23
        and p["parent_source_unchanged"]
        and p["reused_native_fixture"]["prior_answer_generations_replayed"] == 0,
        full_initial_sources=all(
            d["content"] == expected["content"] and d["title"] == expected["title"]
            for d, expected in zip(p["initial_paired_documents"], initial)
        ),
        actual_503_complete_records=p["expected_count"]
        == len(rows)
        == len({(r["documentId"], r["chunkIndex"]) for r in rows})
        == 503
        and p["noise_count"] == 499,
        actual_independent_admin_writer=p["writer"]["pid"] != p["reader_pid"]
        and p["writer"]["database"] == p["reader_database"]
        and p["writer"]["auth_statuses"] == [200, 200]
        and p["writer"]["role"] == "admin"
        and p["writer"]["http_statuses"] == [200, 200],
        real_ordered_commits=possible
        == [
            (initial[0]["content"], initial[1]["content"]),
            (final[0]["content"], initial[1]["content"]),
            (final[0]["content"], final[1]["content"]),
        ]
        and states[1]["revision"] == states[0]["revision"] + 1
        and states[2]["revision"] == states[1]["revision"] + 1,
        actual_500_row_boundary=p["actual_gate"]["actual_rows_yielded"] == p["actual_gate"]["batch_size"] == 500
        and p["actual_gate"]["actual_first_document"]["documentId"] == ids[0]
        and next(i for i, r in enumerate(rows) if r["documentId"] == ids[1]) >= 500,
        physical_write_failure_preserved_previous_snapshot=p["physical_write_failure_kept_snapshot"],
        actual_keyword_fallback=p["fallback_response"]["http_status"] == 200
        and result.get("retrievalMode") == "keyword"
        and all(r["searchType"] == "keyword" for r in returned.values()),
        returned_complete_pair=all(i in returned and returned[i]["content"] == by_id[i]["content"] for i in ids),
        scan_matches_one_actual_committed_state=actual in possible,
        public_result_matches_one_actual_committed_state=all(i in returned for i in ids)
        and tuple(returned[i]["content"] for i in ids) in possible,
        current_full_database_pair=all(
            d["content"] == expected["content"] and d["title"] == expected["title"]
            for d, expected in zip(p["final_paired_documents"], final)
        ),
        other_parent_sources_preserved=p["other_parent_documents_preserved"],
        actual_writer_terminal=p["writer_handle"]["terminal"]
        and p["writer_handle"]["returncode"] == 0
        and p["writer"]["sync_pending_final"] == p["sync_pending_final"] == 0,
    )
    if not p["generation"]:
        checks["baseline_no_cloud_or_question_replay"] = calls == [] and p["new_questions_replayed"] == 0
        return checks
    g = p["generation"][0]
    answers = [c for c in calls[slice(*g["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    a = answers[0] if len(answers) == 1 else {}
    messages = a.get("request", {}).get("messages", [])
    wire = unescape(messages[-1]["content"]) if messages else ""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    raw = a.get("response", {}).get("choices", [{"message": {}}])[0]["message"].get("content", "")
    reply = g["response"].get("reply", "")
    compact = re.sub(r"\s+", "", reply).replace("：", ":")
    citations = g["response"].get("citations") or []
    namespace = p["generation_diagnostics"][0]["citation_namespace"]
    from inference.answer_citations import citation_keys

    keys = citation_keys(raw, namespace)
    actual_titles = {f"doc_{i}_chunk_0": d["title"] for i, d in zip(ids, p["final_paired_documents"])}
    checks.update(
        exactly_one_new_actual_answer=len(p["generation"]) == len(p["generation_diagnostics"]) == len(answers) == 1
        and g["http_status"] == 200,
        current_pro_http200=bool(calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in calls),
        complete_new_question_and_matching_sources=f["cases"][0]["message"] in wire
        and all(d["content"] in wire for d in final),
        source_rules_outside_system=all(d["content"] not in system and d["title"] not in system for d in final),
        current_two_sources_actually_cited=bool(keys)
        and list(dict.fromkeys(keys)) == [c["key"] for c in citations]
        and set(actual_titles).issubset({c["source_id"] for c in citations})
        and all(
            c["source_title"] == actual_titles[c["source_id"]] for c in citations if c["source_id"] in actual_titles
        ),
        current_citation_excerpts=all(
            any(
                c["source_id"] == f"doc_{i}_chunk_0"
                and c["evidence_excerpt"].startswith(f"[{f['knowledge_base']}/未分类] {d['title']}:")
                for c in citations
            )
            for i, d in zip(ids, final)
        ),
        current_version_and_course_fields=all(
            x in compact
            for x in [
                "CR-26",
                "2026",
                "12月26",
                "四楼",
                "兰亭教室",
                "14:50",
                "16:30",
                "禾音",
                "护目镜",
                "耐割手套",
                "12月24",
                "18:30",
            ]
        )
        and bool(re.search(r"周六|星期六", reply))
        and bool(re.search(r"周四|星期四", reply)),
        current_exceptions_and_confirmation=bool(re.search(r"普通雨天.{0,16}(?:照常|正常)", reply))
        and bool(re.search(r"(?:唯一|只有).{0,12}红色暴雨", reply))
        and "书面" in reply
        and "邮件" in reply
        and "确认" in reply
        and bool(re.search(r"(?:不能|不代表|无法).{0,18}(?:证明|个人|任何人).{0,18}(?:预约|确认)", reply)),
        full_persisted_index_503=p["persisted_vector_stats"]["total_documents"]
        == p["persisted_vector_stats"]["index_size"]
        == p["persisted_vector_stats"]["bm25_corpus_size"]
        == p["persisted_valid_chunk_count"]
        == 503,
        reader_work_terminal=p["jobs_terminal"]
        and p["completion_runtime_terminal"]["active"]
        == p["completion_runtime_terminal"]["reserved"]
        == p["queue_stats"]["active"]
        == p["queue_stats"]["queue_size"]
        == 0,
        owned_markers_not_public=bool(namespace) and "[[cite:" + namespace + ":" not in reply,
    )
    return checks
