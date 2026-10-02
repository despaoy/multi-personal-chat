"""Current actual source paths, preserved rules and atomic container metadata."""

import re
from html import unescape


def audit_container(p, f, calls):
    from inference.answer_citations import citation_keys

    initial = p["initial_state"]
    rename = p["observations"][0]
    detach = p["observations"][1]
    doc = p["final_documents"][0]
    checks = dict(
        actual_admin_independent_writers=p["auth_statuses"] == [200, 200]
        and p["actual_account_role"] == "admin"
        and all(
            w["auth_statuses"] == [200, 200]
            and w["role"] == "admin"
            and w["pid"] != p["reader_pid"]
            and w["database"] == p["reader_database"]
            for w in p["writers"].values()
        ),
        complete_verified_parent_reused=p["parent_verified_checks"] == 22
        and p["document_imports_replayed"] == p["reused_native_fixture"]["prior_answer_generations_replayed"] == 0
        and p["parent_source_unchanged"]
        and p["new_setup_api_requests"] == 2,
        actual_rename_preserves_description=rename["before_generation"]["base"]["name"] == f["new_base_name"]
        and rename["before_generation"]["base"]["description"] == p["initial_base"]["description"],
        actual_rename_revision_once=rename["before_generation"]["revision"] == initial["revision"] + 1,
        actual_rename_dirty_before_read=rename["before_generation"]["status"][0] == "dirty",
        actual_folder_removed_and_document_uncategorized=detach["before_generation"]["folder"] is None
        and doc["folder_id"] is None
        and doc["category"] == "未分类",
        actual_folder_change_revision_once=detach["before_generation"]["revision"]
        == rename["after_generation"]["revision"] + 1,
        all_original_rule_bodies_and_chunks_preserved=p["initial_chunks"] == p["final_chunks"]
        and all(
            before["content"] == after["content"] and before["title"] == after["title"]
            for before, after in zip(p["initial_parent_documents"], p["final_documents"])
        ),
        other_documents_unchanged=p["initial_parent_documents"][1:] == p["final_documents"][1:],
        actual_persisted_three_sources=p["persisted_vector_stats"]["total_documents"]
        == p["persisted_vector_stats"]["index_size"]
        == p["persisted_vector_stats"]["bm25_corpus_size"]
        == p["persisted_valid_chunk_count"]
        == 3,
        all_actual_handles_terminal=all(
            h["terminal"] and h["returncode"] == 0 and p["writers"][label]["sync_pending_final"] == 0
            for label, h in p["writer_handles"].items()
        ),
        actual_reader_work_terminal=p["jobs_terminal"]
        and p["sync_pending_final"]
        == p["completion_runtime_terminal"]["active"]
        == p["completion_runtime_terminal"]["reserved"]
        == p["queue_stats"]["active"]
        == p["queue_stats"]["queue_size"]
        == 0,
        actual_current_pro_calls=bool(calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in calls),
        exactly_two_new_answers=len(p["generation"]) == len(p["generation_diagnostics"]) == len(p["observations"]) == 2,
    )
    current_titles = {f"doc_{d['id']}_chunk_0": d["title"] for d in p["final_documents"]}
    for index, (case, g, observation, diagnostic) in enumerate(
        zip(f["cases"], p["generation"], p["observations"], p["generation_diagnostics"])
    ):
        answers = [c for c in calls[slice(*g["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        a = answers[0] if len(answers) == 1 else {}
        messages = a.get("request", {}).get("messages", [])
        wire = unescape(messages[-1]["content"]) if messages else ""
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        raw = a.get("response", {}).get("choices", [{"message": {"content": ""}}])[0]["message"].get("content", "")
        reply = g["response"].get("reply", "")
        compact = re.sub(r"\s+", "", reply).replace("：", ":")
        citations = g["response"].get("citations") or []
        namespace = diagnostic["citation_namespace"]
        keys = citation_keys(raw, namespace)
        category = f["new_folder_name"] if index == 0 else "未分类"
        prefix = f"[{f['new_base_name']}/{category}] {doc['title']}:"
        target = f"doc_{doc['id']}_chunk_0"
        actual = observation["after_generation"]
        old_prefix = f"[{p['initial_base']['name']}/{f['new_folder_name']}] {doc['title']}:"
        local = dict(
            real_one_answer=g["http_status"] == 200 and len(answers) == 1,
            ordinary_generation_refreshes_local_cache=actual["cache_generation"]
            > observation["before_generation"]["cache_generation"]
            and actual["status"][0] == "complete"
            and actual["status"][3] == actual["revision"],
            actual_loaded_target_has_current_path=any(
                m["id"] == target and m["content"].startswith(prefix) for m in actual["vector_metadata"]
            ),
            actual_wire_complete_question_and_all_sources=case["message"] in wire
            and all(d["content"] in wire for d in p["final_documents"]),
            actual_wire_current_target_path=prefix in wire and old_prefix not in wire,
            actual_sources_stay_outside_system=all(
                d["content"] not in system and d["title"] not in system for d in p["final_documents"]
            ),
            current_used_citations_exactly_bind_raw=bool(keys)
            and list(dict.fromkeys(keys)) == [c["key"] for c in citations]
            and all(
                c["source_id"] in current_titles and c["source_title"] == current_titles[c["source_id"]]
                for c in citations
            ),
            actual_citation_excerpt_has_current_path=any(
                c["source_id"] == target and c["evidence_excerpt"].startswith(prefix) for c in citations
            ),
            current_owned_markers_not_public=not re.search(r"\[\[cite:" + re.escape(namespace) + r":", reply),
            full_course_fields_and_exceptions=all(
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
        )
        checks.update({case["id"] + "_" + key: value for key, value in local.items()})
    return checks
