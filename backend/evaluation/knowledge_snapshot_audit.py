"""Audit saved native snapshot evidence without repeating model calls."""

import re
from html import unescape


def audit_snapshot(p, f, calls):
    from inference.answer_citations import citation_keys

    g = p["generation"][0]
    answers = [c for c in calls[slice(*g["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    a = answers[0] if len(answers) == 1 else {}
    messages = a.get("request", {}).get("messages", [])
    wire = unescape(messages[-1]["content"]) if messages else ""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    raw = a.get("response", {}).get("choices", [{"message": {}}])[0]["message"].get("content", "")
    response = g["response"]
    reply = response.get("reply", "")
    compact = re.sub(r"\s+", "", reply).replace("：", ":")
    citations = response.get("citations") or []
    namespace = p["generation_diagnostics"][0]["citation_namespace"]
    keys = citation_keys(raw, namespace)
    docs = p["final_documents"]
    target = docs[0]
    prefix = f"[{f['knowledge_base']}/未分类] {target['title']}:"
    storage = p["storage_concurrency"]
    observed = storage["during_partial_publication"]
    return dict(
        native_login=p["auth_statuses"] == [200, 200],
        verified_complete_parent=p["parent_verified_checks"] == 34
        and p["parent_source_unchanged"]
        and p["reused_native_fixture"]["prior_answer_generations_replayed"] == 0,
        full_database_sources_preserved=p["initial_parent_documents"] == docs
        and p["initial_chunks"] == p["final_chunks"]
        and all(d["content"] == expected["content"] for d, expected in zip(docs, f["documents"])),
        legacy_same_count_refused=p["legacy"]["stats"]["total_documents"]
        == p["legacy"]["stats"]["index_size"]
        == p["legacy"]["stats"]["bm25_corpus_size"]
        == 3
        and not p["legacy"]["validated"]
        and not p["legacy"]["may_skip"],
        actual_database_migration=p["migration"]["http_status"] == 200
        and p["migration"]["validated"]
        and p["migration"]["status"][0] == "complete"
        and p["migration"]["revision"] == 12,
        real_concurrency_complete=all(storage["checks"].values()) and storage["cloud_calls"] == 0,
        real_concurrency_independent_handles=len(
            {storage["reader_pid"], *[w["pid"] for w in storage["writers"].values()]}
        )
        == 3
        and storage["all_writer_handles_terminal"],
        actual_partial_window_binds_keyword=observed["returned_source_id"] == "doc_1_chunk_0"
        and observed["returned_source_contains_query"]
        and observed["bm25_corpus"] == observed["metadata_texts"],
        copied_real_snapshot_hash=p["copied_snapshot_sha256"] == p["published_snapshot_sha256"],
        cold_loaded_real_snapshot=p["cold_snapshot"]["validated"]
        and p["cold_snapshot"]["corpus_aligned"]
        and p["cold_snapshot"]["metadata_ids"] == storage["writers"]["A"]["metadata_ids"],
        persisted_all_three=p["persisted_vector_stats"]["total_documents"]
        == p["persisted_vector_stats"]["index_size"]
        == p["persisted_vector_stats"]["bm25_corpus_size"]
        == p["persisted_valid_chunk_count"]
        == 3,
        exactly_one_new_answer=len(p["generation"]) == len(p["generation_diagnostics"]) == len(f["cases"]) == 1
        and len(answers) == 1
        and g["http_status"] == 200,
        actual_current_pro=bool(calls)
        and all(c["http_status"] == 200 and c["response"].get("model") == "deepseek-v4-pro" for c in calls),
        complete_question_and_full_sources=f["cases"][0]["message"] in wire and all(d["content"] in wire for d in docs),
        actual_wire_current_path=prefix in wire,
        materials_outside_system=all(d["title"] not in system and d["content"] not in system for d in docs),
        current_citations_bind_actual_raw=bool(namespace)
        and bool(keys)
        and list(dict.fromkeys(keys)) == [c["key"] for c in citations]
        and all(
            any(c["source_id"] == f"doc_{d['id']}_chunk_0" and c["source_title"] == d["title"] for d in docs)
            for c in citations
        ),
        actual_target_citation=any(
            c["source_id"] == "doc_1_chunk_0" and c["evidence_excerpt"].startswith(prefix) for c in citations
        ),
        transport_markers_hidden=bool(namespace) and "[[cite:" + namespace + ":" not in reply,
        complete_course_fields=all(
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
                "12月18",
                "19:00",
            ]
        )
        and bool(re.search(r"周日|星期日|星期天", reply))
        and bool(re.search(r"周五|星期五", reply)),
        rain_and_only_exception=bool(re.search(r"普通雨天.{0,16}(?:照常|正常)", reply))
        and bool(re.search(r"(?:唯一|只有).{0,12}红色暴雨", reply)),
        written_confirmation_not_personal_proof=bool(re.search(r"书面.{0,10}确认|确认.{0,10}书面", reply))
        and bool(re.search(r"(?:不能|不代表|无法).{0,18}(?:证明|个人|任何人).{0,18}(?:预约|确认)", reply)),
        actual_reader_work_terminal=p["jobs_terminal"]
        and p["sync_pending_final"]
        == p["completion_runtime_terminal"]["active"]
        == p["completion_runtime_terminal"]["reserved"]
        == p["queue_stats"]["active"]
        == p["queue_stats"]["queue_size"]
        == 0,
    )
