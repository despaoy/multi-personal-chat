"""Audit immutable candidate text and independently checked literal evidence."""

import hashlib
import json


def audit_recovery(p, f, calls):
    original = p["baseline"]
    recovered = p["recovered"]
    citations = recovered["citations"]
    sources = {s["source_id"]: s for s in p["admitted_source_bodies"]}
    docs = {d["id"]: d for d in p["current_source_documents"]}
    call = calls[0] if len(calls) == 1 else {}
    request = call.get("request", {})
    response = call.get("response", {})
    messages = request.get("messages", [])
    try:
        payload = json.loads(messages[-1]["content"]) if messages else {}
    except ValueError:
        payload = {}
    try:
        annotation = json.loads(response.get("choices", [{"message": {}}])[0]["message"].get("content", ""))
    except (ValueError, TypeError):
        annotation = {}
    annotations = annotation.get("citations", []) if isinstance(annotation, dict) else []
    by_key = {
        source["key"]: {
            "metadata": {"source_id": source["source_id"], "source_title": docs[source["source_id"]]["title"]},
            "key": source["key"],
            "body": source["body"],
        }
        for source in p["admitted_source_bodies"]
    }
    from inference.citation_recovery import validate_annotations

    validated = validate_annotations(json.dumps(annotation), original["reply"], list(by_key.values()))
    selected = list(dict.fromkeys(a["key"] for a in annotations))
    first_quotes = {key: next(a["quote"] for a in annotations if a["key"] == key) for key in selected}
    return dict(
        synthetic_sources_exact_fixture_allowlist=sorted(s["body"] for s in p["admitted_source_bodies"])
        == sorted(f["synthetic_source_bodies"]),
        native_clone_login=p["auth_statuses"] == [200, 200],
        exact_verified_parent_with_two_known_omissions=p["parent_verified_checks"] == 24
        and p["parent_known_unresolved"] == ["current_two_sources_actually_cited", "current_citation_excerpts"],
        no_primary_generation_repeated=p["primary_generation_replays"] == 0
        and p["generation"] == []
        and p["generation_diagnostics"] == [],
        actual_original_omission_kept=original["citations"] == []
        and "[[cite:" + original["namespace"] + ":" not in original["reply"],
        baseline_is_preserved_actual_answer=hashlib.sha256(original["reply"].encode()).hexdigest()
        == f["saved_answer_sha256"]
        == original["reply_sha256"],
        actual_current_authority_503=p["authoritative_source_count"] == 503
        and p["parent_source_unchanged"]
        and p["cloned_sources_unchanged"],
        exactly_one_actual_annotation=len(calls) == len(p["annotation_adapter_requests"]) == 1
        and request.get("max_tokens") == 512
        and all(c["request"].get("max_tokens") != 1024 for c in calls),
        actual_current_pro_http200=call.get("http_status") == 200
        and response.get("model") == "deepseek-v4-pro"
        and response.get("choices", [{}])[0].get("finish_reason") == "stop",
        full_original_answer_reaches_annotation=payload.get("answer") == original["reply"],
        all_current_admitted_bodies_reach_annotation=payload.get("sources")
        == [{"key": s["key"], "body": s["body"]} for s in p["admitted_source_bodies"]],
        annotation_has_no_fact_rewrite_field=isinstance(annotation, dict)
        and set(annotation) == {"citations"}
        and all(isinstance(a, dict) and set(a) == {"key", "quote"} for a in annotations),
        recovered_only_exact_supported_spans=recovered["attempted"]
        and recovered["status"] == "recovered_exact_spans"
        and bool(citations)
        and all(
            c["answer_excerpt"] == c["evidence_quote"]
            and c["answer_excerpt"] in original["reply"]
            and c["evidence_quote"] in sources[c["source_id"]]["body"]
            for c in citations
        ),
        each_span_identifies_one_current_source=bool(citations)
        and all(sum(c["evidence_quote"] in s["body"] for s in sources.values()) == 1 for c in citations),
        actual_both_missing_current_sources_bound={"doc_1_chunk_0", "doc_503_chunk_0"}.issubset(
            {c["source_id"] for c in citations}
        ),
        metadata_only_from_authoritative_retrieval=bool(citations)
        and all(
            c["source_id"] in docs
            and c["id"] == c["source_id"]
            and c["source_title"] == docs[c["source_id"]]["title"]
            and c["key"] == sources[c["source_id"]]["key"]
            for c in citations
        ),
        original_reply_byte_identical=recovered["reply"] == original["reply"]
        and recovered["reply_sha256"] == original["reply_sha256"],
        source_excerpts_keep_actual_current_paths=bool(citations)
        and all(
            c["evidence_excerpt"].startswith(f"[{f['knowledge_base']}/未分类] {c['source_title']}:")
            for c in citations
            if c["source_id"] in {"doc_1_chunk_0", "doc_503_chunk_0"}
        ),
        actual_annotation_keys_match_bound_metadata=selected == [c["key"] for c in citations] and validated is not None,
        actual_annotation_quotes_match_bound_spans=[first_quotes[key] for key in selected]
        == [c["answer_excerpt"] for c in citations],
        actual_preserved_persisted_503=p["persisted_vector_stats"]["total_documents"]
        == p["persisted_vector_stats"]["index_size"]
        == p["persisted_vector_stats"]["bm25_corpus_size"]
        == p["persisted_valid_chunk_count"]
        == 503,
        actual_database_work_terminal=p["sync_pending_final"] == 0,
    )
